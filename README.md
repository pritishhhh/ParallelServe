# ParallelServe

A single-host, multi-process PyTorch inference server with dynamic batching,
bounded request admission, least-loaded routing, and automatic worker recovery.
The included CNN classifies synthetic 28×28 images into four geometric patterns.

## Architecture

```text
HTTP POST /predict
        │ validate 784 finite pixels in [0, 1]
        ▼
FastAPI async endpoint ── await result Future (no thread per request)
        │
        ▼
Dispatcher ── bounded admission + atomic least-loaded assignment
        │          │          │
        ▼          ▼          ▼
Worker 0       Worker 1    Worker N    (separate spawned processes)
model          model       model
task/result    task/result task/result queues
        │
        ▼
collect up to MAX_BATCH_SIZE within BATCH_WINDOW_MS
skip expired requests → one batched inference → correlated results

Supervisor: readiness tracking → crash/stall detection → cleanup → replacement
```

Workers become eligible for routing only after loading their models. Routing,
assignment, and outstanding-work accounting share a lock, including during worker
replacement. Async HTTP requests wait on result futures rather than occupying a
large executor pool.

## Quick start

Requires Python 3.11 or newer. Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
python model.py
uvicorn server:app --host 127.0.0.1 --port 8000
```

For CPU-only environments, install the CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

In another terminal:

```bash
python predict.py --n 5
curl http://127.0.0.1:8000/health
```

To change settings on Bash:

```bash
NUM_WORKERS=2 BATCH_WINDOW_MS=4 uvicorn server:app --port 8000
```

On Windows PowerShell:

```powershell
$env:NUM_WORKERS = "2"
$env:BATCH_WINDOW_MS = "4"
uvicorn server:app --port 8000
```

Run one Uvicorn process: `NUM_WORKERS` already controls inference parallelism.
Adding Uvicorn `--workers` creates another full model pool per HTTP process and
multiplies memory consumption and configured admission limits.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `NUM_WORKERS` | `4` | Positive number of inference processes |
| `ENABLE_BATCHING` | `1` | Exactly `0` or `1` |
| `MAX_BATCH_SIZE` | `16` | Positive maximum requests per batch |
| `BATCH_WINDOW_MS` | `8` | Finite, nonnegative batching wait; `0` drains immediately available work |
| `MAX_PENDING_REQUESTS` | `256` | Maximum queued/executing requests across the pool |
| `REQUEST_TIMEOUT_S` | `5` | Finite, positive HTTP prediction deadline |
| `STARTUP_TIMEOUT_S` | `30` | Finite, positive model-loading deadline |
| `WORKER_TIMEOUT_S` | `30` | Finite, positive time without results while a worker has outstanding work before replacing it |
| `TORCH_THREADS` | `1` | Positive PyTorch compute threads per worker; also used for sample training |
| `MODEL_PATH` | `model_weights.pt` beside `model.py` | Path to model weights; relative paths resolve from the working directory |

Invalid settings fail startup. The default of one PyTorch compute thread per
worker prevents each worker from creating a full CPU-sized compute pool. Tune
this alongside worker count for your hardware and model. Set `WORKER_TIMEOUT_S`
above the longest legitimate model batch execution time.

## HTTP behavior

`POST /predict` accepts `{"pixels": [...]}` containing exactly 784 numeric,
finite values between zero and one. Predictions return a request ID, predicted
class, confidence, worker ID, actual batch size, and timing fields:

- `queue_wait_ms`: time from admission until the forward pass begins, including
  batching and tensor preparation; excludes inference itself.
- `inference_ms`: time for the batched model forward pass and classification.
- `total_latency_ms`: time within the endpoint after input validation, including
  result delivery; clients should measure their own network latency.

| Response | Meaning |
|---|---|
| `200` | Successful prediction |
| `422` | Invalid input |
| `503` + `Retry-After: 1` | Queue full, worker failure, or no ready workers |
| `504` | Prediction deadline exceeded |
| `500` | Model inference failed; traceback is logged on the worker |

`GET /health` reports ready worker count, configured count, outstanding requests,
and capacity. It returns `200` with `ok` for a full pool, `200` with `degraded`
while at least one worker can serve, or `503` with `unavailable` when none can.
`GET /live` checks the HTTP process independently of model availability.

Requests assigned to a crashed or stalled worker fail promptly with `503`.
Other ready workers continue serving and replacements load before receiving
traffic. There is no automatic replay or guarantee of zero request loss: clients
may retry idempotent predictions. A deadline/cancellation does not interrupt a
forward pass already executing. Outstanding capacity is retained until the
worker consumes that task or the supervisor retires the worker, so repeated
timeouts cannot accumulate unlimited work. Expired queued tasks skip inference.
Shutdown wakes waiters, stops supervision, and joins or terminates workers.

## Model contract

`model.py` contains a lightweight CNN, a synthetic dataset, and reproducible
sample training. `python model.py` creates local weights, excluded from Git.
Only load weights from a trusted source; the loader uses `weights_only=True`.

To use another classifier:

1. Replace `TinyCNN` and `load_model()` with your architecture and weight loader.
2. Update the request schema and array shaping in `server.py` for its input.
3. Adapt preprocessing and output decoding in `worker._process_batch` when needed.

The current worker expects NumPy inputs stackable into a batch and a CPU PyTorch
model returning finite `[batch, classes]` logits. GPU placement, non-classification
outputs, variable input shapes, and other frameworks require adapter changes.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Tests exercise concurrent result routing, real spawned PyTorch workers, actual
batching, HTTP lifecycle, input validation, overload, cancellation, deadlines,
worker death/stalls, startup failure, recovery, and resource cleanup. Test model
weights are generated in temporary directories; training a model or starting an
external server is not required. GitHub Actions runs on Windows and Linux with
Python 3.11 and 3.12.

## Benchmarks

Train the sample model first, then:

```bash
python benchmarks/load_test.py
# Shorter run:
python benchmarks/load_test.py --concurrency 8 --requests-per-client 5 --workers 2
```

The benchmark starts and cleans up an isolated local server for each of four
configurations: one/multiple workers, batching off/on. It warms up first and
records successful throughput, p50/p95/p99 client latency, failures, accuracy,
mean batch size, parameters, model-weight checksum, and environment information.
Failed HTTP responses and malformed results do not count toward throughput. The benchmark exits with
a nonzero status when measured requests fail.

Results, charts, and per-configuration server logs are written to
`benchmarks/results/` (ignored by Git); choose another location with `--output-dir`.
Batching trades extra waiting time for compute efficiency. Small models, low
concurrency, process/serialization overhead, and distributing traffic across too
many workers can outweigh its benefits. Compare on your actual deployment
hardware before choosing settings.

## Scope and next steps

This is serving infrastructure for one machine, with local multiprocessing
queues. It does not yet coordinate workers across machines, version multiple
models, or export a full metrics/tracing backend. For a public deployment, add an
authenticated gateway with request/body limits and TLS. Model memory is duplicated
per worker. The queue limit controls inference work, while ingress connections
and request parsing still need limits at the gateway.

See [the engineering review](docs/engineering-review.md) for the findings and
prioritized follow-up work.
