# ParallelServe — Distributed ML Inference Server

A multi-process, load-balanced inference server for machine learning models. This project provides the serving infrastructure to host models efficiently, demonstrating concepts like dynamic request batching, load balancing, and fault tolerance.

## Architecture

```
                       ┌─────────────────────────┐
   HTTP request  ───▶  │   FastAPI  (async)      │
   POST /predict       │   /predict endpoint     │
                       └────────────┬────────────┘
                                    │  (blocking call, run in thread pool)
                                    ▼
                       ┌─────────────────────────┐
                       │      Dispatcher         │
                       │  - least-loaded routing │
                       │  - health monitor       │
                       │  - result listeners     │
                       └──┬──────┬──────┬────────┘
                          │      │      │  (each worker: own task_queue
                          ▼      ▼      ▼   + own result_queue)
                     ┌────────┐┌────────┐┌────────┐
                     │Worker 0││Worker 1││Worker N│   ◀── separate OS
                     │(process││(process││(process│       processes
                     │ model  ││ model  ││ model  │       (bypasses GIL)
                     │ loaded ││ loaded ││ loaded │
                     └────────┘└────────┘└────────┘
                          │
                          ▼
                  dynamic batching buffers 
                  requests up to `batch_window_ms` 
                  or `max_batch_size`, then
                  runs ONE batched forward pass
```

## Features

1. **Process-level Parallelism**: Workers are separate OS processes, bypassing the Python GIL for CPU-bound inference tasks.
2. **Dynamic Batching**: Buffers incoming requests and runs them as a single batch to maximize hardware utilization (similar to Triton/TorchServe).
3. **Load Balancing**: Routes incoming requests to the worker with the fewest active tasks.
4. **Fault Tolerance**: Background monitoring detects worker crashes and automatically respawns replacements without losing server availability.

## Running the Server

Install dependencies and train the default model:
```bash
pip install -r requirements.txt
python model.py                 # trains and saves the default CNN
```

Start the inference server:
```bash
NUM_WORKERS=4 uvicorn server:app --host 0.0.0.0 --port 8000
```

Send a request:
```bash
curl localhost:8000/health

# Using the provided predict script:
python predict.py --n 5
```

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `NUM_WORKERS` | 4 | Number of worker processes to spawn |
| `ENABLE_BATCHING` | 1 | Set to 0 to disable dynamic batching |
| `MAX_BATCH_SIZE` | 16 | Maximum requests per batch |
| `BATCH_WINDOW_MS` | 8.0 | Max wait time to accumulate a batch |

## Default Model (Sample)

To make this project fully self-contained, a lightweight sample PyTorch model is included out of the box.

- **`model.py`**: Defines a small Convolutional Neural Network (CNN) that classifies synthetic 28x28 grayscale images into 4 distinct geometric patterns (e.g., horizontal stripes, checkerboard).
- Running `python model.py` automatically generates synthetic data, trains the CNN, and saves the learned weights to **`model_weights.pt`**.
- **`sample_input.py`**: A utility script to generate sample image payloads that match the model's expected input shape for testing.

**Note:** This default model is just a placeholder to demonstrate the serving infrastructure. The server is completely agnostic to the model's architecture — **any ML model can be inserted** into this serving layer.

## Swapping in your own Model

The serving infrastructure is decoupled from the ML model. The server treats the model as a black box function.

To serve your own PyTorch model, modify `model.py`:

**1. Replace `TinyCNN` with your model class:**
```python
# In model.py
class MyModel(nn.Module): 
    # your existing architecture
    ...
```

**2. Update `load_model()` to load your weights:**
```python
def load_model():
    model = MyModel() 
    model.load_state_dict(torch.load("your_weights.pth", map_location="cpu"))
    model.eval()
    return model
```

**3. Adjust the input dimensions:**
Update `IMG_SIZE` (or modify `PredictRequest` in `server.py` to match your input format).

Everything else — request queuing, batching, load balancing, and fault tolerance — will continue to work automatically.

## Tests and Benchmarks

- **Fault tolerance**: Run `python test_fault_tolerance.py` to simulate a worker crash and verify auto-recovery.
- **Benchmarking**: Run `python benchmarks/load_test.py` to measure throughput and latency across different configurations (batching vs non-batching, single vs multi-worker).
