# Engineering review

Reviewed the original commit `776c98f` and its server, dispatcher, worker, model,
client scripts, tests, benchmark runner, documentation, and checked-in benchmark
artifacts. Improvements focus on demonstrable correctness and reproducible
measurement before adding more features.

## Findings addressed

| Finding | Original behavior | Change |
|---|---|---|
| Broken benchmark | Syntax error inside readiness polling; missing execution code; nonexistent `random_sample` import | Rebuilt runnable benchmark with warmup, four configurations, logs, and reproducible parameters |
| Concurrent routing race | Worker dictionary changed during selection; selection and counter increment were separate | One lock protects readiness, assignment, request registry, and counters |
| Dead-worker requests | Requests assigned to a dead process waited for their normal timeout | Fail assigned futures when failure is detected; HTTP `503` is retryable |
| Premature routing | Replacement processes received traffic before model initialization completed | Route only to workers reporting ready |
| Unsafe replacement on Unix | Monitor thread used `fork` after threads/PyTorch initialization | Consistent `spawn` process context |
| Unbounded demand | Unbounded queues plus executor work could accumulate | Bounded global admission, bounded IPC queues, explicit overload response |
| Thread per request | HTTP requests occupied a custom 256-thread pool while waiting for IPC | Await correlated result futures directly |
| Timeout accounting | In-flight count decreased even while timed-out work was still queued/executing | Track physical outstanding work through cancellation and late results |
| Stalled processes | Alive-but-stuck workers stayed eligible indefinitely | Replace workers that stop producing results while they have outstanding work |
| Resource leaks | Startup exceptions left earlier workers running; dead listeners/queues were retained; shutdown did not terminate stuck workers | Startup cleanup, per-worker listener stop/join, queue cleanup, bounded joins and forced termination |
| Misleading health | Always `ok`, even without a dispatcher | Readiness based on live, initialized workers; separate process liveness |
| Input validation gaps | Length checked, but nonfinite/out-of-range values reached inference | Validated length and finite `[0,1]` numbers; serialization-safe validation errors |
| Error responses | Timeouts became generic server errors; worker tracebacks could be returned in successful HTTP responses | Explicit `503`/`504`/`500` responses and server-side traceback logging |
| Misleading latency | `queue_wait_ms` included inference time | Stop queue timing before inference starts |
| CPU contention | Each process used PyTorch's default compute thread count | Configurable compute threads, default one per worker |
| Client mismatch | Prediction client defaulted to port 8080 while documentation used 8000 | Consistent port 8000 and request failure handling |
| Inadequate tests | Smoke script performed HTTP calls at import time; crash demo allowed losses without asserting request behavior | Isolated regression suite, real spawned workers, HTTP integration, crash/stall recovery assertions, cross-platform CI |
| Overstated model support | README implied any model could work automatically | Document actual input/output contract and required adapter changes |
| Unsupported performance evidence | Original artifacts lacked sufficient reproduction provenance and the checked-in runner could not generate them | Replace artifacts with a working benchmark; retain new evidence with environment and run parameters |

## Design tradeoffs

Each worker owns its queues and model. This makes request ownership and worker
failure isolation explicit, at the cost of per-worker model memory and a result
listener thread per worker. HTTP concurrency no longer creates one thread per
request. The registry remains bounded even if callers cancel or time out.

Worker failures produce a retryable response rather than replaying a request
automatically. This avoids claiming lossless execution or introducing duplicate
side effects into future model adapters. Surviving workers continue serving.
All replacements wait for model readiness; repeated failures back off.

The stall watchdog observes time since the last result when outstanding work
exists. It cannot distinguish a legitimately slow forward pass from a hang.
Configure its threshold above the maximum legitimate execution time.

## Prioritized future improvements

1. **Measure with a representative model and workload.** The tiny synthetic CNN
   can be dominated by HTTP, serialization, scheduling, and batching-window costs.
   Measure concurrent workloads and tune worker count, batch size/window, compute
   threads, and admission limits together. Keep failure rate beside throughput.
2. **Add operational metrics.** Export queue depth/wait, ready worker count,
   restart count, rejection/timeout rates, inference time, and batch distribution
   to a metrics backend. Add request-correlated structured logs and tracing.
3. **Define a model adapter interface.** Separate validation/preprocessing,
   loading, inference, and output decoding. Version the model artifact and
   response contract before adding hot reload or multiple models.
4. **Add a deployment boundary.** Use an authenticated gateway with TLS, request
   body and connection limits, and rate limiting. The present admission limit
   bounds inference tasks, not every inbound connection or JSON body.
5. **Evaluate queue topology for batching efficiency.** Least-loaded per-worker
   queues can fragment small bursts. Compare a shared scheduling/batching stage
   on representative workloads, including crash ownership and deadline behavior,
   before changing the architecture.
6. **Add GPU or remote workers only with explicit resource management.** Current
   execution is CPU, local queues, and one copied model per process. GPU memory,
   device placement, shared tensors, transport/authentication, and distributed
   retries require a separate design.

## Verification

The regression suite passes locally on Windows/Python 3.12, including real
PyTorch inference and HTTP lifecycle tests. GitHub Actions also verified the
runtime changes on Windows/Linux with Python 3.11/3.12. See the pull request for
the latest checks and [recorded benchmark verification](benchmarks/README.md) for
parameters and results. Synthetic accuracy and short benchmark measurements are
smoke evidence, not a production performance guarantee.
