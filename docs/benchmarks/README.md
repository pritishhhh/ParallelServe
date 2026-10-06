# Local benchmark verification

Run on 6 October 2026 on Windows 11, Python 3.12.6, CPU PyTorch 2.14.1, with
16 logical CPUs. The included synthetic CNN was trained with `python model.py`
(8 epochs, seed 42). The [raw results](windows-cpu-1ms.json) include environment,
parameters, model-weight SHA-256 checksum, success/error counts, accuracy, and
batch-size information.

```bash
python benchmarks/load_test.py --concurrency 50 --requests-per-client 20 --workers 4 --warmup 3 --window-ms 1 --output-dir benchmarks/results/verified-1ms
```

| Configuration | Successful / attempted | Throughput (requests/s) | p95 client latency (ms) | Mean batch size |
|---|---:|---:|---:|---:|
| One worker, batching off | 1,000 / 1,000 | 667.45 | 96.506 | 1.000 |
| One worker, batching on | 1,000 / 1,000 | 668.29 | 97.250 | 11.140 |
| Four workers, batching off | 1,000 / 1,000 | 661.13 | 99.223 | 1.000 |
| Four workers, batching on | 1,000 / 1,000 | 650.67 | 96.748 | 2.708 |

All measured responses were successful, with 100% synthetic pattern accuracy.
The runner also completed chart generation and process cleanup. These are short
local smoke measurements (about 1.5 seconds of measured traffic per configuration),
so small throughput differences are not evidence of a general ranking.

The principal conclusion is that this tiny model does not establish a universal
batching or multi-process speedup. HTTP/JSON handling, scheduling, serialization,
and waiting can dominate its compute. Batching demonstrably combines requests,
but throughput gains require a representative model, longer repeated runs, and
controlled client/server hardware. The 1 ms window was a benchmark parameter;
the server's configurable default remains 8 ms.
