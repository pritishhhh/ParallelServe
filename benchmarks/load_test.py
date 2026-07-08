"""
load_test.py

Benchmarks the serving system under several configurations by
launching real uvicorn server subprocesses (same code, different env
vars) and hammering each with concurrent requests:

  A. 1 worker,  batching OFF   (naive baseline: one process, no batching)
  B. 1 worker,  batching ON    (isolates the effect of batching alone)
  C. N workers, batching OFF   (isolates the effect of parallel workers alone)
  D. N workers, batching ON    (full system)

For each configuration we fire CONCURRENCY concurrent clients, each
sending NUM_REQUESTS_PER_CLIENT sequential requests, and record:
  - throughput (requests/sec)
  - p50 / p95 / p99 latency (ms)

Produces benchmark_results.json and a bar chart PNG.
"""

import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_input import random_sample  # noqa: E402

HOST = "127.0.0.1"
PORT = 8100
BASE_URL = f"http://{HOST}:{PORT}"
CONCURRENCY = 50
NUM_REQUESTS_PER_CLIENT = 6
NUM_WORKERS_MULTI = 4
STARTUP_WAIT_S = 6
REQUEST_TIMEOUT_S = 15


def start_server(num_workers: int, enable_batching: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["NUM_WORKERS"] = str(num_workers)
    env["ENABLE_BATCHING"] = "1" if enable_batching else "0"
    env["MAX_BATCH_SIZE"] = "16"
    env["BATCH_WINDOW_MS"] = "8.0"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        cwd=project_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_ready(timeout_s: float = STARTUP_WAIT_S + 20):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def stop_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _client_task(n_requests: int):
    latencies = []
    for _ in range(n_requests):
        pixels, _ = random_sample()
        start = time.perf_counter()
        try:
            r = requests.post(
                f"{BASE_URL}/predict", json={"pixels": pixels}, timeout=REQUEST_TIMEOUT_S
            )
            r.raise_for_status()
            r.json()
        except Exception as e:
            latencies.append(None)
            continue
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def run_load(concurrency: int, requests_per_client: int):
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_client_task, requests_per_client) for _ in range(concurrency)]
        all_latencies = []
        for f in futures:
            all_latencies.extend(f.result())
    wall_s = time.perf_counter() - start

    ok_latencies = [l for l in all_latencies if l is not None]
    failed = len(all_latencies) - len(ok_latencies)
    total = len(all_latencies)

    if not ok_latencies:
        return {
            "throughput_rps": 0.0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "failed": failed,
            "total": total,
            "wall_s": wall_s,
        }

    arr = np.array(ok_latencies)
    return {
        "throughput_rps": round(len(ok_latencies) / wall_s, 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "failed": failed,
        "total": total,
        "wall_s": round(wall_s, 2),
    }


def run_config(name: str, num_workers: int, enable_batching: bool):
    print(f"\n=== Config: {name} (workers={num_workers}, batching={enable_batching}) ===")
    proc = start_server(num_workers, enable_batching)
    try:
        if not wait_for_ready():
            raise RuntimeError(f"Server for config '{name}' failed to start")
        # brief warmup so first-call overhead doesn't skew results
        _client_task(2)
        result = run_load(CONCURRENCY, NUM_REQUESTS_PER_CLIENT)
        print(json.dumps(result, indent=2))
        return result
    finally:
        stop_server(proc)
        time.sleep(1.5)  # let the port free up before the next config


def main():
    configs = [
        ("1_worker_no_batching", 1, False),
        ("1_worker_with_batching", 1, True),
        (f"{NUM_WORKERS_MULTI}_workers_no_batching", NUM_WORKERS_MULTI, False),
        (f"{NUM_WORKERS_MULTI}_workers_with_batching", NUM_WORKERS_MULTI, True),
    ]

    results = {}
    for name, workers, batching in configs:
        results[name] = run_config(name, workers, batching)

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    _plot(results)


def _plot(results: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    throughputs = [results[n]["throughput_rps"] for n in names]
    p95s = [results[n]["p95_ms"] or 0 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#9aa5b1", "#5c9ded", "#f2a154", "#3fb27f"]

    axes[0].bar(range(len(names)), throughputs, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    axes[0].set_ylabel("Throughput (requests/sec)")
    axes[0].set_title(f"Throughput under {CONCURRENCY} concurrent clients")
    for i, v in enumerate(throughputs):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(range(len(names)), p95s, color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    axes[1].set_ylabel("p95 latency (ms)")
    axes[1].set_title(f"p95 latency under {CONCURRENCY} concurrent clients")
    for i, v in enumerate(p95s):
        axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_chart.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    main()
