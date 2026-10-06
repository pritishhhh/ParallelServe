"""Compare single/multiple workers with batching on/off using real HTTP traffic.

Only validated successful predictions contribute to throughput and latency.
Results include failures, accuracy, batching, configuration and environment.
"""

import argparse
import concurrent.futures as cf
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sample_input import make_payload  # noqa: E402


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_server(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False, capture_output=True
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def start_server(workers, batching, args, log):
    env = os.environ.copy()
    env.update(
        NUM_WORKERS=str(workers),
        ENABLE_BATCHING=str(int(batching)),
        MAX_BATCH_SIZE=str(args.batch_size),
        BATCH_WINDOW_MS=str(args.window_ms),
        MAX_PENDING_REQUESTS=str(max(256, args.concurrency)),
        TORCH_THREADS=str(args.torch_threads),
        REQUEST_TIMEOUT_S=str(args.timeout),
        STARTUP_TIMEOUT_S=str(args.startup_timeout),
    )
    port = free_port()
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log,
        stderr=log,
        **options,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + args.startup_timeout + 5
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Benchmark server exited; inspect the server log")
            try:
                response = requests.get(f"{url}/health", timeout=1)
                if response.status_code == 200 and response.json().get("workers") == workers:
                    return proc, url
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.1)
        raise TimeoutError("Benchmark server did not become ready")
    except BaseException:
        stop_server(proc)
        raise


def client_run(url, count, timeout, seed):
    rng = np.random.default_rng(seed)
    latencies, batch_sizes, errors = [], [], {}
    correct = 0
    with requests.Session() as session:
        for _ in range(count):
            payload, label = make_payload(rng)
            start = time.perf_counter()
            try:
                response = session.post(f"{url}/predict", json=payload, timeout=timeout + 2)
                response.raise_for_status()
                result = response.json()
                if (
                    not isinstance(result.get("prediction"), int)
                    or result["prediction"] not in range(4)
                    or not 0 <= result.get("confidence", -1) <= 1
                    or not isinstance(result.get("batch_size"), int)
                    or result["batch_size"] < 1
                ):
                    raise ValueError("Invalid prediction response")
                latencies.append((time.perf_counter() - start) * 1000)
                batch_sizes.append(result["batch_size"])
                correct += result["prediction"] == label
            except (requests.RequestException, ValueError) as exc:
                if isinstance(exc, requests.HTTPError):
                    key = f"http_{exc.response.status_code}"
                else:
                    key = type(exc).__name__
                errors[key] = errors.get(key, 0) + 1
    return latencies, batch_sizes, correct, errors


def measure(url, args):
    # Warm up every process without mixing initialization into measured traffic.
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        warmups = list(
            executor.map(
                lambda i: client_run(url, args.warmup, args.timeout, 1000 + i),
                range(args.concurrency),
            )
        )
        if any(errors for _, _, _, errors in warmups):
            raise RuntimeError("Warmup predictions failed; inspect the server log")
        start = time.perf_counter()
        chunks = list(
            executor.map(
                lambda i: client_run(url, args.requests_per_client, args.timeout, i),
                range(args.concurrency),
            )
        )
        elapsed = time.perf_counter() - start
    latencies = [x for chunk, _, _, _ in chunks for x in chunk]
    batch_sizes = [x for _, chunk, _, _ in chunks for x in chunk]
    errors = {}
    for _, _, _, counts in chunks:
        for key, value in counts.items():
            errors[key] = errors.get(key, 0) + value
    successful = len(latencies)
    attempted = args.concurrency * args.requests_per_client
    return {
        "attempted": attempted,
        "successful": successful,
        "failed": attempted - successful,
        "errors": errors,
        "duration_s": round(elapsed, 3),
        "throughput_rps": round(successful / elapsed, 2),
        **{
            f"p{p}_ms": round(float(np.percentile(latencies, p)), 3) if latencies else None
            for p in (50, 95, 99)
        },
        "accuracy": sum(correct for _, _, correct, _ in chunks) / successful
        if successful
        else None,
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else None,
    }


def plot(results, output_dir):
    # Keep font caches beside generated benchmark artifacts, not in the home directory.
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#9aa5b1", "#5c9ded", "#f2a154", "#3fb27f"]
    for axis, key, title in (
        (axes[0], "throughput_rps", "Successful predictions / second"),
        (axes[1], "p95_ms", "p95 latency (ms)"),
    ):
        values = [results[name][key] or 0 for name in names]
        axis.bar(range(len(names)), values, color=colors)
        axis.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_chart.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests-per-client", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-ms", type=float, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args(argv)
    for name in (
        "concurrency",
        "requests_per_client",
        "workers",
        "batch_size",
        "torch_threads",
        "timeout",
        "startup_timeout",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not np.isfinite(args.window_ms) or args.window_ms < 0 or args.warmup < 1:
        parser.error("--window-ms must be finite and nonnegative; --warmup must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    configurations = [
        ("single_no_batch", 1, False),
        ("single_batch", 1, True),
        ("multi_no_batch", args.workers, False),
        ("multi_batch", args.workers, True),
    ]
    for name, workers, batching in configurations:
        with (args.output_dir / f"{name}.log").open("w") as log:
            proc, url = start_server(workers, batching, args, log)
            try:
                results[name] = {"workers": workers, "batching": batching, **measure(url, args)}
            finally:
                stop_server(proc)
        print(f"{name}: {results[name]}")
    document = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
        },
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": results,
    }
    output = args.output_dir / "benchmark_results.json"
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    plot(results, args.output_dir)
    print(f"Saved results to {output}")
    return 1 if any(r["failed"] for r in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
