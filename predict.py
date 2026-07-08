"""
predict.py  —  Quick test script for the distributed inference server.

Usage:
    python predict.py                      # send 1 random prediction
    python predict.py --n 5               # send 5 predictions
    python predict.py --host 127.0.0.1 --port 8080  # custom host/port
"""

import argparse
import json
import sys

import requests

from sample_input import make_payload

CLASS_NAMES = {
    0: "Horizontal stripes",
    1: "Vertical stripes",
    2: "Diagonal gradient",
    3: "Checkerboard",
}


def predict_one(url: str, rng=None) -> dict:
    payload, true_label = make_payload(rng)
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    result["true_label"] = true_label
    result["true_class"] = CLASS_NAMES[true_label]
    result["predicted_class"] = CLASS_NAMES.get(result.get("prediction", -1), "?")
    result["correct"] = result["prediction"] == true_label
    return result


def main():
    parser = argparse.ArgumentParser(description="Call the /predict endpoint")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8080, type=int, help="Server port (default: 8080)")
    parser.add_argument("--n", default=1, type=int, help="Number of predictions to send (default: 1)")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/predict"

    print(f"\n[INFO] Connecting to server at {url}\n")

    correct = 0
    for i in range(args.n):
        try:
            result = predict_one(url)
        except requests.ConnectionError:
            print(f"[ERROR] Could not connect to {url}")
            print("   Make sure the server is running:  uvicorn server:app --host 127.0.0.1 --port 8080")
            sys.exit(1)

        correct += result["correct"]
        tick = "[OK]" if result["correct"] else "[WRONG]"
        print(
            f"[{i+1}/{args.n}] {tick}  "
            f"Predicted: {result['predicted_class']:22s}  "
            f"Actual: {result['true_class']:22s}  "
            f"Confidence: {result['confidence']:.1%}  "
            f"Latency: {result['total_latency_ms']:.1f} ms  "
            f"Batch size: {result['batch_size']}  "
            f"Worker: {result['worker_id']}"
        )

    if args.n > 1:
        print(f"\n[SUMMARY] Accuracy: {correct}/{args.n} ({correct/args.n:.0%})")


if __name__ == "__main__":
    main()
