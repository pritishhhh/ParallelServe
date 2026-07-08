import sys
import os
import requests
import concurrent.futures as cf
from sample_input import make_payload

print("---HEALTH---")
print(requests.get("http://127.0.0.1:8000/health", timeout=5).json())

print("---SINGLE PREDICT---")
payload, label = make_payload()
pixels = payload["pixels"]
r = requests.post("http://127.0.0.1:8000/predict", json={"pixels": pixels}, timeout=10)
print("true label:", label)
print(r.json())

print("---CONCURRENT (20 req, 8 threads)---")


def call(_):
    payload, lbl = make_payload()
    px = payload["pixels"]
    resp = requests.post("http://127.0.0.1:8000/predict", json={"pixels": px}, timeout=10)
    d = resp.json()
    d["true_label"] = lbl
    return d


with cf.ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(call, range(20)))

correct = sum(1 for r in results if r.get("prediction") == r["true_label"])
workers_used = set(r.get("worker_id") for r in results)
batch_sizes = [r.get("batch_size") for r in results]
print("accuracy:", correct, "/20")
print("workers used:", workers_used)
print("batch sizes seen:", batch_sizes)
