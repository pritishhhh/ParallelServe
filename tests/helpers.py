"""Spawn-safe test workers for deterministic lifecycle and routing failures."""

import time


def fake_worker(worker_id, tasks, results, status, *config):
    status.put(("ready", worker_id))
    while True:
        item = tasks.get()
        if item is None:
            return
        request_id, payload, _, deadline = item
        time.sleep(payload.get("delay", 0))
        if time.monotonic() > deadline:
            results.put({"request_id": request_id, "expired": True})
        elif payload.get("error"):
            results.put({"request_id": request_id, "error": "private traceback"})
        else:
            results.put({"request_id": request_id, "prediction": payload["prediction"]})


def failing_worker(worker_id, tasks, results, status, *config):
    status.put(("crash", worker_id, "missing model"))


def silent_worker(worker_id, tasks, results, status, *config):
    time.sleep(60)
