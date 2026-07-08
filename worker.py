"""
worker.py

Worker process implementation for distributed model serving.
Each worker runs in a separate OS process to bypass the Python GIL and achieve true parallelism.
Workers load the model once at startup and process requests using dynamic batching.
"""

import queue
import time
import traceback
import numpy as np
import torch

from model import load_model


def worker_loop(
    worker_id: int,
    task_queue,
    result_queue,
    status_queue,
    max_batch_size: int = 16,
    batch_window_ms: float = 8.0,
    enable_batching: bool = True,
):
    """Main loop run inside each worker process."""
    try:
        model = load_model()
        status_queue.put(("ready", worker_id))
    except Exception:
        status_queue.put(("crash", worker_id, traceback.format_exc()))
        return

    while True:
        batch = []
        try:
            item = task_queue.get()  # blocks until at least one request arrives
        except (KeyboardInterrupt, EOFError):
            break

        if item is None:  # sentinel used to shut the worker down
            break
        batch.append(item)

        if enable_batching:
            deadline = time.monotonic() + (batch_window_ms / 1000.0)
            while len(batch) < max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = task_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    task_queue.put(None)  # let outer loop see shutdown too
                    break
                batch.append(nxt)

        try:
            _process_batch(model, batch, result_queue)
        except Exception:
            err = traceback.format_exc()
            for req_id, _, _ in batch:
                result_queue.put({"request_id": req_id, "error": err})


def _process_batch(model, batch, result_queue):
    """Runs one forward pass over a batch of requests and pushes each
    tagged result onto the worker's shared result_queue."""
    request_ids = [b[0] for b in batch]
    arrays = [b[1] for b in batch]
    enqueue_times = [b[2] for b in batch]

    x = torch.tensor(np.stack(arrays), dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
    infer_ms = (time.perf_counter() - start) * 1000.0

    now = time.perf_counter()
    for i, req_id in enumerate(request_ids):
        result_queue.put(
            {
                "request_id": req_id,
                "prediction": int(preds[i].item()),
                "confidence": float(probs[i, preds[i]].item()),
                "batch_size": len(batch),
                "inference_ms": round(infer_ms, 3),
                "queue_wait_ms": round((now - enqueue_times[i]) * 1000.0, 3),
            }
        )
