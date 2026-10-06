"""Per-process model loading, deadline-aware dynamic batching and inference."""

import logging
import queue
import signal
import time
import traceback

import numpy as np
import torch

from model import load_model

logger = logging.getLogger(__name__)


def worker_loop(
    worker_id,
    task_queue,
    result_queue,
    status_queue,
    max_batch_size=16,
    batch_window_ms=8.0,
    enable_batching=True,
    torch_threads=1,
):
    # The HTTP/dispatcher process owns graceful shutdown via queue sentinels.
    # Console interrupts on Windows otherwise kill the whole process group.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    try:
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
        model = load_model()
        status_queue.put(("ready", worker_id))
    except Exception:
        status_queue.put(("crash", worker_id, traceback.format_exc()))
        return

    stopping = False
    while not stopping:
        try:
            item = task_queue.get()
        except (KeyboardInterrupt, EOFError):
            break
        if item is None:
            break
        batch = [item]
        if enable_batching:
            deadline = min(time.monotonic() + batch_window_ms / 1000, item[3])
            while len(batch) < max_batch_size:
                remaining = deadline - time.monotonic()
                try:
                    # A zero window still batches requests already queued.
                    nxt = task_queue.get(timeout=max(0, remaining))
                except queue.Empty:
                    break
                if nxt is None:
                    stopping = True
                    break
                batch.append(nxt)
                deadline = min(deadline, nxt[3])
        now = time.monotonic()
        active = []
        for item in batch:
            if item[3] <= now:
                result_queue.put({"request_id": item[0], "expired": True})
            else:
                active.append(item)
        if not active:
            continue
        try:
            _process_batch(model, active, result_queue)
        except Exception:
            logger.exception("Worker %s inference failed", worker_id)
            for item in active:
                result_queue.put({"request_id": item[0], "error": "Model inference failed"})


def _process_batch(model, batch, result_queue):
    x = torch.from_numpy(np.stack([item[1] for item in batch]).astype(np.float32, copy=False))
    start = time.monotonic()
    with torch.inference_mode():
        logits = model(x)
        if logits.ndim != 2 or logits.shape[0] != len(batch) or not torch.isfinite(logits).all():
            raise ValueError("Model must return finite [batch, classes] logits")
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
    infer_ms = (time.monotonic() - start) * 1000
    predictions, confidences = predictions.tolist(), confidences.tolist()
    for i, (req_id, _, enqueue_time, _) in enumerate(batch):
        result_queue.put(
            {
                "request_id": req_id,
                "prediction": predictions[i],
                "confidence": confidences[i],
                "batch_size": len(batch),
                "inference_ms": round(infer_ms, 3),
                "queue_wait_ms": round(max(0, start - enqueue_time) * 1000, 3),
            }
        )
