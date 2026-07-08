"""
server.py

FastAPI HTTP frontend for the distributed inference service.
Configurations are loaded from environment variables.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dispatcher import Dispatcher
from model import IMG_SIZE

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
ENABLE_BATCHING = os.environ.get("ENABLE_BATCHING", "1") == "1"
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
BATCH_WINDOW_MS = float(os.environ.get("BATCH_WINDOW_MS", "8.0"))

app = FastAPI(title="Distributed Model-Serving System")
dispatcher: Dispatcher | None = None


class PredictRequest(BaseModel):
    # Flat list of IMG_SIZE * IMG_SIZE floats in [0, 1].
    pixels: list[float]


@app.on_event("startup")
def _startup():
    global dispatcher

    # dispatcher.predict() is a blocking call. We use a custom ThreadPoolExecutor
    # sized to the maximum expected concurrency (256) to prevent the default 
    # asyncio thread pool from bottlenecking the async event loop during IPC waits.
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=256))

    print(
        f"[server] starting dispatcher: workers={NUM_WORKERS} "
        f"batching={ENABLE_BATCHING} max_batch={MAX_BATCH_SIZE} "
        f"window_ms={BATCH_WINDOW_MS}"
    )
    dispatcher = Dispatcher(
        num_workers=NUM_WORKERS,
        max_batch_size=MAX_BATCH_SIZE,
        batch_window_ms=BATCH_WINDOW_MS,
        enable_batching=ENABLE_BATCHING,
    )


@app.on_event("shutdown")
def _shutdown():
    if dispatcher:
        dispatcher.shutdown()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "workers": dispatcher.worker_count() if dispatcher else 0,
        "batching_enabled": ENABLE_BATCHING,
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if len(req.pixels) != IMG_SIZE * IMG_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {IMG_SIZE * IMG_SIZE} pixel values, got {len(req.pixels)}",
        )
    array = np.array(req.pixels, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE)

    loop = asyncio.get_event_loop()
    # Run the blocking dispatcher.predict() in a separate thread
    # to avoid blocking the asyncio event loop.
    def _run():
        return dispatcher.predict(array)

    start = time.perf_counter()
    result = await loop.run_in_executor(None, _run)
    result["total_latency_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    return result
