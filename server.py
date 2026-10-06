"""Validated HTTP API with bounded asynchronous inference and explicit readiness."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dispatcher import Dispatcher, InferenceError, ServiceUnavailable
from model import IMG_SIZE

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    batching = os.environ.get("ENABLE_BATCHING", "1")
    if batching not in {"0", "1"}:
        raise ValueError("ENABLE_BATCHING must be 0 or 1")
    timeout = float(os.environ.get("REQUEST_TIMEOUT_S", "5"))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("REQUEST_TIMEOUT_S must be finite and positive")
    app.state.request_timeout_s = timeout
    app.state.dispatcher = await asyncio.to_thread(
        Dispatcher,
        num_workers=int(os.environ.get("NUM_WORKERS", "4")),
        enable_batching=batching == "1",
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "16")),
        batch_window_ms=float(os.environ.get("BATCH_WINDOW_MS", "8")),
        max_pending_requests=int(os.environ.get("MAX_PENDING_REQUESTS", "256")),
        startup_timeout_s=float(os.environ.get("STARTUP_TIMEOUT_S", "30")),
        torch_threads=int(os.environ.get("TORCH_THREADS", "1")),
        worker_timeout_s=float(os.environ.get("WORKER_TIMEOUT_S", "30")),
    )
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.dispatcher.shutdown)
        app.state.dispatcher = None


app = FastAPI(title="ParallelServe", lifespan=lifespan)
Pixel = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Raw JSON NaN/Infinity values must not cause a second serialization error
    # while rendering the validation error. Also avoid echoing entire images.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()
            ]
        },
    )


class PredictRequest(BaseModel):
    pixels: list[Pixel] = Field(min_length=IMG_SIZE * IMG_SIZE, max_length=IMG_SIZE * IMG_SIZE)


@app.get("/health")
async def health():
    dispatcher = getattr(app.state, "dispatcher", None)
    stats = (
        dispatcher.stats()
        if dispatcher
        else {
            "workers": 0,
            "configured_workers": 0,
            "pending_requests": 0,
            "max_pending_requests": 0,
        }
    )
    ready = stats["workers"] > 0
    status = (
        ("ok" if stats["workers"] == stats["configured_workers"] else "degraded")
        if ready
        else "unavailable"
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": status,
            **stats,
            "batching_enabled": dispatcher.enable_batching if dispatcher else False,
        },
    )


@app.get("/live")
async def live():
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    dispatcher = getattr(app.state, "dispatcher", None)
    if dispatcher is None:
        raise HTTPException(503, "Inference service is unavailable")
    array = np.asarray(req.pixels, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE)
    start = time.perf_counter()
    try:
        result = await dispatcher.predict_async(array, app.state.request_timeout_s)
    except ServiceUnavailable as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    except TimeoutError as exc:
        raise HTTPException(504, "Prediction deadline exceeded") from exc
    except InferenceError as exc:
        logger.exception("Inference failed")
        raise HTTPException(500, "Model inference failed") from exc
    result["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return result
