import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from dispatcher import Dispatcher
from model import TinyCNN
from server import app
from worker import _process_batch, worker_loop


@pytest.fixture
def weights(tmp_path, monkeypatch):
    path = tmp_path / "weights.pt"
    torch.manual_seed(42)
    torch.save(TinyCNN().state_dict(), path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    return path


def test_real_spawn_batching(weights):
    pool = Dispatcher(num_workers=2, batch_window_ms=50, max_batch_size=8)
    try:
        array = np.zeros((1, 28, 28), dtype=np.float32)
        futures = [pool.submit(array) for _ in range(40)]
        results = [f.result(timeout=10) for f in futures]
        assert len({r["request_id"] for r in results}) == 40
        assert len({r["worker_id"] for r in results}) == 2
        assert any(r["batch_size"] > 1 for r in results)
        assert all(1 <= r["batch_size"] <= 8 and 0 <= r["confidence"] <= 1 for r in results)
        assert pool.stats()["pending_requests"] == 0
    finally:
        pool.shutdown()


def test_real_http_lifespan(weights, monkeypatch):
    monkeypatch.setenv("NUM_WORKERS", "2")
    with TestClient(app) as client:
        assert client.get("/health").json()["workers"] == 2
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(
                executor.map(
                    lambda _: client.post("/predict", json={"pixels": [0.5] * 784}), range(24)
                )
            )
        assert all(r.status_code == 200 for r in responses)
        assert len({r.json()["request_id"] for r in responses}) == 24
    assert app.state.dispatcher is None


def test_missing_weights_fails_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(RuntimeError):
        Dispatcher(num_workers=1)


def test_queue_wait_excludes_inference_time(monkeypatch):
    import queue
    from types import SimpleNamespace

    import worker

    class Model:
        def __call__(self, x):
            return torch.zeros((len(x), 4))

    results = queue.Queue()
    ticks = iter([1.1, 1.6])
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    _process_batch(Model(), [("a", np.zeros((1, 28, 28)), 1, 6)], results)
    result = results.get()
    assert result["inference_ms"] == 500
    assert result["queue_wait_ms"] == 100


def test_worker_skips_expired_requests_and_flushes_shutdown(monkeypatch):
    import queue

    import worker

    calls = []
    monkeypatch.setattr(worker, "load_model", lambda: object())
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(worker, "_process_batch", lambda model, batch, results: calls.append(batch))
    tasks, results, status = queue.Queue(), queue.Queue(), queue.Queue()
    now = time.monotonic()
    tasks.put(("expired", None, now - 1, now - 0.5))
    tasks.put(("valid", None, now, now + 5))
    tasks.put(None)
    worker_loop(0, tasks, results, status, batch_window_ms=0)
    assert results.get_nowait() == {"request_id": "expired", "expired": True}
    assert [item[0] for batch in calls for item in batch] == ["valid"]
