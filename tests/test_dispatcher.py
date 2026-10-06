import asyncio
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from dispatcher import Dispatcher, InferenceError, Overloaded, ServiceUnavailable
from tests.helpers import failing_worker, fake_worker, silent_worker


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), "Condition was not reached before deadline"


@pytest.fixture
def pool():
    dispatcher = Dispatcher(num_workers=2, worker_target=fake_worker)
    yield dispatcher
    dispatcher.shutdown()


def test_concurrent_routing_and_accounting(pool):
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(
            executor.map(lambda i: pool.predict({"prediction": i, "delay": 0.005}), range(80))
        )
    assert [r["prediction"] for r in results] == list(range(80))
    assert len({r["worker_id"] for r in results}) == 2
    assert pool.stats()["pending_requests"] == 0
    assert all(h.in_flight == 0 for h in pool.workers.values())


def test_async_concurrency(pool):
    async def run():
        return await asyncio.gather(*[pool.predict_async({"prediction": i}) for i in range(50)])

    assert [r["prediction"] for r in asyncio.run(run())] == list(range(50))


def test_backpressure_counts_cancelled_physical_work():
    pool = Dispatcher(num_workers=1, max_pending_requests=1, worker_target=fake_worker)
    try:
        future = pool.submit({"prediction": 1, "delay": 0.3})
        future.cancel()
        with pytest.raises(Overloaded):
            pool.submit({"prediction": 2})
        wait_until(lambda: pool.stats()["pending_requests"] == 0)
        assert pool.predict({"prediction": 3})["prediction"] == 3
    finally:
        pool.shutdown()


def test_timeout_and_late_result_cleanup(pool):
    with pytest.raises(TimeoutError):
        pool.predict({"prediction": 1, "delay": 0.2}, timeout_s=0.01)
    wait_until(lambda: pool.stats()["pending_requests"] == 0)
    assert pool.predict({"prediction": 2})["prediction"] == 2


def test_async_cancellation_cleanup(pool):
    async def run():
        task = asyncio.create_task(pool.predict_async({"prediction": 1, "delay": 0.2}))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    wait_until(lambda: pool.stats()["pending_requests"] == 0)


def test_worker_crash_fails_assigned_requests_and_recovers(pool):
    future = pool.submit({"prediction": 1, "delay": 10}, timeout_s=15)
    with pool._lock:
        owner = next(h for _, h in pool._pending.values())
    owner.process.terminate()
    owner.process.join(timeout=2)
    start = time.monotonic()
    with pytest.raises(ServiceUnavailable):
        future.result(timeout=3)
    assert time.monotonic() - start < 3
    assert pool.predict({"prediction": 2})["prediction"] == 2
    wait_until(lambda: pool.worker_count() == 2)
    assert owner.worker_id not in pool.workers
    assert not owner.listener.is_alive()


def test_shutdown_wakes_waiters_and_stops_threads(pool):
    handles = list(pool.workers.values())
    future = pool.submit({"prediction": 1, "delay": 10})
    pool.shutdown()
    with pytest.raises(ServiceUnavailable):
        future.result(timeout=1)
    assert not pool.monitor_thread.is_alive()
    assert all(not h.process.is_alive() and not h.listener.is_alive() for h in handles)
    with pytest.raises(ServiceUnavailable):
        pool.predict({"prediction": 1})
    pool.shutdown()


@pytest.mark.parametrize("worker_target,timeout", [(failing_worker, 10), (silent_worker, 0.2)])
def test_startup_failure_cleans_children(worker_target, timeout):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError):
        Dispatcher(num_workers=2, worker_target=worker_target, startup_timeout_s=timeout)
    assert {p.pid for p in mp.active_children()} == before


def test_worker_errors_are_exceptions(pool):
    with pytest.raises(InferenceError, match="Model inference failed"):
        pool.predict({"prediction": 1, "error": True})


def test_hung_worker_is_replaced():
    pool = Dispatcher(num_workers=1, worker_target=fake_worker, worker_timeout_s=0.3)
    try:
        future = pool.submit({"prediction": 1, "delay": 60}, timeout_s=10)
        with pytest.raises(ServiceUnavailable):
            future.result(timeout=3)
        wait_until(lambda: pool.worker_count() == 1)
        assert pool.predict({"prediction": 2})["prediction"] == 2
    finally:
        pool.shutdown()


@pytest.mark.parametrize(
    "config",
    [
        {"num_workers": 0},
        {"max_batch_size": -1},
        {"batch_window_ms": float("nan")},
        {"max_pending_requests": 0},
        {"torch_threads": 0},
        {"startup_timeout_s": 0},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        Dispatcher(**config)
