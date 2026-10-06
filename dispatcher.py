"""Bounded, thread-safe routing to a supervised pool of inference processes."""

import asyncio
import logging
import math
import multiprocessing as mp
import queue
import threading
import time
import uuid
from concurrent.futures import Future, InvalidStateError

from worker import worker_loop

logger = logging.getLogger(__name__)


class ServiceUnavailable(RuntimeError):
    """The pool is stopping or has no ready workers."""


class Overloaded(ServiceUnavailable):
    """The outstanding-request limit has been reached."""


class InferenceError(RuntimeError):
    """A worker could not process the input."""


class WorkerHandle:
    def __init__(self, worker_id, process, task_queue, result_queue):
        self.worker_id = worker_id
        self.process = process
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.in_flight = 0
        self.ready = False
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at
        self.stop = threading.Event()
        self.listener = None


class Dispatcher:
    def __init__(
        self,
        num_workers=4,
        max_batch_size=16,
        batch_window_ms=8.0,
        enable_batching=True,
        max_pending_requests=256,
        startup_timeout_s=30.0,
        torch_threads=1,
        worker_timeout_s=30.0,
        worker_target=worker_loop,
    ):
        for name, value in (
            ("num_workers", num_workers),
            ("max_batch_size", max_batch_size),
            ("max_pending_requests", max_pending_requests),
            ("torch_threads", torch_threads),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(batch_window_ms) or batch_window_ms < 0:
            raise ValueError("batch_window_ms must be finite and nonnegative")
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be finite and positive")
        if not math.isfinite(worker_timeout_s) or worker_timeout_s <= 0:
            raise ValueError("worker_timeout_s must be finite and positive")
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self.enable_batching = enable_batching
        self.max_pending_requests = max_pending_requests
        self.startup_timeout_s = startup_timeout_s
        self.torch_threads = torch_threads
        self.worker_timeout_s = worker_timeout_s
        self.worker_target = worker_target
        # Replacements start from a monitor thread: never fork a threaded
        # process (including the PyTorch runtime), even on Unix.
        self.ctx = mp.get_context("spawn")
        self.status_queue = self.ctx.Queue()
        self.workers = {}
        self._pending = {}  # request_id -> (Future, WorkerHandle)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._next_worker_id = 0
        self.monitor_thread = None
        try:
            for _ in range(num_workers):
                self._spawn_worker()
            self._await_initial_ready()
            self.monitor_thread = threading.Thread(
                target=self._health_monitor, name="parallelserve-monitor", daemon=True
            )
            self.monitor_thread.start()
        except BaseException:
            self.shutdown()
            raise

    def _spawn_worker(self):
        # Caller holds _lock once monitoring begins.
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        task_queue = self.ctx.Queue(maxsize=self.max_pending_requests)
        result_queue = self.ctx.Queue(maxsize=self.max_pending_requests)
        process = self.ctx.Process(
            target=self.worker_target,
            args=(
                worker_id,
                task_queue,
                result_queue,
                self.status_queue,
                self.max_batch_size,
                self.batch_window_ms,
                self.enable_batching,
                self.torch_threads,
            ),
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            self._close_queue(task_queue)
            self._close_queue(result_queue)
            raise
        handle = WorkerHandle(worker_id, process, task_queue, result_queue)
        self.workers[worker_id] = handle
        handle.listener = threading.Thread(
            target=self._result_listener,
            args=(handle,),
            name=f"parallelserve-results-{worker_id}",
            daemon=True,
        )
        handle.listener.start()

    @staticmethod
    def _complete(future, *, result=None, error=None):
        try:
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)
        except InvalidStateError:
            pass  # Cancellation can race with IPC completion.

    def _result_listener(self, handle):
        while not handle.stop.is_set():
            try:
                result = handle.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                break
            with self._lock:
                entry = self._pending.pop(result["request_id"], None)
                if entry is None:
                    continue
                future, owner = entry
                owner.in_flight -= 1
                owner.last_progress_at = time.monotonic()
                if "error" in result:
                    self._complete(future, error=InferenceError("Model inference failed"))
                elif result.get("expired"):
                    self._complete(future, error=TimeoutError("Prediction deadline exceeded"))
                else:
                    result["worker_id"] = handle.worker_id
                    self._complete(future, result=result)

    def _read_status(self, timeout=0):
        try:
            msg = self.status_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            handle = self.workers.get(msg[1])
            if handle is not None and msg[0] == "ready":
                handle.ready = True
        return msg

    def _await_initial_ready(self):
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            msg = self._read_status(timeout=min(0.1, max(0, deadline - time.monotonic())))
            if msg and msg[0] == "crash":
                raise RuntimeError(f"Worker {msg[1]} failed to load the model:\n{msg[2]}")
            if any(not h.process.is_alive() for h in self.workers.values()):
                raise RuntimeError("An inference worker exited during startup")
            if all(h.ready for h in self.workers.values()):
                return
        raise RuntimeError("Inference workers did not become ready before startup timeout")

    def _fail_worker_locked(self, handle, error):
        handle.ready = False
        handle.stop.set()
        for request_id, (future, owner) in list(self._pending.items()):
            if owner is handle:
                del self._pending[request_id]
                handle.in_flight -= 1
                self._complete(future, error=error)

    def _health_monitor(self):
        while not self._stop.wait(0.1):
            while self._read_status() is not None:
                pass
            retired = []
            with self._lock:
                if self._stop.is_set():
                    break
                for worker_id, handle in list(self.workers.items()):
                    stalled = (
                        not handle.ready
                        and time.monotonic() - handle.started_at > self.startup_timeout_s
                    )
                    stalled = stalled or (
                        handle.ready
                        and handle.in_flight > 0
                        and time.monotonic() - handle.last_progress_at > self.worker_timeout_s
                    )
                    if not handle.process.is_alive() or stalled:
                        del self.workers[worker_id]
                        self._fail_worker_locked(
                            handle,
                            ServiceUnavailable(f"Worker {worker_id} exited; retry the request"),
                        )
                        retired.append(handle)
            for handle in retired:
                logger.warning("Replacing failed worker %s", handle.worker_id)
                self._retire_worker(handle)
            # Back off after failure, including repeated model-load failures.
            if retired and self._stop.wait(1.0):
                break
            with self._lock:
                if self._stop.is_set():
                    break
                while len(self.workers) < self.num_workers:
                    try:
                        self._spawn_worker()
                    except Exception:
                        logger.exception("Could not start replacement worker")
                        break

    def submit(self, array, timeout_s=5.0):
        """Return a Future without holding an HTTP executor thread.

        Cancelled/timed-out requests retain capacity until consumed, preventing
        repeated timeouts from bypassing backpressure.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        request_id = uuid.uuid4().hex
        future = Future()
        with self._lock:
            if self._stop.is_set():
                raise ServiceUnavailable("Dispatcher is shutting down")
            if len(self._pending) >= self.max_pending_requests:
                raise Overloaded("Inference queue is full; retry later")
            ready = [h for h in self.workers.values() if h.ready and h.process.is_alive()]
            if not ready:
                raise ServiceUnavailable("No inference workers are ready")
            handle = min(ready, key=lambda h: h.in_flight)
            self._pending[request_id] = (future, handle)
            if handle.in_flight == 0:
                handle.last_progress_at = time.monotonic()
            handle.in_flight += 1
            now = time.monotonic()
            try:
                handle.task_queue.put_nowait((request_id, array, now, now + timeout_s))
            except (queue.Full, OSError, ValueError) as exc:
                del self._pending[request_id]
                handle.in_flight -= 1
                raise ServiceUnavailable("Could not enqueue inference request") from exc
        return future

    def predict(self, array, timeout_s=5.0):
        future = self.submit(array, timeout_s)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError:
            future.cancel()
            raise

    async def predict_async(self, array, timeout_s=5.0):
        future = self.submit(array, timeout_s)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout_s)
        finally:
            if not future.done():
                future.cancel()

    def worker_count(self):
        with self._lock:
            return sum(h.ready and h.process.is_alive() for h in self.workers.values())

    def stats(self):
        with self._lock:
            return {
                "workers": self.worker_count(),
                "configured_workers": self.num_workers,
                "pending_requests": len(self._pending),
                "max_pending_requests": self.max_pending_requests,
            }

    @staticmethod
    def _close_queue(ipc_queue):
        # Joining a feeder after a crash can block on unread payloads.
        ipc_queue.cancel_join_thread()
        ipc_queue.close()

    def _retire_worker(self, handle):
        handle.stop.set()
        handle.process.join(timeout=0.2)
        if handle.process.is_alive():
            handle.process.terminate()
            handle.process.join(timeout=2)
        if handle.process.is_alive():
            handle.process.kill()
            handle.process.join(timeout=2)
        handle.listener.join(timeout=1)
        self._close_queue(handle.task_queue)
        self._close_queue(handle.result_queue)

    def shutdown(self):
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
            handles = list(self.workers.values())
            self.workers.clear()
            for handle in handles:
                self._fail_worker_locked(handle, ServiceUnavailable("Dispatcher is shutting down"))
                try:
                    handle.task_queue.put_nowait(None)
                except (queue.Full, OSError, ValueError):
                    pass
        if self.monitor_thread is not None:
            self.monitor_thread.join()
        for handle in handles:
            self._retire_worker(handle)
        self._close_queue(self.status_queue)
