"""
dispatcher.py

Manages the pool of worker processes:
  - Load balancing: routes incoming requests to the least-loaded worker.
  - Fault tolerance: monitors worker health and automatically respawns dead workers.
  - Result routing: maps async worker responses back to the correct synchronous HTTP requests.
"""

import multiprocessing as mp
import threading
import time
import uuid
import os

from worker import worker_loop


class _PendingRequest:
    __slots__ = ("event", "result")

    def __init__(self):
        self.event = threading.Event()
        self.result = None


class WorkerHandle:
    def __init__(self, worker_id, process, task_queue, result_queue):
        self.worker_id = worker_id
        self.process = process
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.in_flight = 0
        self.lock = threading.Lock()


class Dispatcher:
    def __init__(
        self,
        num_workers: int = 4,
        max_batch_size: int = 16,
        batch_window_ms: float = 8.0,
        enable_batching: bool = True,
    ):
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self.enable_batching = enable_batching

        # Use 'spawn' on Windows (fork not available) and 'fork' on Unix-like systems.
        if os.name == "nt":
            self.ctx = mp.get_context("spawn")
        else:
            self.ctx = mp.get_context("fork")
        self.status_queue = self.ctx.Queue()

        self.workers: dict[int, WorkerHandle] = {}
        self._pending: dict[str, _PendingRequest] = {}
        self._pending_lock = threading.Lock()
        self._next_worker_id = 0
        self._stop = threading.Event()

        for _ in range(num_workers):
            self._spawn_worker()

        self._await_initial_ready(timeout_s=30)

        self.monitor_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self.monitor_thread.start()

    # ---------------------------------------------------------------
    # Worker lifecycle
    # ---------------------------------------------------------------
    def _spawn_worker(self):
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        task_queue = self.ctx.Queue()
        result_queue = self.ctx.Queue()
        process = self.ctx.Process(
            target=worker_loop,
            args=(
                worker_id,
                task_queue,
                result_queue,
                self.status_queue,
                self.max_batch_size,
                self.batch_window_ms,
                self.enable_batching,
            ),
            daemon=True,
        )
        process.start()
        handle = WorkerHandle(worker_id, process, task_queue, result_queue)
        self.workers[worker_id] = handle

        listener = threading.Thread(
            target=self._result_listener, args=(handle,), daemon=True
        )
        listener.start()
        return worker_id

    def _result_listener(self, handle: "WorkerHandle"):
        """One thread per worker: drains that worker's result_queue and
        wakes up whichever HTTP request (thread) is waiting on each
        result, identified by request_id."""
        while not self._stop.is_set():
            try:
                result = handle.result_queue.get(timeout=0.5)
            except Exception:
                continue
            req_id = result.get("request_id")
            with self._pending_lock:
                pending = self._pending.pop(req_id, None)
            if pending is not None:
                pending.result = result
                pending.event.set()

    def _await_initial_ready(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        remaining_ids = set(self.workers.keys())
        while remaining_ids and time.monotonic() < deadline:
            try:
                msg = self.status_queue.get(timeout=1.0)
            except Exception:
                continue
            if msg[0] == "ready":
                remaining_ids.discard(msg[1])
            elif msg[0] == "crash":
                raise RuntimeError(f"Worker {msg[1]} crashed on startup:\n{msg[2]}")
        if remaining_ids:
            raise RuntimeError(f"Workers {remaining_ids} failed to become ready in time")

    def _health_monitor(self):
        """Detects dead worker processes (crashed, OOM-killed, etc.)
        and transparently respawns them so the service keeps serving
        traffic without manual intervention."""
        while not self._stop.is_set():
            for worker_id, handle in list(self.workers.items()):
                if not handle.process.is_alive():
                    print(f"[dispatcher] worker {worker_id} died, respawning...")
                    del self.workers[worker_id]
                    new_id = self._spawn_worker()
                    print(f"[dispatcher] spawned replacement worker {new_id}")
            time.sleep(0.5)

    # ---------------------------------------------------------------
    # Request handling
    # ---------------------------------------------------------------
    def _pick_worker(self) -> WorkerHandle:
        """Least-loaded load balancing across currently live workers."""
        return min(self.workers.values(), key=lambda w: w.in_flight)

    def predict(self, array, timeout_s: float = 5.0):
        """Submits one request and blocks the calling thread until a
        result arrives or the timeout elapses. Safe to call
        concurrently from many threads (e.g. FastAPI's executor pool)."""
        request_id = str(uuid.uuid4())
        pending = _PendingRequest()
        with self._pending_lock:
            self._pending[request_id] = pending

        handle = self._pick_worker()
        with handle.lock:
            handle.in_flight += 1

        enqueue_time = time.perf_counter()
        handle.task_queue.put((request_id, array, enqueue_time))

        try:
            got = pending.event.wait(timeout=timeout_s)
            if not got:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                raise TimeoutError(f"Prediction timed out after {timeout_s}s")
            result = pending.result
        finally:
            with handle.lock:
                handle.in_flight -= 1

        result["worker_id"] = handle.worker_id
        return result

    def worker_count(self) -> int:
        return len(self.workers)

    def shutdown(self):
        self._stop.set()
        for handle in self.workers.values():
            try:
                handle.task_queue.put(None)
            except Exception:
                pass
        for handle in self.workers.values():
            handle.process.join(timeout=2)
