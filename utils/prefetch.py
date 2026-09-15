"""Background batch prefetcher: overlaps numpy sampling + H2D copy with GPU compute.

Place at: utils/prefetch.py
"""

import queue
import threading

import jax


class BatchPrefetcher:
    """Sample + device_put batches on a background thread.

    `sample_fn` runs on the worker thread and must return a pytree of
    numpy arrays (e.g. one (K, B, ...) super-batch). The worker pushes
    `jax.device_put`-ed batches into a small queue; `get()` pops one.
    With capacity>=2 the sampling/copy of batch i+1 overlaps the GPU
    compute of batch i.

    Dataset swaps must go through `swap()`, which stops the worker,
    installs the new sample_fn, and restarts — discarding any queued
    batches from the old dataset.
    """

    def __init__(self, sample_fn, capacity: int = 2):
        self._sample_fn = sample_fn
        self._capacity = capacity
        self._start()

    def _start(self):
        self._q = queue.Queue(maxsize=self._capacity)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            batch = jax.device_put(self._sample_fn())
            while not self._stop.is_set():
                try:
                    self._q.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def get(self):
        return self._q.get()

    def close(self):
        self._stop.set()
        try:  # drain so a worker blocked on put() can exit
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)

    def swap(self, new_sample_fn):
        """Replace the sampling source (e.g. after a dataset swap)."""
        self.close()
        self._sample_fn = new_sample_fn
        self._start()