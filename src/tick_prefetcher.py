"""Overlapped host-to-device copy: a single background thread reads the
next batch (mmap column slicing), pins it, and launches its H2D copy on a
side CUDA stream while the main thread is still computing on the current
batch. This is the standard CUDA double-buffering idiom (a dedicated
stream plus a recorded event the consumer waits on), kept to one
background thread rather than a DataLoader worker pool so its effect is
isolated from the num_workers/pin_memory fix this repo already measured
in bench/bench_dataloader.py.
"""
import queue
import threading

import torch


class BackgroundPrefetcher:
    def __init__(self, dataset, batch_size, device, queue_size=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.queue = queue.Queue(maxsize=queue_size)
        self.idx = 0
        self.n = len(dataset)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            xs, ys = [], []
            for _ in range(self.batch_size):
                x, y = self.dataset[self.idx % self.n]
                self.idx += 1
                xs.append(x)
                ys.append(y)
            xb = torch.stack(xs).pin_memory()
            yb = torch.tensor(ys, dtype=torch.long).pin_memory()
            with torch.cuda.stream(self.stream):
                xb_gpu = xb.to(self.device, non_blocking=True)
                yb_gpu = yb.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
            try:
                self.queue.put((xb_gpu, yb_gpu, event), timeout=5.0)
            except queue.Full:
                if self._stop.is_set():
                    return
                raise

    def next(self):
        xb, yb, event = self.queue.get()
        torch.cuda.current_stream(self.device).wait_event(event)
        return xb, yb

    def stop(self):
        self._stop.set()
        # Drain so the worker thread's blocking put() can observe the stop
        # flag and exit instead of waiting on a full queue forever.
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)
