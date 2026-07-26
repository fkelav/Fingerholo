"""Low-latency capture, recording, and runtime performance helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Full, Queue
import statistics
import threading
import time


class LatestFrameCamera:
    """Read a camera continuously and expose only the newest captured frame."""

    def __init__(self, capture, clock=time.perf_counter):
        self._capture = capture
        self._clock = clock
        self._condition = threading.Condition()
        self._running = False
        self._thread = None
        self._frame = None
        self._frame_id = 0
        self._captured_at = 0.0
        self._consumed_frame_id = 0
        self._failed = False
        self.dropped_frames = 0

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="camera-latest-frame", daemon=True
        )
        self._thread.start()
        return self

    def _reader_loop(self):
        while self._running:
            success, frame = self._capture.read()
            captured_at = self._clock()
            with self._condition:
                if not success:
                    self._failed = True
                    self._running = False
                    self._condition.notify_all()
                    return
                if self._frame_id > self._consumed_frame_id:
                    self.dropped_frames += 1
                self._frame = frame
                self._frame_id += 1
                self._captured_at = captured_at
                self._condition.notify_all()

    def read_latest(self, timeout=1.0):
        deadline = self._clock() + max(0.0, float(timeout))
        with self._condition:
            while (
                self._running
                and not self._failed
                and self._frame_id <= self._consumed_frame_id
            ):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._frame_id <= self._consumed_frame_id:
                return False, None, self._frame_id, self._captured_at
            self._consumed_frame_id = self._frame_id
            return True, self._frame, self._frame_id, self._captured_at

    def get(self, property_id):
        return self._capture.get(property_id)

    def isOpened(self):
        return self._capture.isOpened()

    def release(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._capture.release()


class AsyncVideoWriter:
    """Write video on a worker thread using a bounded, drop-oldest queue."""

    def __init__(self, writer, queue_size=2):
        self._writer = writer
        self._queue = Queue(maxsize=max(1, int(queue_size)))
        self._closed = False
        self._error = None
        self.dropped_frames = 0
        self.written_frames = 0
        self.max_queue_depth = 0
        self._thread = threading.Thread(
            target=self._writer_loop, name="video-writer", daemon=True
        )
        self._thread.start()

    def isOpened(self):
        return (
            not self._closed
            and self._error is None
            and self._writer.isOpened()
        )

    @property
    def error(self):
        return self._error

    @property
    def queue_depth(self):
        return self._queue.qsize()

    def write(self, frame):
        if self._closed or self._error is not None:
            return False
        queued_frame = frame.copy()
        try:
            self._queue.put_nowait(queued_frame)
        except Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                pass
            self.dropped_frames += 1
            self._queue.put_nowait(queued_frame)
        self.max_queue_depth = max(self.max_queue_depth, self._queue.qsize())
        return True

    def _writer_loop(self):
        while True:
            frame = self._queue.get()
            try:
                if frame is None:
                    return
                self._writer.write(frame)
                self.written_frames += 1
            except Exception as exc:  # Preserve the failure for the preview thread.
                self._error = exc
                return
            finally:
                self._queue.task_done()

    def release(self):
        if self._closed:
            return
        self._closed = True
        if not self._thread.is_alive():
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except Empty:
                    break
            self._writer.release()
            return
        while True:
            try:
                self._queue.put(None, timeout=0.1)
                break
            except Full:
                continue
        self._thread.join(timeout=5.0)
        self._writer.release()


@dataclass(frozen=True)
class StageSummary:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class PerformanceMonitor:
    """Keep bounded per-stage timing samples and derive percentile summaries."""

    def __init__(self, max_samples=1800, clock=time.perf_counter):
        self._clock = clock
        self._samples = defaultdict(lambda: deque(maxlen=max_samples))
        self.counters = defaultdict(int)
        self.gauges = {}

    @contextmanager
    def measure(self, stage):
        started = self._clock()
        try:
            yield
        finally:
            self.add(stage, self._clock() - started)

    def add(self, stage, seconds):
        self._samples[str(stage)].append(max(0.0, float(seconds)))

    def increment(self, name, amount=1):
        self.counters[str(name)] += int(amount)

    def set_gauge(self, name, value):
        self.gauges[str(name)] = float(value)

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = index - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    def summary(self):
        output = {}
        for stage, values in self._samples.items():
            if not values:
                continue
            output[stage] = StageSummary(
                count=len(values),
                mean_ms=statistics.fmean(values) * 1000.0,
                p50_ms=self._percentile(values, 0.50) * 1000.0,
                p95_ms=self._percentile(values, 0.95) * 1000.0,
                max_ms=max(values) * 1000.0,
            )
        return output

    def to_dict(self):
        return {
            "stages": {
                name: {
                    "count": value.count,
                    "mean_ms": value.mean_ms,
                    "p50_ms": value.p50_ms,
                    "p95_ms": value.p95_ms,
                    "max_ms": value.max_ms,
                }
                for name, value in self.summary().items()
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
        }
