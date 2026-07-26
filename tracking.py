"""Backend-neutral tracking contracts and MediaPipe Tasks live-stream support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
import threading
import time

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # Allows tests and legacy fallback on unsupported Python versions.
    mp = None


@dataclass(frozen=True)
class SnapshotHand:
    """Normalized detector output for one hand."""

    normalized_landmarks: np.ndarray
    pixel_landmarks: np.ndarray | None = None
    world_landmarks: np.ndarray | None = None
    handedness: str | None = None
    handedness_score: float = 0.0


@dataclass(frozen=True)
class TrackingSnapshot:
    """Timestamped detector result shared by all tracking backends."""

    frame_id: int
    captured_at: float
    result_at: float
    inference_seconds: float
    hands: tuple[SnapshotHand, ...]
    sequence: int = 0

    def age_seconds(self, now: float | None = None) -> float:
        current = time.perf_counter() if now is None else now
        return max(0.0, current - self.captured_at)


class TrackingBackend(Protocol):
    """Minimal processor contract consumed by :class:`HandTracker`."""

    backend_name: str
    latest_snapshot: TrackingSnapshot | None
    dropped_submissions: int

    def set_frame_context(self, frame_id: int, captured_at: float) -> None:
        """Attach capture metadata to the next submitted image."""

    def process(self, rgb_frame: np.ndarray):
        """Submit/process an RGB frame and return the newest compatible result."""

    def close(self) -> None:
        """Release backend resources."""


def _landmark_namespace(points: np.ndarray) -> SimpleNamespace:
    landmarks = [
        SimpleNamespace(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        for point in points
    ]
    return SimpleNamespace(landmark=landmarks)


def _classification_namespace(label: str | None, score: float) -> SimpleNamespace:
    category = SimpleNamespace(label=label or "", category_name=label or "", score=score)
    return SimpleNamespace(classification=[category])


class AsyncTasksHandsProcessor:
    """Adapt Tasks Hand Landmarker LIVE_STREAM output to the legacy result shape.

    Only one image is allowed in flight. Calls made while inference is busy are
    counted and discarded, so latency cannot grow through an input queue.
    """

    backend_name = "tasks-live-stream"

    def __init__(
        self,
        model_path: str | Path,
        sensitivity: float = 0.55,
        landmarker_factory=None,
        mp_module=None,
        clock=time.perf_counter,
    ):
        self.model_path = Path(model_path)
        self.sensitivity = float(np.clip(sensitivity, 0.10, 0.95))
        self._mp = mp if mp_module is None else mp_module
        self._landmarker_factory = landmarker_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = False
        self._inflight = False
        self._inflight_image = None
        self._frame_context = (0, self._clock())
        self._inflight_context = None
        self._last_timestamp_ms = -1
        self._sequence = 0
        self._latest_result = SimpleNamespace(
            multi_hand_landmarks=None,
            multi_hand_world_landmarks=None,
            multi_handedness=None,
        )
        self.latest_snapshot: TrackingSnapshot | None = None
        self.dropped_submissions = 0
        self.accepted_submissions = 0
        self._landmarker = self._create_landmarker()

    @classmethod
    def is_available(cls, model_path: str | Path, mp_module=None) -> bool:
        module = mp if mp_module is None else mp_module
        return (
            module is not None
            and hasattr(module, "tasks")
            and Path(model_path).is_file()
        )

    def _create_landmarker(self):
        if self._landmarker_factory is not None:
            return self._landmarker_factory(self._handle_result)
        if self._mp is None:
            raise RuntimeError("MediaPipe is not installed.")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {self.model_path}. "
                "Run tools/download_models.py first."
            )
        options = self._mp.tasks.vision.HandLandmarkerOptions(
            base_options=self._mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=self._mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=self.sensitivity,
            min_hand_presence_confidence=self.sensitivity,
            min_tracking_confidence=self.sensitivity,
            result_callback=self._handle_result,
        )
        return self._mp.tasks.vision.HandLandmarker.create_from_options(options)

    def set_frame_context(self, frame_id: int, captured_at: float) -> None:
        with self._lock:
            self._frame_context = (int(frame_id), float(captured_at))

    def process(self, rgb_frame: np.ndarray):
        with self._lock:
            if self._closed:
                return self._latest_result
            if self._inflight:
                self.dropped_submissions += 1
                return self._latest_result
            submitted_at = self._clock()
            frame_id, captured_at = self._frame_context
            timestamp_ms = max(
                self._last_timestamp_ms + 1, int(round(submitted_at * 1000.0))
            )
            self._last_timestamp_ms = timestamp_ms
            self._inflight = True
            self._inflight_context = (
                frame_id,
                captured_at,
                submitted_at,
                timestamp_ms,
            )
            self.accepted_submissions += 1

        try:
            if self._mp is None or self._landmarker_factory is not None:
                image = rgb_frame
            else:
                image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
                )
            with self._lock:
                self._inflight_image = image
            self._landmarker.detect_async(image, timestamp_ms)
        except Exception:
            with self._lock:
                self._inflight = False
                self._inflight_image = None
                self._inflight_context = None
            raise

        with self._lock:
            return self._latest_result

    @staticmethod
    def _points_array(landmarks) -> np.ndarray:
        return np.asarray(
            [[point.x, point.y, point.z] for point in landmarks],
            dtype=np.float32,
        )

    @staticmethod
    def _category_value(category, *names, default=None):
        for name in names:
            value = getattr(category, name, None)
            if value is not None:
                return value
        return default

    def _handle_result(self, result, _output_image, timestamp_ms: int) -> None:
        result_at = self._clock()
        with self._lock:
            context = self._inflight_context
        if context is None:
            return
        frame_id, captured_at, submitted_at, expected_timestamp = context
        if int(timestamp_ms) != int(expected_timestamp):
            return

        normalized_sets = list(getattr(result, "hand_landmarks", None) or [])
        world_sets = list(getattr(result, "hand_world_landmarks", None) or [])
        handedness_sets = list(getattr(result, "handedness", None) or [])
        snapshot_hands = []
        adapted_hands = []
        adapted_world = []
        adapted_handedness = []

        for index, landmarks in enumerate(normalized_sets):
            normalized = self._points_array(landmarks)
            world = (
                self._points_array(world_sets[index])
                if index < len(world_sets)
                else None
            )
            categories = handedness_sets[index] if index < len(handedness_sets) else []
            category = categories[0] if categories else None
            label = (
                str(
                    self._category_value(
                        category, "category_name", "display_name", default=""
                    )
                )
                if category is not None
                else None
            )
            score = (
                float(self._category_value(category, "score", default=0.0))
                if category is not None
                else 0.0
            )
            snapshot_hands.append(
                SnapshotHand(
                    normalized_landmarks=normalized.copy(),
                    world_landmarks=None if world is None else world.copy(),
                    handedness=label or None,
                    handedness_score=score,
                )
            )
            adapted_hands.append(_landmark_namespace(normalized))
            if world is not None:
                adapted_world.append(_landmark_namespace(world))
            adapted_handedness.append(_classification_namespace(label, score))

        adapted = SimpleNamespace(
            multi_hand_landmarks=adapted_hands or None,
            multi_hand_world_landmarks=adapted_world or None,
            multi_handedness=adapted_handedness or None,
        )
        with self._lock:
            self._sequence += 1
            self.latest_snapshot = TrackingSnapshot(
                frame_id=frame_id,
                captured_at=captured_at,
                result_at=result_at,
                inference_seconds=max(0.0, result_at - submitted_at),
                hands=tuple(snapshot_hands),
                sequence=self._sequence,
            )
            self._latest_result = adapted
            self._inflight = False
            self._inflight_image = None
            self._inflight_context = None

    def adjust_sensitivity(self, value: float) -> None:
        """Rebuild the Tasks graph with updated confidence thresholds."""
        updated = float(np.clip(value, 0.10, 0.95))
        if updated == self.sensitivity:
            return
        with self._lock:
            if self._inflight:
                return
            old_landmarker = self._landmarker
            self.sensitivity = updated
        old_landmarker.close()
        self._landmarker = self._create_landmarker()

    def reset(self) -> None:
        with self._lock:
            self._latest_result = SimpleNamespace(
                multi_hand_landmarks=None,
                multi_hand_world_landmarks=None,
                multi_handedness=None,
            )
            self.latest_snapshot = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            landmarker = self._landmarker
        landmarker.close()
