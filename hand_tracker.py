"""MediaPipe helpers for finding and smoothing hand landmark points."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # Allows mocked tracker tests on unsupported Python versions.
    mp = None

from geometry import build_fingertip_quadrilateral
from tracking import SnapshotHand, TrackingSnapshot


# name, fingertip landmark, middle-joint landmark, base-joint landmark
FINGER_SPECS = (
    ("thumb", 4, 3, 2),
    ("index", 8, 6, 5),
    ("middle", 12, 10, 9),
    ("ring", 16, 14, 13),
    ("pinky", 20, 18, 17),
)
FINGERTIP_IDS = tuple(spec[1] for spec in FINGER_SPECS)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
PRIMARY_CORNER_IDS = (THUMB_TIP, INDEX_TIP)


@dataclass(frozen=True)
class TrackedHand:
    """Stable, pixel-space hand data consumed by gestures and renderers."""

    slot: int
    landmarks: np.ndarray
    center: np.ndarray
    palm_center: np.ndarray
    palm_scale: float
    tip_points: dict[int, np.ndarray]
    scores: dict[int, float]
    extended_ids: tuple[int, ...]
    normalized_landmarks: np.ndarray | None = None
    world_landmarks: np.ndarray | None = None
    handedness: str | None = None
    handedness_score: float = 0.0


class HandTracker:
    """Detect up to two stable hands and expose their pixel-space landmarks."""

    def __init__(
        self,
        processing_width=480,
        sensitivity=0.55,
        hands_processor=None,
        inference_interval=2,
        tracking_grace_seconds=0.0,
        max_result_age_seconds=0.20,
        clock=time.perf_counter,
    ):
        self.processing_width = max(160, int(processing_width))
        self.sensitivity = float(np.clip(sensitivity, 0.10, 0.95))
        self.inference_interval = max(1, int(inference_interval))
        self.tracking_grace_seconds = float(
            np.clip(tracking_grace_seconds, 0.0, 2.0)
        )
        self.max_result_age_seconds = max(0.01, float(max_result_age_seconds))
        self._clock = clock
        self._frame_counter = 0
        self.inference_count = 0
        self._last_result = (0, None, 0)
        self._last_valid_result = (0, None, 0)
        self._last_valid_at = None
        self._last_valid_confidence = 0.0
        # All five raw tips are kept for the optional on-screen labels.
        self.tracked_fingertips = []
        self.tracked_hands: list[TrackedHand] = []
        self.selected_finger_ids = []
        self.hand_centers = []
        self.tracking_confidence = 0.0
        self._previous_centers = None
        self._previous_handedness = None
        self._incomplete_detection_frames = 0
        self.latest_snapshot: TrackingSnapshot | None = None
        self.snapshot_age_seconds = 0.0
        self.snapshot_is_new = False
        self.snapshot_stale = False
        self.last_inference_seconds = 0.0
        self.preprocess_seconds = 0.0
        self.hand_swap_count = 0
        self._last_snapshot_sequence = None
        self._owns_hands_processor = hands_processor is None
        self._hands = hands_processor or self._create_hands_processor()

    @property
    def backend_name(self):
        return getattr(self._hands, "backend_name", "legacy-hands")

    @property
    def dropped_submissions(self):
        return int(getattr(self._hands, "dropped_submissions", 0))

    def _create_hands_processor(self):
        if mp is None:
            raise RuntimeError(
                "MediaPipe is not installed. Install requirements.txt with a "
                "MediaPipe-supported Python version."
            )
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            # The lite tracking model is substantially faster and the gestures
            # only need stable 2-D landmarks, not the heavier world model.
            model_complexity=0,
            min_detection_confidence=self.sensitivity,
            min_tracking_confidence=self.sensitivity,
        )

    def adjust_sensitivity(self, amount):
        """Adjust MediaPipe confidence thresholds and return the new value."""
        updated = float(np.clip(self.sensitivity + amount, 0.10, 0.95))
        if updated == self.sensitivity:
            return self.sensitivity
        self.sensitivity = updated
        if self._owns_hands_processor:
            self._hands.close()
            self._hands = self._create_hands_processor()
            self._previous_centers = None
            self._previous_handedness = None
            self._frame_counter = 0
            self._last_result = (0, None, 0)
        elif hasattr(self._hands, "adjust_sensitivity"):
            self._hands.adjust_sensitivity(updated)
            self._previous_centers = None
            self._previous_handedness = None
        return self.sensitivity

    def find_fingertips(self, frame, frame_id=None, captured_at=None):
        """Return ``(hand_count, quadrilateral, extra_fingers)``.

        The compatibility quadrilateral still comes from thumb and index
        fingertips. ``tracked_hands`` exposes every landmark for gesture-driven
        selection and hand-mask rendering.
        """
        now = self._clock()
        captured_at = now if captured_at is None else float(captured_at)
        self._frame_counter += 1
        frame_id = self._frame_counter if frame_id is None else int(frame_id)
        self.snapshot_is_new = False
        if (
            self.tracked_hands
            and (self._frame_counter - 1) % self.inference_interval != 0
        ):
            if self.latest_snapshot is not None:
                self.snapshot_age_seconds = self.latest_snapshot.age_seconds(now)
                self.snapshot_stale = (
                    self.snapshot_age_seconds > self.max_result_age_seconds
                )
            return self._last_result
        self.inference_count += 1

        def finish(result):
            self._last_result = result
            return result

        frame_height, frame_width = frame.shape[:2]

        # MediaPipe landmarks are normalized, so a smaller inference image can
        # still be mapped back to the full camera resolution.
        preprocess_started = self._clock()
        if frame_width > self.processing_width:
            processing_height = round(
                frame_height * self.processing_width / frame_width
            )
            processing_frame = cv2.resize(
                frame,
                (self.processing_width, processing_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            processing_frame = frame

        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.preprocess_seconds = max(0.0, self._clock() - preprocess_started)
        if hasattr(self._hands, "set_frame_context"):
            self._hands.set_frame_context(frame_id, captured_at)
        inference_started = self._clock()
        results = self._hands.process(rgb_frame)
        process_seconds = max(0.0, self._clock() - inference_started)

        backend_snapshot = getattr(self._hands, "latest_snapshot", None)
        if isinstance(backend_snapshot, TrackingSnapshot):
            self.latest_snapshot = backend_snapshot
            self.last_inference_seconds = backend_snapshot.inference_seconds
            self.snapshot_age_seconds = backend_snapshot.age_seconds(now)
            self.snapshot_is_new = (
                backend_snapshot.sequence != self._last_snapshot_sequence
            )
            if self.snapshot_is_new:
                self._last_snapshot_sequence = backend_snapshot.sequence
            self.snapshot_stale = (
                self.snapshot_age_seconds > self.max_result_age_seconds
            )
        else:
            self.last_inference_seconds = process_seconds
            self.snapshot_age_seconds = max(0.0, now - captured_at)
            self.snapshot_is_new = True
            self.snapshot_stale = False

        if not results.multi_hand_landmarks:
            self._note_incomplete_detection()
            return self._reuse_or_clear_tracking(now, finish)

        source_at = (
            backend_snapshot.captured_at
            if isinstance(backend_snapshot, TrackingSnapshot)
            else captured_at
        )
        maximum_geometry_age = max(
            self.max_result_age_seconds, self.tracking_grace_seconds
        )
        if (
            isinstance(backend_snapshot, TrackingSnapshot)
            and now - source_at > maximum_geometry_age
        ):
            self._note_incomplete_detection()
            return self._reuse_or_clear_tracking(now, finish)

        self.tracked_fingertips = []
        self.tracked_hands = []
        self.selected_finger_ids = []
        self.hand_centers = []
        self.tracking_confidence = 0.0

        detected_hands = []
        handedness_sets = list(getattr(results, "multi_handedness", None) or [])
        world_sets = list(
            getattr(results, "multi_hand_world_landmarks", None) or []
        )
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark
            normalized_landmarks = np.asarray(
                [[point.x, point.y, point.z] for point in landmarks],
                dtype=np.float32,
            )
            world_landmarks = None
            if hand_index < len(world_sets):
                world_landmarks = np.asarray(
                    [
                        [point.x, point.y, point.z]
                        for point in world_sets[hand_index].landmark
                    ],
                    dtype=np.float32,
                )
            handedness, handedness_score = self._read_handedness(
                handedness_sets, hand_index
            )
            center = np.array(
                [
                    np.mean([point.x for point in landmarks]) * frame_width,
                    np.mean([point.y for point in landmarks]) * frame_height,
                ],
                dtype=np.float32,
            )
            tip_points = {
                tip_id: np.array(
                    self._to_pixel(landmarks[tip_id], frame_width, frame_height),
                    dtype=np.float32,
                )
                for tip_id in FINGERTIP_IDS
            }
            scores, extended_ids = self._finger_extension_scores(landmarks)
            wrist_point = np.array(
                self._to_pixel(landmarks[WRIST], frame_width, frame_height),
                dtype=np.float32,
            )
            middle_mcp = np.array(
                self._to_pixel(landmarks[MIDDLE_MCP], frame_width, frame_height),
                dtype=np.float32,
            )
            pixel_landmarks = np.array(
                [
                    self._to_pixel(landmark, frame_width, frame_height)
                    for landmark in landmarks
                ],
                dtype=np.float32,
            )
            palm_center = pixel_landmarks[[0, 5, 9, 13, 17]].mean(axis=0)
            detected_hands.append(
                {
                    "average_x": float(center[0]),
                    "center": center,
                    "tip_points": tip_points,
                    "scores": scores,
                    "extended_ids": extended_ids,
                    "palm_scale": float(np.linalg.norm(middle_mcp - wrist_point)),
                    "landmarks": pixel_landmarks,
                    "palm_center": palm_center,
                    "normalized_landmarks": normalized_landmarks,
                    "world_landmarks": world_landmarks,
                    "handedness": handedness,
                    "handedness_score": handedness_score,
                }
            )

        detected_hands = self._assign_hands(detected_hands)
        self.hand_centers = [hand["center"].copy() for hand in detected_hands]
        self.tracked_fingertips = [
            np.array(
                [hand["tip_points"][tip_id] for tip_id in FINGERTIP_IDS],
                dtype=np.float32,
            )
            for hand in detected_hands
        ]
        self.tracked_hands = [
            TrackedHand(
                slot=slot,
                landmarks=hand["landmarks"].copy(),
                center=hand["center"].copy(),
                palm_center=hand["palm_center"].copy(),
                palm_scale=hand["palm_scale"],
                tip_points={
                    tip_id: point.copy()
                    for tip_id, point in hand["tip_points"].items()
                },
                scores=dict(hand["scores"]),
                extended_ids=tuple(hand["extended_ids"]),
                normalized_landmarks=hand["normalized_landmarks"].copy(),
                world_landmarks=(
                    None
                    if hand["world_landmarks"] is None
                    else hand["world_landmarks"].copy()
                ),
                handedness=hand["handedness"],
                handedness_score=hand["handedness_score"],
            )
            for slot, hand in enumerate(detected_hands)
        ]

        hand_count = len(detected_hands)
        self.tracking_confidence = 0.35 * min(hand_count, 2) / 2
        snapshot_metadata = (
            backend_snapshot
            if isinstance(backend_snapshot, TrackingSnapshot)
            else TrackingSnapshot(
                frame_id=frame_id,
                captured_at=captured_at,
                result_at=self._clock(),
                inference_seconds=process_seconds,
                hands=(),
                sequence=self.inference_count,
            )
        )
        self.latest_snapshot = TrackingSnapshot(
            frame_id=snapshot_metadata.frame_id,
            captured_at=snapshot_metadata.captured_at,
            result_at=snapshot_metadata.result_at,
            inference_seconds=snapshot_metadata.inference_seconds,
            hands=tuple(
                SnapshotHand(
                    normalized_landmarks=hand["normalized_landmarks"].copy(),
                    pixel_landmarks=hand["landmarks"].copy(),
                    world_landmarks=(
                        None
                        if hand["world_landmarks"] is None
                        else hand["world_landmarks"].copy()
                    ),
                    handedness=hand["handedness"],
                    handedness_score=hand["handedness_score"],
                )
                for hand in detected_hands
            ),
            sequence=snapshot_metadata.sequence,
        )

        def finish_valid(result):
            self._last_valid_result = result
            self._last_valid_at = source_at
            self._last_valid_confidence = self.tracking_confidence
            return finish(result)

        if hand_count != 2:
            self._note_incomplete_detection()
            return finish_valid((hand_count, None, 0))
        self._incomplete_detection_frames = 0

        first_hand, second_hand = detected_hands
        self.tracking_confidence = self._confidence_for_hands(detected_hands)
        extra_fingers = sum(
            tip_id not in PRIMARY_CORNER_IDS
            for hand in detected_hands
            for tip_id in hand["extended_ids"]
        )

        if not all(self._primary_pair_is_usable(hand) for hand in detected_hands):
            return finish_valid((hand_count, None, extra_fingers))

        center_distance = float(
            np.linalg.norm(second_hand["center"] - first_hand["center"])
        )
        minimum_separation = max(
            24.0,
            0.70 * (first_hand["palm_scale"] + second_hand["palm_scale"]),
        )
        if center_distance < minimum_separation:
            return finish_valid((hand_count, None, extra_fingers))

        self.selected_finger_ids = [PRIMARY_CORNER_IDS, PRIMARY_CORNER_IDS]
        first_pair = [
            first_hand["tip_points"][tip_id] for tip_id in PRIMARY_CORNER_IDS
        ]
        second_pair = [
            second_hand["tip_points"][tip_id] for tip_id in PRIMARY_CORNER_IDS
        ]
        quadrilateral = build_fingertip_quadrilateral(first_pair, second_pair)
        return finish_valid((hand_count, quadrilateral, extra_fingers))

    def panel_for_fingers(self, selected_finger_ids):
        """Build a panel from two selected fingertip IDs on each tracked hand."""
        if len(self.tracked_hands) != 2 or selected_finger_ids is None:
            self.selected_finger_ids = []
            return None
        try:
            first_ids = tuple(selected_finger_ids[0])
            second_ids = tuple(selected_finger_ids[1])
            if (
                len(first_ids) != 2
                or len(second_ids) != 2
                or first_ids[0] == first_ids[1]
                or second_ids[0] == second_ids[1]
                or any(tip_id not in FINGERTIP_IDS for tip_id in first_ids + second_ids)
            ):
                raise ValueError
        except (IndexError, TypeError, ValueError):
            self.selected_finger_ids = []
            return None

        self.selected_finger_ids = [first_ids, second_ids]
        first_pair = [
            self.tracked_hands[0].tip_points[tip_id] for tip_id in first_ids
        ]
        second_pair = [
            self.tracked_hands[1].tip_points[tip_id] for tip_id in second_ids
        ]
        return build_fingertip_quadrilateral(first_pair, second_pair)

    @staticmethod
    def _finger_extension_scores(landmarks):
        """Score straightness for all five fingers, including the thumb."""
        wrist = landmarks[WRIST]
        wrist_point = np.array([wrist.x, wrist.y], dtype=np.float32)
        scores = {}
        extended_ids = []

        for _name, tip_id, joint_id, base_id in FINGER_SPECS:
            tip = landmarks[tip_id]
            joint = landmarks[joint_id]
            base = landmarks[base_id]
            tip_point = np.array([tip.x, tip.y], dtype=np.float32)
            joint_point = np.array([joint.x, joint.y], dtype=np.float32)
            base_point = np.array([base.x, base.y], dtype=np.float32)

            upper = base_point - joint_point
            lower = tip_point - joint_point
            denominator = np.linalg.norm(upper) * np.linalg.norm(lower)
            if denominator <= 1e-6:
                angle = 0.0
            else:
                cosine = np.clip(np.dot(upper, lower) / denominator, -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cosine)))

            tip_distance = float(np.linalg.norm(tip_point - wrist_point))
            joint_distance = float(np.linalg.norm(joint_point - wrist_point))
            distance_ratio = tip_distance / max(joint_distance, 1e-6)

            # This score also provides a fallback when touching/occluded fingers
            # fall just below the strict threshold for a frame or two.
            scores[tip_id] = angle + min(distance_ratio, 1.4) * 25.0
            if angle >= 120.0 and distance_ratio >= 0.94:
                extended_ids.append(tip_id)

        return scores, extended_ids

    def _assign_hands(self, hands):
        """Match hands by motion and confident handedness, not only screen order."""
        if len(hands) != 2:
            return sorted(hands, key=lambda hand: hand["average_x"])

        by_x = sorted(hands, key=lambda hand: hand["average_x"])
        if self._previous_centers is None:
            assigned = by_x
        else:
            swapped = [hands[1], hands[0]]
            direct_cost = self._assignment_cost(hands)
            swapped_cost = self._assignment_cost(swapped)
            assigned = hands if direct_cost <= swapped_cost else swapped

        previous_handedness = self._previous_handedness
        current_handedness = [hand["handedness"] for hand in assigned]
        if previous_handedness is not None:
            for previous, hand in zip(previous_handedness, assigned):
                current = hand["handedness"]
                if (
                    previous
                    and current
                    and previous != current
                    and hand["handedness_score"] >= 0.75
                ):
                    self.hand_swap_count += 1
        self._previous_centers = [hand["center"].copy() for hand in assigned]
        self._previous_handedness = current_handedness
        return assigned

    def _assignment_cost(self, candidate_hands):
        total = 0.0
        previous_handedness = self._previous_handedness or [None, None]
        for hand, previous_center, previous_label in zip(
            candidate_hands, self._previous_centers, previous_handedness
        ):
            total += float(np.linalg.norm(hand["center"] - previous_center))
            current_label = hand["handedness"]
            if previous_label and current_label and previous_label != current_label:
                total += (
                    max(20.0, hand["palm_scale"])
                    * 1.5
                    * hand["handedness_score"]
                )
        return total

    def _note_incomplete_detection(self):
        self._incomplete_detection_frames += 1
        if self._incomplete_detection_frames > 10:
            self._previous_centers = None
            self._previous_handedness = None

    @staticmethod
    def _read_handedness(handedness_sets, index):
        if index >= len(handedness_sets):
            return None, 0.0
        classifications = getattr(
            handedness_sets[index], "classification", handedness_sets[index]
        )
        if not classifications:
            return None, 0.0
        category = classifications[0]
        label = (
            getattr(category, "label", None)
            or getattr(category, "category_name", None)
            or getattr(category, "display_name", None)
        )
        return (str(label) if label else None), float(
            getattr(category, "score", 0.0)
        )

    def _clear_tracking(self):
        self.tracked_fingertips = []
        self.tracked_hands = []
        self.selected_finger_ids = []
        self.hand_centers = []
        self.tracking_confidence = 0.0
        self.snapshot_stale = True

    def _reuse_or_clear_tracking(self, now, finish):
        if (
            self.tracking_grace_seconds > 0.0
            and self._last_valid_at is not None
            and self.tracked_hands
        ):
            age = max(0.0, now - self._last_valid_at)
            if age <= self.tracking_grace_seconds:
                remaining = 1.0 - age / self.tracking_grace_seconds
                self.tracking_confidence = self._last_valid_confidence * remaining
                self.snapshot_age_seconds = age
                self.snapshot_stale = True
                self.snapshot_is_new = False
                return finish(self._last_valid_result)
        self._clear_tracking()
        return finish((0, None, 0))

    def _primary_pair_is_usable(self, hand):
        """Reject fists/random hand poses without breaking a light pinch."""
        threshold_scale = 0.65 + self.sensitivity * 0.65
        return (
            hand["scores"][THUMB_TIP] >= 80.0 * threshold_scale
            and hand["scores"][INDEX_TIP] >= 112.0 * threshold_scale
        )

    def _confidence_for_hands(self, hands):
        threshold_scale = 0.65 + self.sensitivity * 0.65
        ratios = []
        for hand in hands:
            ratios.extend(
                (
                    hand["scores"][THUMB_TIP] / (80.0 * threshold_scale),
                    hand["scores"][INDEX_TIP] / (112.0 * threshold_scale),
                )
            )
        return float(np.clip(min(ratios, default=0.0), 0.0, 1.0))

    @staticmethod
    def _to_pixel(landmark, width, height):
        x = int(np.clip(landmark.x * width, 0, width - 1))
        y = int(np.clip(landmark.y * height, 0, height - 1))
        return x, y

    def close(self):
        self._hands.close()

    def switch_to_legacy_backend(self):
        """Replace the current processor with the lightweight legacy tracker."""
        if self.backend_name == "legacy-hands":
            return False
        replacement = self._create_hands_processor()
        previous = self._hands
        self._hands = replacement
        self._owns_hands_processor = True
        self.inference_interval = 2
        previous.close()
        self.reset()
        return True

    def reset(self):
        """Clear identity, grace, snapshot, and backend recovery state."""
        self._clear_tracking()
        self._previous_centers = None
        self._previous_handedness = None
        self._incomplete_detection_frames = 0
        self._last_result = (0, None, 0)
        self._last_valid_result = (0, None, 0)
        self._last_valid_at = None
        self._last_valid_confidence = 0.0
        self.latest_snapshot = None
        self.snapshot_age_seconds = 0.0
        self.snapshot_is_new = False
        self.snapshot_stale = False
        self._last_snapshot_sequence = None
        if hasattr(self._hands, "reset"):
            self._hands.reset()


class PointSmoother:
    """Timestamp-aware One Euro filter for responsive, low-jitter landmarks."""

    def __init__(
        self,
        smoothing_amount=0.30,
        dead_zone_pixels=3.0,
        min_cutoff=None,
        beta=None,
        derivative_cutoff=1.0,
    ):
        # Keep ``dead_zone_pixels`` for API compatibility; One Euro filtering
        # replaces the old hard dead zone that caused visible sticking.
        self.smoothing_amount = float(np.clip(smoothing_amount, 0.0, 1.0))
        self.dead_zone_pixels = float(dead_zone_pixels)
        self.min_cutoff = (
            0.60 + self.smoothing_amount * 4.0
            if min_cutoff is None
            else max(0.01, float(min_cutoff))
        )
        self.beta = (
            0.005 + self.smoothing_amount * 0.12
            if beta is None
            else max(0.0, float(beta))
        )
        self.derivative_cutoff = max(0.01, float(derivative_cutoff))
        self._previous_raw = None
        self._previous_points = None
        self._previous_derivative = None
        self._last_timestamp = None

    @staticmethod
    def _alpha(cutoff, delta_seconds):
        cutoff = np.asarray(cutoff, dtype=np.float32)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / delta_seconds)

    def update(self, new_points, enabled=True, timestamp=None):
        new_points = np.asarray(new_points, dtype=np.float32)
        timestamp = (
            time.perf_counter()
            if timestamp is None and self._last_timestamp is None
            else (
                self._last_timestamp + 1.0 / 30.0
                if timestamp is None
                else float(timestamp)
            )
        )

        if (
            self._previous_points is None
            or self._previous_points.shape != new_points.shape
        ):
            self._previous_raw = new_points.copy()
            self._previous_points = new_points.copy()
            self._previous_derivative = np.zeros_like(new_points)
            self._last_timestamp = timestamp
            return new_points.copy()

        if not enabled:
            self._previous_raw = new_points.copy()
            self._previous_points = new_points.copy()
            self._previous_derivative = np.zeros_like(new_points)
            self._last_timestamp = timestamp
            return new_points.copy()

        delta_seconds = float(
            np.clip(timestamp - self._last_timestamp, 1.0 / 240.0, 0.25)
        )
        derivative = (new_points - self._previous_raw) / delta_seconds
        derivative_alpha = self._alpha(self.derivative_cutoff, delta_seconds)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._previous_derivative
        )
        speed = np.linalg.norm(filtered_derivative, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        point_alpha = self._alpha(cutoff, delta_seconds)
        output = (
            point_alpha * new_points
            + (1.0 - point_alpha) * self._previous_points
        )

        self._previous_raw = new_points.copy()
        self._previous_points = output
        self._previous_derivative = filtered_derivative
        self._last_timestamp = timestamp
        return output.copy()

    def reset(self):
        self._previous_raw = None
        self._previous_points = None
        self._previous_derivative = None
        self._last_timestamp = None
