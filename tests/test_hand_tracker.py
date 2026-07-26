from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hand_tracker import HandTracker, PointSmoother


def make_landmarks(center_x):
    points = [SimpleNamespace(x=center_x, y=0.5, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=center_x, y=0.72, z=0.0)
    points[4] = SimpleNamespace(x=center_x, y=0.62, z=0.0)
    points[8] = SimpleNamespace(x=center_x, y=0.30, z=0.0)
    points[9] = SimpleNamespace(x=center_x, y=0.50, z=0.0)
    return SimpleNamespace(landmark=points)


class HandTrackerTests(unittest.TestCase):
    def test_mocked_landmarks_produce_panel_and_count_extra_fingers(self):
        processor = Mock()
        processor.process.return_value = SimpleNamespace(
            multi_hand_landmarks=[make_landmarks(0.2), make_landmarks(0.8)]
        )
        tracker = HandTracker(processing_width=320, hands_processor=processor)
        scores = {4: 160.0, 8: 160.0, 12: 150.0, 16: 20.0, 20: 20.0}

        with patch.object(
            HandTracker,
            "_finger_extension_scores",
            return_value=(scores, [4, 8, 12]),
        ):
            hand_count, panel, extra_fingers = tracker.find_fingertips(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )

        self.assertEqual(hand_count, 2)
        self.assertEqual(extra_fingers, 2)
        self.assertEqual(panel.shape, (4, 2))
        self.assertEqual(len(tracker.hand_centers), 2)
        self.assertEqual(len(tracker.tracked_hands), 2)
        self.assertEqual(tracker.tracked_hands[0].landmarks.shape, (21, 2))
        self.assertGreater(tracker.tracking_confidence, 0.8)
        self.assertEqual(
            tracker.latest_snapshot.hands[0].pixel_landmarks.shape,
            (21, 2),
        )

        selected = tracker.panel_for_fingers(((4, 8), (4, 8)))
        self.assertEqual(selected.shape, (4, 2))
        self.assertEqual(tracker.selected_finger_ids, [(4, 8), (4, 8)])

    def test_missing_landmarks_report_zero_hands(self):
        processor = Mock()
        processor.process.return_value = SimpleNamespace(multi_hand_landmarks=None)
        tracker = HandTracker(hands_processor=processor)

        result = tracker.find_fingertips(np.zeros((120, 160, 3), dtype=np.uint8))

        self.assertEqual(result, (0, None, 0))
        self.assertEqual(tracker.tracked_hands, [])
        self.assertEqual(tracker.tracking_confidence, 0.0)

    def test_reuses_tracking_on_alternate_frames_for_smooth_30_fps_loop(self):
        processor = Mock()
        processor.process.return_value = SimpleNamespace(
            multi_hand_landmarks=[make_landmarks(0.2), make_landmarks(0.8)]
        )
        tracker = HandTracker(hands_processor=processor, inference_interval=2)
        scores = {4: 160.0, 8: 160.0, 12: 150.0, 16: 20.0, 20: 20.0}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(
            HandTracker,
            "_finger_extension_scores",
            return_value=(scores, [4, 8, 12]),
        ):
            first = tracker.find_fingertips(frame)
            second = tracker.find_fingertips(frame)

        self.assertEqual(processor.process.call_count, 1)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_inference_interval_one_processes_every_frame(self):
        processor = Mock()
        processor.process.return_value = SimpleNamespace(
            multi_hand_landmarks=[make_landmarks(0.2), make_landmarks(0.8)]
        )
        tracker = HandTracker(hands_processor=processor, inference_interval=1)
        scores = {4: 160.0, 8: 160.0, 12: 150.0, 16: 20.0, 20: 20.0}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(
            HandTracker,
            "_finger_extension_scores",
            return_value=(scores, [4, 8, 12]),
        ):
            tracker.find_fingertips(frame)
            tracker.find_fingertips(frame)

        self.assertEqual(processor.process.call_count, 2)

    def test_tracking_grace_retains_geometry_but_marks_it_stale(self):
        now = [10.0]
        processor = Mock()
        processor.process.side_effect = [
            SimpleNamespace(
                multi_hand_landmarks=[
                    make_landmarks(0.2),
                    make_landmarks(0.8),
                ]
            ),
            SimpleNamespace(multi_hand_landmarks=None),
            SimpleNamespace(multi_hand_landmarks=None),
        ]
        tracker = HandTracker(
            hands_processor=processor,
            inference_interval=1,
            tracking_grace_seconds=0.5,
            clock=lambda: now[0],
        )
        scores = {4: 160.0, 8: 160.0, 12: 150.0, 16: 20.0, 20: 20.0}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(
            HandTracker,
            "_finger_extension_scores",
            return_value=(scores, [4, 8, 12]),
        ):
            first = tracker.find_fingertips(frame)
            original_confidence = tracker.tracking_confidence
            now[0] = 10.1
            recovered = tracker.find_fingertips(frame)
            recovery_confidence = tracker.tracking_confidence
            now[0] = 10.6
            expired = tracker.find_fingertips(frame)

        np.testing.assert_array_equal(recovered[1], first[1])
        self.assertTrue(tracker.snapshot_stale)
        self.assertGreater(recovery_confidence, 0.0)
        self.assertLess(recovery_confidence, original_confidence)
        self.assertEqual(expired, (0, None, 0))
        self.assertEqual(tracker.tracked_hands, [])

    def test_one_euro_filter_smooths_stationary_motion_more_than_fast_motion(self):
        slow = PointSmoother(min_cutoff=1.0, beta=0.1)
        fast = PointSmoother(min_cutoff=1.0, beta=0.1)
        origin = np.zeros((1, 2), dtype=np.float32)
        slow.update(origin, timestamp=0.0)
        fast.update(origin, timestamp=0.0)

        slow_output = slow.update(
            np.array([[1.0, 0.0]], dtype=np.float32),
            timestamp=1.0 / 30.0,
        )
        fast_output = fast.update(
            np.array([[100.0, 0.0]], dtype=np.float32),
            timestamp=1.0 / 30.0,
        )

        self.assertGreater(fast_output[0, 0] / 100.0, slow_output[0, 0])

    def test_can_replace_tasks_processor_with_legacy_backend(self):
        tasks_processor = Mock()
        tasks_processor.backend_name = "tasks-live-stream"
        legacy_processor = Mock(spec=["process", "close"])
        tracker = HandTracker(
            hands_processor=tasks_processor,
            inference_interval=1,
        )

        with patch.object(
            tracker,
            "_create_hands_processor",
            return_value=legacy_processor,
        ):
            changed = tracker.switch_to_legacy_backend()

        self.assertTrue(changed)
        self.assertIs(tracker._hands, legacy_processor)
        self.assertEqual(tracker.inference_interval, 2)
        self.assertTrue(tracker._owns_hands_processor)
        tasks_processor.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
