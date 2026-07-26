from types import SimpleNamespace
import unittest

import numpy as np

from tracking import AsyncTasksHandsProcessor, SnapshotHand, TrackingSnapshot


class FakeLandmarker:
    def __init__(self, callback):
        self.callback = callback
        self.submissions = []
        self.closed = False

    def detect_async(self, image, timestamp_ms):
        self.submissions.append((image, timestamp_ms))

    def complete(self, result, submission=0):
        image, timestamp_ms = self.submissions[submission]
        self.callback(result, image, timestamp_ms)

    def close(self):
        self.closed = True


def make_task_result():
    points = [
        SimpleNamespace(x=index / 100.0, y=0.5, z=-index / 1000.0)
        for index in range(21)
    ]
    world = [
        SimpleNamespace(x=index / 1000.0, y=0.01, z=-index / 2000.0)
        for index in range(21)
    ]
    category = SimpleNamespace(category_name="Left", score=0.92)
    return SimpleNamespace(
        hand_landmarks=[points],
        hand_world_landmarks=[world],
        handedness=[[category]],
    )


class TrackingContractTests(unittest.TestCase):
    def test_snapshot_age_is_never_negative(self):
        hand = SnapshotHand(np.zeros((21, 3), dtype=np.float32))
        snapshot = TrackingSnapshot(
            frame_id=4,
            captured_at=10.0,
            result_at=10.1,
            inference_seconds=0.1,
            hands=(hand,),
        )

        self.assertEqual(snapshot.age_seconds(9.0), 0.0)
        self.assertAlmostEqual(snapshot.age_seconds(10.25), 0.25)

    def test_tasks_backend_allows_one_inflight_frame_and_adapts_result(self):
        now = [10.0]
        landmarker = None

        def factory(callback):
            nonlocal landmarker
            landmarker = FakeLandmarker(callback)
            return landmarker

        processor = AsyncTasksHandsProcessor(
            "unused.task",
            landmarker_factory=factory,
            clock=lambda: now[0],
        )
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        processor.set_frame_context(7, 9.95)

        empty = processor.process(frame)
        dropped = processor.process(frame)

        self.assertIsNone(empty.multi_hand_landmarks)
        self.assertIsNone(dropped.multi_hand_landmarks)
        self.assertEqual(processor.accepted_submissions, 1)
        self.assertEqual(processor.dropped_submissions, 1)
        self.assertEqual(len(landmarker.submissions), 1)

        now[0] = 10.04
        landmarker.complete(make_task_result())
        snapshot = processor.latest_snapshot

        self.assertEqual(snapshot.frame_id, 7)
        self.assertEqual(snapshot.sequence, 1)
        self.assertAlmostEqual(snapshot.inference_seconds, 0.04)
        self.assertEqual(len(snapshot.hands), 1)
        self.assertEqual(snapshot.hands[0].normalized_landmarks.shape, (21, 3))
        self.assertEqual(snapshot.hands[0].world_landmarks.shape, (21, 3))
        self.assertEqual(snapshot.hands[0].handedness, "Left")
        self.assertAlmostEqual(snapshot.hands[0].handedness_score, 0.92)

        adapted = processor.process(frame)
        self.assertEqual(len(adapted.multi_hand_landmarks), 1)
        self.assertEqual(
            adapted.multi_handedness[0].classification[0].category_name,
            "Left",
        )
        self.assertEqual(processor.accepted_submissions, 2)

        processor.close()
        self.assertTrue(landmarker.closed)


if __name__ == "__main__":
    unittest.main()
