import unittest

import cv2
import numpy as np

from geometry import (
    build_fingertip_quadrilateral,
    is_simple_convex_quadrilateral,
    is_valid_quadrilateral,
)
from renderer import OverlayRenderer


FRAME_SHAPE = (720, 1280, 3)


class GeometryTests(unittest.TestCase):
    def test_builds_convex_panel_from_two_thumb_index_pairs(self):
        left = np.array([[150, 310], [150, 170]], dtype=np.float32)
        right = np.array([[620, 300], [620, 180]], dtype=np.float32)

        panel = build_fingertip_quadrilateral(left, right)

        self.assertIsNotNone(panel)
        self.assertTrue(is_valid_quadrilateral(panel, FRAME_SHAPE))
        self.assertGreater(abs(cv2.contourArea(panel)), 50_000)
        np.testing.assert_array_equal(
            panel,
            np.array(
                [[150, 170], [620, 180], [620, 300], [150, 310]],
                dtype=np.float32,
            ),
        )

    def test_pair_builder_chooses_non_crossing_correspondence(self):
        left = np.array([[150, 150], [150, 300]], dtype=np.float32)
        right = np.array([[600, 300], [600, 150]], dtype=np.float32)

        panel = build_fingertip_quadrilateral(left, right)

        self.assertIsNotNone(panel)
        self.assertTrue(is_simple_convex_quadrilateral(panel))

    def test_rejects_bow_tie_and_concave_targets(self):
        bow_tie = np.array(
            [[100, 100], [400, 300], [400, 100], [100, 300]], dtype=np.float32
        )
        concave = np.array(
            [[100, 100], [400, 100], [220, 180], [100, 300]], dtype=np.float32
        )

        self.assertFalse(is_valid_quadrilateral(bow_tie, FRAME_SHAPE))
        self.assertFalse(is_valid_quadrilateral(concave, FRAME_SHAPE))

    def test_keeps_a_deliberately_thin_but_wide_strip(self):
        strip = np.array(
            [[100, 200], [500, 200], [500, 204], [100, 204]], dtype=np.float32
        )

        self.assertTrue(is_valid_quadrilateral(strip, FRAME_SHAPE))

    def test_rejects_clasped_hands_with_no_cross_hand_span(self):
        tiny = np.array(
            [[100, 100], [120, 100], [120, 115], [100, 115]], dtype=np.float32
        )

        self.assertFalse(is_valid_quadrilateral(tiny, FRAME_SHAPE))

    def test_renderer_ignores_invalid_polygon(self):
        frame = np.full((120, 180, 3), 25, dtype=np.uint8)
        original = frame.copy()
        overlay = np.full((30, 60, 4), 255, dtype=np.uint8)
        bow_tie = np.array(
            [[10, 10], [160, 100], [160, 10], [10, 100]], dtype=np.float32
        )

        OverlayRenderer._warp_and_blend_roi(frame, overlay, bow_tie, 0.8)

        np.testing.assert_array_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
