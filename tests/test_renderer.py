from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from renderer import MODE_SIZES, OverlayRenderer


def closed_capture(*_args, **_kwargs):
    capture = Mock()
    capture.isOpened.return_value = False
    return capture


class RendererTests(unittest.TestCase):
    def test_overlay_keeps_inverted_camera_visible_inside_panel_only(self):
        frame = np.full((40, 40, 3), (10, 20, 30), dtype=np.uint8)
        overlay = np.full((12, 12, 4), (100, 120, 140, 255), dtype=np.uint8)
        quadrilateral = np.array(
            [[10, 10], [29, 10], [29, 29], [10, 29]], dtype=np.float32
        )

        OverlayRenderer._warp_and_blend_roi(
            frame,
            overlay,
            quadrilateral,
            opacity=1.0,
            invert_camera=True,
        )

        expected = np.array((150, 160, 169), dtype=np.uint8)
        np.testing.assert_array_equal(frame[20, 20], expected)
        np.testing.assert_array_equal(frame[2, 2], (10, 20, 30))

    def test_overlay_uses_normal_camera_until_inversion_is_enabled(self):
        frame = np.full((40, 40, 3), (10, 20, 30), dtype=np.uint8)
        overlay = np.full((12, 12, 4), (100, 120, 140, 255), dtype=np.uint8)
        quadrilateral = np.array(
            [[10, 10], [29, 10], [29, 29], [10, 29]], dtype=np.float32
        )

        OverlayRenderer._warp_and_blend_roi(
            frame, overlay, quadrilateral, opacity=0.5
        )

        np.testing.assert_array_equal(frame[20, 20], (55, 70, 85))

    @patch("renderer.cv2.VideoCapture", side_effect=closed_capture)
    def test_missing_video_assets_render_visible_feedback(self, _capture):
        with tempfile.TemporaryDirectory() as folder:
            renderer = OverlayRenderer(folder)
            frame = np.zeros((120, 180, 3), dtype=np.uint8)
            panel = np.array(
                [[20, 20], [160, 20], [160, 100], [20, 100]], dtype=np.float32
            )

            rendered = renderer.render(frame, panel, 0.0)

        self.assertFalse(rendered)
        self.assertGreater(int(frame.max()), 0)

    @patch("renderer.cv2.VideoCapture", side_effect=closed_capture)
    def test_opacity_is_clamped_during_init_and_adjustment(self, _capture):
        renderer = OverlayRenderer(Path("unused"), opacity=5.0)
        self.assertEqual(renderer.opacity, 1.0)
        renderer.adjust_opacity(-20.0)
        self.assertEqual(renderer.opacity, 0.10)

    @patch("renderer.cv2.VideoCapture", side_effect=closed_capture)
    def test_inversion_is_off_by_default_and_can_be_toggled(self, _capture):
        renderer = OverlayRenderer(Path("unused"))

        self.assertFalse(renderer.inversion_enabled)
        self.assertTrue(renderer.toggle_inversion())
        self.assertTrue(renderer.inversion_enabled)
        self.assertFalse(renderer.toggle_inversion())
        self.assertFalse(renderer.inversion_enabled)

    def test_split_effect_levels_keep_dimensions_and_brighten_level_two(self):
        renderer = OverlayRenderer.__new__(OverlayRenderer)
        dark = np.full((180, 480, 4), 20, dtype=np.uint8)
        color = np.full((180, 480, 4), 100, dtype=np.uint8)
        renderer._video_overlay = lambda key: dark.copy() if key == "1" else color.copy()

        level_one = renderer._make_split_overlay(1, 0.0)
        level_two = renderer._make_split_overlay(2, 0.0)

        self.assertEqual(level_one.shape, (180, 480, 4))
        self.assertEqual(level_two.shape, level_one.shape)
        self.assertGreater(level_two[-1, :, :3].mean(), level_one[-1, :, :3].mean())

    def test_bundled_selection_reads_each_overlay_capture(self):
        renderer = OverlayRenderer.__new__(OverlayRenderer)
        renderer.overlay_specs = {
            key: {"label": key, "mode": "wide_strip"} for key in "1234"
        }
        renderer.selected_key = "1"
        renderer._frame_cache = {}
        renderer._render_counter = 0
        renderer._captures = {}
        for key in "1234":
            capture = Mock()
            value = int(key) * 40
            capture.read.return_value = (
                True,
                np.full((30, 60, 3), value, dtype=np.uint8),
            )
            renderer._captures[key] = capture

        means = []
        for key in "1234":
            renderer.select(key)
            means.append(round(renderer.overlay_for_frame(0.0)[:, :, :3].mean()))

        self.assertEqual(means, [40, 80, 120, 160])

    def test_prepared_renderer_output_dimensions_match_each_mode(self):
        source = np.zeros((90, 160, 3), dtype=np.uint8)
        for mode, (width, height) in MODE_SIZES.items():
            with self.subTest(mode=mode):
                output = OverlayRenderer._prepare_for_mode(source, mode)
                self.assertEqual(output.shape[:2], (height, width))

    def test_hand_panel_is_one_solid_shape_with_every_fingertip_corner(self):
        landmarks = np.zeros((21, 2), dtype=np.float32)
        landmarks[:] = (80, 80)
        landmarks[0] = (80, 125)
        landmarks[1] = (65, 100)
        landmarks[5] = (58, 82)
        landmarks[9] = (75, 75)
        landmarks[13] = (92, 80)
        landmarks[17] = (104, 92)
        landmarks[4] = (42, 58)
        landmarks[8] = (55, 25)
        landmarks[12] = (76, 18)
        landmarks[16] = (98, 28)
        landmarks[20] = (116, 48)

        mask = OverlayRenderer.build_hand_mask((150, 160, 3), landmarks, 45.0)
        polygon = OverlayRenderer.fingertip_panel_polygon(landmarks)

        self.assertEqual(mask.shape, (150, 160))
        np.testing.assert_array_equal(polygon[:5], landmarks[[4, 8, 12, 16, 20]])
        self.assertGreater(mask[80, 80], 200)
        self.assertGreater(mask[25, 55], 100)

    def test_hand_render_uses_shared_overlay_without_advancing_again(self):
        renderer = OverlayRenderer.__new__(OverlayRenderer)
        renderer.opacity = 0.8
        renderer._render_counter = 7
        frame = np.zeros((140, 180, 3), dtype=np.uint8)
        overlay = np.full((80, 80, 4), 180, dtype=np.uint8)
        overlay[:, :, 3] = 255
        landmarks = np.tile(np.array([90, 75], dtype=np.float32), (21, 1))
        landmarks[0] = (90, 125)
        landmarks[5] = (70, 80)
        landmarks[9] = (85, 65)
        landmarks[13] = (102, 72)
        landmarks[17] = (115, 88)
        landmarks[4] = (55, 55)
        landmarks[8] = (65, 25)
        landmarks[12] = (85, 18)
        landmarks[16] = (108, 28)
        landmarks[20] = (128, 50)

        rendered = renderer.render_hands(
            frame,
            [(landmarks, 50.0)],
            0.0,
            advance=False,
            overlay=overlay,
        )

        self.assertTrue(rendered)
        self.assertEqual(renderer._render_counter, 7)
        self.assertGreater(int(frame.max()), 0)

    def test_hand_panel_keeps_custom_image_top_toward_fingertips(self):
        renderer = OverlayRenderer.__new__(OverlayRenderer)
        renderer.opacity = 1.0
        frame = np.zeros((150, 180, 3), dtype=np.uint8)
        overlay = np.zeros((100, 100, 4), dtype=np.uint8)
        overlay[:50, :, :3] = (0, 0, 255)
        overlay[50:, :, :3] = (255, 0, 0)
        overlay[:, :, 3] = 255
        landmarks = np.tile(np.array([90, 75], dtype=np.float32), (21, 1))
        landmarks[0] = (90, 125)
        landmarks[2] = (68, 100)
        landmarks[5] = (70, 80)
        landmarks[9] = (90, 65)
        landmarks[13] = (108, 75)
        landmarks[17] = (120, 90)
        landmarks[4] = (52, 58)
        landmarks[8] = (65, 25)
        landmarks[12] = (90, 18)
        landmarks[16] = (113, 28)
        landmarks[20] = (132, 52)

        renderer._render_fingertip_panel(frame, overlay, landmarks, 50.0)

        self.assertGreater(frame[32, 90, 2], frame[32, 90, 0])
        self.assertGreater(frame[105, 90, 0], frame[105, 90, 2])

    def test_hand_render_draws_both_hands_in_the_same_frame(self):
        renderer = OverlayRenderer.__new__(OverlayRenderer)
        renderer.opacity = 0.8
        renderer._render_fingertip_panel = Mock(return_value=True)
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        overlay = np.full((40, 40, 4), 255, dtype=np.uint8)
        landmarks = np.zeros((21, 2), dtype=np.float32)

        rendered = renderer.render_hands(
            frame,
            [(landmarks, 40.0), (landmarks + (100, 0), 40.0)],
            0.0,
            advance=False,
            overlay=overlay,
        )

        self.assertTrue(rendered)
        self.assertEqual(renderer._render_fingertip_panel.call_count, 2)

    @patch("renderer.cv2.VideoCapture", side_effect=closed_capture)
    @patch("renderer.cv2.imread")
    def test_custom_image_filename_becomes_its_hotkey(self, imread, _capture):
        imread.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            custom = Path(folder) / "custom"
            custom.mkdir()
            (custom / "a.png").touch()

            renderer = OverlayRenderer(folder, custom_folder=custom)
            renderer.select("a")

        self.assertTrue(renderer.has_key("a"))
        self.assertEqual(renderer.custom_keys, ("a",))
        self.assertEqual(renderer.selected_key, "a")
        self.assertTrue(renderer.selected_asset_available)
        self.assertEqual(renderer.mode, "wide_strip")

    @patch("renderer.cv2.VideoCapture", side_effect=closed_capture)
    def test_reserved_and_multicharacter_custom_names_are_ignored(self, _capture):
        with tempfile.TemporaryDirectory() as folder:
            custom = Path(folder) / "custom"
            custom.mkdir()
            (custom / "help.mp4").touch()
            (custom / "i.mp4").touch()
            (custom / "q.mp4").touch()

            renderer = OverlayRenderer(folder, custom_folder=custom)

        self.assertFalse(renderer.has_key("help"))
        self.assertFalse(renderer.has_key("i"))
        self.assertFalse(renderer.has_key("q"))


if __name__ == "__main__":
    unittest.main()
