import json
from pathlib import Path
import tempfile
import unittest

from config import AppConfig, load_config, parse_resolution
from main import configuration_from_args, resolve_output_path


class ConfigTests(unittest.TestCase):
    def test_clamps_opacity_and_smoothing_bounds(self):
        low = AppConfig(opacity=-5, smoothing_amount=-1).validate()
        high = AppConfig(opacity=8, smoothing_amount=2).validate()

        self.assertEqual(low.opacity, 0.10)
        self.assertEqual(low.smoothing_amount, 0.0)
        self.assertEqual(high.opacity, 1.0)
        self.assertEqual(high.smoothing_amount, 1.0)

    def test_loads_json_and_cli_values_override_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(
                json.dumps({"camera_index": 2, "opacity": 0.4}), encoding="utf-8"
            )

            loaded = load_config(path)
            overridden = configuration_from_args(
                [
                    "--config",
                    str(path),
                    "--opacity",
                    "0.8",
                    "--resolution",
                    "640x480",
                    "--performance-output",
                    "artifacts/cli_{timestamp}.json",
                ]
            )

        self.assertEqual(loaded.camera_index, 2)
        self.assertEqual(loaded.opacity, 0.4)
        self.assertEqual(overridden.opacity, 0.8)
        self.assertEqual(overridden.resolution, (640, 480))
        self.assertEqual(
            overridden.performance_output_filename,
            "artifacts/cli_{timestamp}.json",
        )

    def test_rejects_bad_resolution(self):
        with self.assertRaises(ValueError):
            parse_resolution("large")

    def test_validates_tracking_and_recording_pipeline_settings(self):
        config = AppConfig(
            max_tracking_result_age_seconds=-1,
            tracking_backend="TASKS",
            recording_queue_size=100,
            performance_output_filename="artifacts/test_{timestamp}.json",
        ).validate()

        self.assertEqual(config.max_tracking_result_age_seconds, 0.01)
        self.assertEqual(config.tracking_backend, "tasks")
        self.assertEqual(config.recording_queue_size, 16)
        self.assertEqual(
            config.performance_output_filename,
            "artifacts/test_{timestamp}.json",
        )

        with self.assertRaises(ValueError):
            AppConfig(tracking_backend="unknown").validate()
        with self.assertRaises(ValueError):
            AppConfig(performance_output_filename="").validate()

    def test_timestamped_output_name_is_deterministic(self):
        from datetime import datetime

        path = resolve_output_path(
            "output/capture_{timestamp}.mp4", datetime(2026, 7, 22, 9, 8, 7)
        )

        self.assertEqual(path, Path("output/capture_20260722_090807.mp4"))


if __name__ == "__main__":
    unittest.main()
