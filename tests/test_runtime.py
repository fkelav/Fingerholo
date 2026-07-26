from datetime import datetime
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import threading
import time

import numpy as np

from config import AppConfig
from main import (
    build_runtime_report,
    create_video_writer,
    format_runtime_summary,
    main as run_application,
    open_camera,
    save_runtime_report,
    tasks_backend_is_too_slow,
    wait_for_console_exit,
)
from runtime_pipeline import AsyncVideoWriter, LatestFrameCamera, PerformanceMonitor


class RuntimeTests(unittest.TestCase):
    def test_invalid_camera_returns_closed_fallback_without_crashing(self):
        captures = []

        def factory(*_args):
            capture = Mock()
            capture.isOpened.return_value = False
            captures.append(capture)
            return capture

        camera = open_camera(AppConfig(camera_index=7), capture_factory=factory)

        self.assertFalse(camera.isOpened())
        self.assertEqual(len(captures), 2)
        captures[0].release.assert_called_once()

    def test_packaged_console_pause_runs_after_camera_open_failure(self):
        camera = Mock()
        camera.isOpened.return_value = False

        with (
            patch("main.open_camera", return_value=camera),
            patch("main.wait_for_console_exit") as wait_for_exit,
        ):
            exit_code = run_application(config=AppConfig())

        self.assertEqual(exit_code, 1)
        camera.release.assert_called_once()
        wait_for_exit.assert_called_once_with()

    def test_video_writer_failure_releases_writer_and_returns_none(self):
        failed_writer = Mock()
        failed_writer.isOpened.return_value = False

        writer = create_video_writer(
            np.zeros((48, 64, 3), dtype=np.uint8),
            30.0,
            "output/test.mp4",
            writer_factory=lambda *_args: failed_writer,
        )

        self.assertIsNone(writer)
        failed_writer.release.assert_called_once()

    def test_latest_frame_camera_discards_superseded_frames(self):
        frames = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (1, 2, 3)
        ]
        capture = Mock()
        capture.read.side_effect = [
            (True, frames[0]),
            (True, frames[1]),
            (True, frames[2]),
            (False, None),
        ]
        capture.isOpened.return_value = True
        camera = LatestFrameCamera(capture).start()
        deadline = time.perf_counter() + 1.0
        while not camera._failed and time.perf_counter() < deadline:
            time.sleep(0.001)

        success, frame, frame_id, _captured_at = camera.read_latest(timeout=0.0)
        camera.release()

        self.assertTrue(success)
        self.assertEqual(frame_id, 3)
        np.testing.assert_array_equal(frame, frames[2])
        self.assertEqual(camera.dropped_frames, 2)
        capture.release.assert_called_once()

    def test_async_writer_drops_oldest_queued_frame(self):
        entered = threading.Event()
        unblock = threading.Event()
        written = []

        class SlowWriter:
            def isOpened(self):
                return True

            def write(self, frame):
                if not written:
                    entered.set()
                    unblock.wait(1.0)
                written.append(int(frame[0, 0, 0]))

            def release(self):
                pass

        writer = AsyncVideoWriter(SlowWriter(), queue_size=1)
        writer.write(np.full((2, 2, 3), 1, dtype=np.uint8))
        self.assertTrue(entered.wait(1.0))
        writer.write(np.full((2, 2, 3), 2, dtype=np.uint8))
        writer.write(np.full((2, 2, 3), 3, dtype=np.uint8))
        unblock.set()
        writer.release()

        self.assertEqual(written, [1, 3])
        self.assertEqual(writer.dropped_frames, 1)
        self.assertEqual(writer.written_frames, 2)
        self.assertLessEqual(writer.max_queue_depth, 1)

    def test_performance_monitor_reports_percentiles_and_counters(self):
        monitor = PerformanceMonitor()
        for seconds in (0.001, 0.002, 0.004):
            monitor.add("inference", seconds)
        monitor.increment("dropped_submissions", 2)
        monitor.set_gauge("display_fps", 30.0)

        report = monitor.to_dict()

        self.assertEqual(report["stages"]["inference"]["count"], 3)
        self.assertEqual(report["stages"]["inference"]["p50_ms"], 2.0)
        self.assertAlmostEqual(report["stages"]["inference"]["p95_ms"], 3.8)
        self.assertEqual(report["counters"]["dropped_submissions"], 2)
        self.assertEqual(report["gauges"]["display_fps"], 30.0)

    def test_runtime_report_is_gate_ready_and_persists(self):
        monitor = PerformanceMonitor()
        monitor.add("tracking_result_age", 0.010)
        monitor.add("tracking_result_age", 0.030)
        monitor.increment("displayed_frames", 60)
        config = AppConfig().validate()
        tracker = SimpleNamespace(backend_name="tasks-live-stream")

        report = build_runtime_report(
            monitor,
            config,
            tracker,
            runtime_seconds=2.0,
            process_cpu_seconds=1.0,
        )

        self.assertEqual(report["backend"], "tasks-live-stream")
        self.assertEqual(report["display_fps"], 30.0)
        self.assertAlmostEqual(report["p95_result_age_ms"], 29.0)
        self.assertGreaterEqual(report["average_cpu_percent"], 0.0)
        self.assertEqual(
            report["environment"]["python_version"],
            platform.python_version(),
        )

        with tempfile.TemporaryDirectory() as folder:
            path = save_runtime_report(
                report,
                str(Path(folder) / "report_{timestamp}.json"),
                datetime(2026, 7, 26, 12, 34, 56),
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "report_20260726_123456.json")
        self.assertEqual(loaded["backend"], "tasks-live-stream")

    def test_runtime_report_has_a_readable_console_summary(self):
        monitor = PerformanceMonitor()
        monitor.add("inference", 0.010)
        monitor.add("inference", 0.020)
        monitor.increment("displayed_frames", 1800)
        monitor.increment("camera_dropped_frames", 3)
        monitor.increment("tracking_dropped_submissions", 4)
        monitor.set_gauge("runtime_seconds", 65.2)
        monitor.set_gauge("inference_frequency_hz", 27.5)
        report = build_runtime_report(
            monitor,
            AppConfig().validate(),
            SimpleNamespace(backend_name="tasks-live-stream"),
            runtime_seconds=60.0,
            process_cpu_seconds=12.0,
        )

        summary = format_runtime_summary(report)

        self.assertIn("FINGER HOLOGRAM - PERFORMANCE SUMMARY", summary)
        self.assertIn("Session duration       : 1m 05.2s", summary)
        self.assertIn("Tracking backend       : tasks-live-stream", summary)
        self.assertIn("Average display rate   : 30.0 FPS", summary)
        self.assertIn("Displayed frames              : 1,800", summary)
        self.assertIn("Camera frames dropped         : 3", summary)
        self.assertIn("Hand inference", summary)
        self.assertIn("Full machine-readable metrics", summary)
        self.assertNotIn('"schema_version"', summary)

    def test_packaged_console_waits_for_enter_before_exit(self):
        prompts = []

        waited = wait_for_console_exit(
            input_func=lambda prompt: prompts.append(prompt),
            frozen=True,
        )

        self.assertTrue(waited)
        self.assertEqual(prompts, ["\nPress Enter to close this window..."])

    def test_source_run_does_not_wait_for_console_input(self):
        waited = wait_for_console_exit(
            input_func=Mock(side_effect=AssertionError("input called")),
            frozen=False,
        )

        self.assertFalse(waited)

    def test_packaged_console_handles_detached_standard_input(self):
        waited = wait_for_console_exit(
            input_func=Mock(side_effect=EOFError),
            frozen=True,
        )

        self.assertTrue(waited)

    def test_auto_backend_falls_back_only_after_a_slow_warmup(self):
        self.assertFalse(
            tasks_backend_is_too_slow(2.9, 90, 80, 10)
        )
        self.assertFalse(
            tasks_backend_is_too_slow(3.0, 90, 10, 75)
        )
        self.assertTrue(
            tasks_backend_is_too_slow(3.0, 90, 60, 20)
        )
        self.assertTrue(
            tasks_backend_is_too_slow(3.0, 90, 10, 20)
        )



if __name__ == "__main__":
    unittest.main()
