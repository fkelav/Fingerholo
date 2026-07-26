"""Run the real-time finger hologram webcam effect."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

from config import AppConfig, load_config, parse_resolution
from gestures import EffectState, GestureController, GestureKind
from geometry import is_valid_quadrilateral
from hand_tracker import HandTracker, PointSmoother
from renderer import OverlayRenderer, draw_finger_labels, draw_neon_border
from runtime_pipeline import AsyncVideoWriter, LatestFrameCamera, PerformanceMonitor
from tracking import AsyncTasksHandsProcessor


WINDOW_NAME = "Finger Hologram Effect"


def draw_message(frame, text, y=42, color=(255, 255, 255), scale=0.75, x=22):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_panel(frame, lines, origin=(18, 18), width=520, line_height=25):
    """Draw readable text on a translucent panel without hiding the camera."""
    x, y = origin
    height = 20 + line_height * len(lines)
    width = min(width, frame.shape[1] - x - 8)
    height = min(height, frame.shape[0] - y - 8)
    if width <= 20 or height <= 20:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (110, 230, 255), 1)
    for index, (text, color) in enumerate(lines):
        baseline = y + 25 + index * line_height
        if baseline >= y + height:
            break
        cv2.putText(
            frame,
            text,
            (x + 12, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def confidence_bar(frame, confidence, x=22, y=92, width=180):
    confidence = float(np.clip(confidence, 0.0, 1.0))
    color = (80, 220, 80) if confidence >= 0.72 else (50, 190, 255)
    if confidence < 0.4:
        color = (80, 80, 255)
    cv2.rectangle(frame, (x, y), (x + width, y + 12), (45, 45, 45), -1)
    cv2.rectangle(frame, (x, y), (x + round(width * confidence), y + 12), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 12), (220, 220, 220), 1)


def create_video_writer(
    frame, camera_fps, output_path, writer_factory=None
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fps = camera_fps if 1.0 <= camera_fps <= 120.0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    factory = writer_factory or cv2.VideoWriter
    writer = factory(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        return None
    return writer


def open_camera(config, capture_factory=None):
    """Open the selected low-latency Windows webcam, with a portable fallback."""
    factory = capture_factory or cv2.VideoCapture
    camera = factory(config.camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = factory(config.camera_index)

    width, height = config.resolution
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def resolve_project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def create_hand_tracker(config, processor_factory=None):
    """Create the requested backend, falling back only in ``auto`` mode."""
    processor = None
    backend = config.tracking_backend
    model_path = resolve_project_path(config.hand_model_path)
    if backend in {"auto", "tasks"}:
        available = AsyncTasksHandsProcessor.is_available(model_path)
        if available:
            factory = processor_factory or AsyncTasksHandsProcessor
            processor = factory(
                model_path=model_path,
                sensitivity=config.detection_sensitivity,
            )
        elif backend == "tasks":
            raise RuntimeError(
                f"Tasks tracking requires the model at {model_path}. "
                "Run python tools/download_models.py."
            )
        else:
            print(
                "Tasks model unavailable; using legacy CPU tracking. "
                "Run python tools/download_models.py to enable live-stream tracking."
            )
    return HandTracker(
        processing_width=config.processing_width,
        sensitivity=config.detection_sensitivity,
        hands_processor=processor,
        inference_interval=1 if processor is not None else 2,
        tracking_grace_seconds=config.tracking_grace_seconds,
        max_result_age_seconds=config.max_tracking_result_age_seconds,
    )


def tasks_backend_is_too_slow(
    elapsed_seconds,
    submitted_frames,
    dropped_submissions,
    completed_results,
):
    """Decide once whether automatic Tasks tracking should fall back."""
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    submitted_frames = max(int(submitted_frames), 0)
    if elapsed_seconds < 3.0 or submitted_frames < 45:
        return False
    result_hz = completed_results / max(elapsed_seconds, 1e-6)
    drop_ratio = dropped_submissions / max(submitted_frames, 1)
    return result_hz < 18.0 or drop_ratio > 0.50


def resolve_output_path(template, moment=None):
    stamp = (moment or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(str(template).replace("{timestamp}", stamp))


def screenshot_path(output_template, moment=None):
    output_path = resolve_output_path(output_template, moment)
    return output_path.with_name(f"{output_path.stem}_screenshot.png")


def build_runtime_report(
    performance,
    config,
    tracker,
    runtime_seconds,
    process_cpu_seconds,
):
    """Build a gate-ready report that remains comparable across machines."""
    details = performance.to_dict()
    result_age = details["stages"].get("tracking_result_age", {})
    displayed_frames = details["counters"].get("displayed_frames", 0)
    logical_cpu_count = max(1, os.cpu_count() or 1)
    one_core_cpu_percent = (
        max(0.0, float(process_cpu_seconds))
        / max(float(runtime_seconds), 1e-6)
        * 100.0
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "backend": tracker.backend_name,
        "p95_result_age_ms": result_age.get("p95_ms", 0.0),
        "display_fps": displayed_frames / max(float(runtime_seconds), 1e-6),
        "average_cpu_percent": one_core_cpu_percent / logical_cpu_count,
        "process_cpu_percent_one_core": one_core_cpu_percent,
        "tracking_accuracy": None,
        "gesture_accuracy": None,
        "offline_assets": resolve_project_path(config.hand_model_path).is_file(),
        "gpu_active": False,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "logical_cpu_count": logical_cpu_count,
        },
        "configuration": config.to_dict(),
        "performance": details,
    }


def save_runtime_report(report, output_template, moment=None):
    """Write one timestamped runtime report and return its absolute path."""
    path = resolve_project_path(resolve_output_path(output_template, moment))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def format_runtime_summary(report):
    """Format the saved runtime metrics as a compact console summary."""
    performance = report.get("performance", {})
    stages = performance.get("stages", {})
    counters = performance.get("counters", {})
    gauges = performance.get("gauges", {})

    runtime_seconds = max(float(gauges.get("runtime_seconds", 0.0)), 0.0)
    minutes, seconds = divmod(runtime_seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        duration = f"{hours}h {minutes:02d}m {seconds:04.1f}s"
    elif minutes:
        duration = f"{minutes}m {seconds:04.1f}s"
    else:
        duration = f"{seconds:.1f}s"

    lines = [
        "=" * 72,
        "FINGER HOLOGRAM - PERFORMANCE SUMMARY",
        "=" * 72,
        f"Session duration       : {duration}",
        f"Tracking backend       : {report.get('backend', 'unknown')}",
        f"Average display rate   : {float(report.get('display_fps', 0.0)):.1f} FPS",
        (
            "Process CPU usage      : "
            f"{float(report.get('average_cpu_percent', 0.0)):.1f}% "
            "of total CPU capacity"
        ),
        (
            "Tracking result age    : "
            f"{float(report.get('p95_result_age_ms', 0.0)):.1f} ms (p95)"
        ),
        (
            "Completed inference    : "
            f"{float(gauges.get('inference_frequency_hz', 0.0)):.1f} Hz"
        ),
        "",
        "FRAMES AND RELIABILITY",
        f"  Displayed frames              : {int(counters.get('displayed_frames', 0)):,}",
        f"  Camera frames dropped         : {int(counters.get('camera_dropped_frames', 0)):,}",
        (
            "  Tracking submissions dropped  : "
            f"{int(counters.get('tracking_dropped_submissions', 0)):,}"
        ),
        (
            "  Stale tracking results         : "
            f"{int(counters.get('stale_tracking_results', 0)):,}"
        ),
        f"  Hand-swap warnings             : {int(counters.get('hand_swap_warnings', 0)):,}",
    ]

    recording_written = int(counters.get("recording_written_frames", 0))
    recording_dropped = int(counters.get("recording_dropped_frames", 0))
    if recording_written or recording_dropped:
        lines.extend(
            [
                f"  Recording frames written       : {recording_written:,}",
                f"  Recording frames dropped       : {recording_dropped:,}",
            ]
        )
    if counters.get("automatic_legacy_fallback", 0):
        lines.append("  Automatic legacy fallback      : yes")

    if stages:
        labels = {
            "capture_wait": "Camera wait",
            "tracking_submit": "Tracking submit",
            "preprocess": "Preprocessing",
            "inference": "Hand inference",
            "tracking_result_age": "Tracking result age",
            "gestures": "Gesture processing",
            "render": "Effect rendering",
            "hud": "HUD drawing",
            "recording_enqueue": "Recording enqueue",
            "display": "Preview display",
            "frame_total": "Complete frame",
        }
        preferred_order = list(labels)
        stage_names = [
            name for name in preferred_order if name in stages
        ] + sorted(name for name in stages if name not in labels)
        lines.extend(
            [
                "",
                "STAGE TIMINGS (milliseconds)",
                (
                    f"  {'Stage':<22} {'Samples':>7} {'Mean':>9} "
                    f"{'p50':>9} {'p95':>9} {'Max':>9}"
                ),
                "  " + "-" * 68,
            ]
        )
        for name in stage_names:
            values = stages[name]
            label = labels.get(name, name.replace("_", " ").title())
            lines.append(
                f"  {label:<22} {int(values.get('count', 0)):>7,} "
                f"{float(values.get('mean_ms', 0.0)):>9.2f} "
                f"{float(values.get('p50_ms', 0.0)):>9.2f} "
                f"{float(values.get('p95_ms', 0.0)):>9.2f} "
                f"{float(values.get('max_ms', 0.0)):>9.2f}"
            )

    lines.extend(["=" * 72, "Full machine-readable metrics are saved in the JSON report."])
    return "\n".join(lines)


def wait_for_console_exit(input_func=input, frozen=None):
    """Keep a packaged EXE console open long enough to read its final summary."""
    is_frozen = (
        bool(getattr(sys, "frozen", False))
        if frozen is None
        else bool(frozen)
    )
    if not is_frozen:
        return False
    try:
        input_func("\nPress Enter to close this window...")
    except (EOFError, KeyboardInterrupt):
        # Redirected or detached standard input should never block shutdown.
        pass
    return True


def open_fullscreen_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )


def tracking_guidance(
    tracker,
    hand_count,
    frame_shape,
    brightness,
    effect_state,
    panel_visible,
    hand_fill_visible,
):
    if brightness < 48:
        return "LOW LIGHT - face a light source", (60, 90, 255)
    if hand_count == 0:
        return "Show both hands to the camera", (60, 180, 255)
    if hand_count == 1:
        return "One hand found - show the other hand", (60, 180, 255)
    if effect_state.panel_active and len(tracker.hand_centers) == 2 and not panel_visible:
        distance = float(np.linalg.norm(tracker.hand_centers[1] - tracker.hand_centers[0]))
        distance_ratio = distance / max(frame_shape[1], 1)
        if distance_ratio < 0.22:
            return "MOVE HANDS APART", (60, 90, 255)
        if distance_ratio > 0.82:
            return "Bring hands slightly closer", (60, 180, 255)
    if not effect_state.panel_active and not effect_state.hand_fill_active:
        return "Touch four fingertips to select a panel, or bunch both hands", (60, 180, 255)
    if not panel_visible and not hand_fill_visible:
        return "Effects armed - restore the tracked hands", (60, 180, 255)
    return "Effects tracking", (80, 230, 100)


def draw_calibration(frame, tracker, guidance, guidance_color):
    height, width = frame.shape[:2]
    cv2.rectangle(
        frame,
        (round(width * 0.18), round(height * 0.16)),
        (round(width * 0.82), round(height * 0.84)),
        (80, 210, 255),
        2,
    )
    for center in tracker.hand_centers:
        cv2.circle(frame, tuple(np.rint(center).astype(int)), 28, (255, 180, 60), 2)
    draw_message(
        frame,
        f"CALIBRATION: {guidance}",
        y=round(height * 0.13),
        x=max(22, round(width * 0.18)),
        color=guidance_color,
        scale=0.65,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--resolution", type=parse_resolution, metavar="WIDTHxHEIGHT")
    parser.add_argument("--processing-width", type=int)
    parser.add_argument("--opacity", type=float)
    parser.add_argument("--smoothing", type=float, dest="smoothing_amount")
    parser.add_argument("--output", dest="output_filename")
    parser.add_argument("--default-overlay", choices=("1", "2", "3", "4"))
    parser.add_argument("--asset-dir", dest="asset_directory")
    parser.add_argument("--sensitivity", type=float, dest="detection_sensitivity")
    parser.add_argument("--tracking-grace", type=float, dest="tracking_grace_seconds")
    parser.add_argument(
        "--max-result-age",
        type=float,
        dest="max_tracking_result_age_seconds",
    )
    parser.add_argument(
        "--tracking-backend", choices=("auto", "tasks", "legacy")
    )
    parser.add_argument("--hand-model", dest="hand_model_path")
    parser.add_argument(
        "--recording-queue-size", type=int, dest="recording_queue_size"
    )
    parser.add_argument(
        "--performance-output", dest="performance_output_filename"
    )
    background = parser.add_mutually_exclusive_group()
    background.add_argument(
        "--camera-background", action="store_true", dest="record_camera_background"
    )
    background.add_argument(
        "--no-camera-background", action="store_false", dest="record_camera_background"
    )
    parser.set_defaults(record_camera_background=None)
    return parser


def configuration_from_args(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    for name in (
        "camera_index",
        "resolution",
        "processing_width",
        "opacity",
        "smoothing_amount",
        "output_filename",
        "default_overlay",
        "asset_directory",
        "detection_sensitivity",
        "tracking_grace_seconds",
        "max_tracking_result_age_seconds",
        "tracking_backend",
        "hand_model_path",
        "recording_queue_size",
        "performance_output_filename",
        "record_camera_background",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    return config.validate()


def main(config=None, argv=None):
    config = config or configuration_from_args(argv)
    cv2.setUseOptimized(True)
    camera = open_camera(config)

    if not camera.isOpened():
        print(f"Could not open webcam (camera {config.camera_index}).")
        camera.release()
        wait_for_console_exit()
        return 1

    camera_fps = camera.get(cv2.CAP_PROP_FPS)
    try:
        tracker = create_hand_tracker(config)
    except Exception:
        camera.release()
        raise
    camera = LatestFrameCamera(camera).start()
    smoother = PointSmoother(config.smoothing_amount)
    hand_smoothers = [
        PointSmoother(config.smoothing_amount),
        PointSmoother(config.smoothing_amount),
    ]
    gesture_controller = GestureController()
    effect_state = EffectState()
    assets_folder = Path(config.asset_directory)
    if not assets_folder.is_absolute():
        assets_folder = Path(__file__).resolve().parent / assets_folder
    custom_folder = Path(__file__).resolve().parent / "custom"
    custom_folder.mkdir(exist_ok=True)
    overlay_renderer = OverlayRenderer(
        assets_folder, config.opacity, custom_folder=custom_folder
    )
    overlay_renderer.select(config.default_overlay)

    smoothing_enabled = True
    glow_enabled = False
    border_enabled = False
    debug_points_enabled = False
    hud_state = {"visible": True}
    help_enabled = False
    calibration_enabled = False
    manual_split_enabled = False
    record_camera_background = config.record_camera_background
    recording = False
    writer = None
    recording_path = None
    recording_started_at = None
    started_at = time.perf_counter()
    process_cpu_started_at = time.process_time()
    performance = PerformanceMonitor()
    last_corner_signature = None
    fps = 0.0
    previous_frame_at = started_at
    feedback = ""
    feedback_until = 0.0
    automatic_backend_evaluated = (
        config.tracking_backend != "auto"
        or tracker.backend_name != "tasks-live-stream"
    )
    automatic_backend_started_at = started_at
    automatic_tasks_results = 0

    print("Press H for in-app controls and live status. Press Escape or Q to quit.")
    open_fullscreen_window()

    try:
        while True:
            frame_started = time.perf_counter()
            capture_wait_started = frame_started
            success, camera_frame, frame_id, captured_at = camera.read_latest()
            performance.add(
                "capture_wait", time.perf_counter() - capture_wait_started
            )
            if not success:
                print("Could not read a frame from the webcam.")
                break

            camera_frame = cv2.flip(camera_frame, 1)
            tracking_started = time.perf_counter()
            hand_count, _default_points, _extra_finger_count = tracker.find_fingertips(
                camera_frame,
                frame_id=frame_id,
                captured_at=captured_at,
            )
            performance.add("tracking_submit", time.perf_counter() - tracking_started)
            if tracker.snapshot_is_new:
                automatic_tasks_results += 1
                performance.add("preprocess", tracker.preprocess_seconds)
                performance.add("inference", tracker.last_inference_seconds)
                performance.add(
                    "tracking_result_age", tracker.snapshot_age_seconds
                )
                if tracker.snapshot_stale:
                    performance.increment("stale_tracking_results")
            now = time.perf_counter()
            if (
                not automatic_backend_evaluated
                and now - automatic_backend_started_at >= 3.0
            ):
                automatic_backend_evaluated = True
                if tasks_backend_is_too_slow(
                    now - automatic_backend_started_at,
                    tracker.inference_count,
                    tracker.dropped_submissions,
                    automatic_tasks_results,
                ):
                    tracker.switch_to_legacy_backend()
                    smoother.reset()
                    for hand_smoother in hand_smoothers:
                        hand_smoother.reset()
                    performance.increment("automatic_legacy_fallback")
                    feedback = "Tasks tracker too slow; using legacy lite tracker"
                    feedback_until = now + 5.0
                    print(feedback)
            frame_delta = max(now - previous_frame_at, 1e-6)
            instant_fps = 1.0 / frame_delta
            fps = instant_fps if fps == 0.0 else fps * 0.90 + instant_fps * 0.10
            performance.set_gauge("display_fps", fps)
            previous_frame_at = now
            # A sparse RGB sample is enough for the lighting warning and avoids
            # converting the full 720p frame to grayscale every camera tick.
            brightness = float(camera_frame[::8, ::8].mean())

            gesture_started = time.perf_counter()
            gesture_events = (
                gesture_controller.update(
                    tracker.tracked_hands, now, effect_state
                )
                if tracker.snapshot_is_new and not tracker.snapshot_stale
                else []
            )
            performance.add("gestures", time.perf_counter() - gesture_started)
            for event in gesture_events:
                effect_state.apply(event)
                if event.kind is GestureKind.PANEL_OPEN:
                    feedback = "PANEL ARMED - separate the selected fingertips"
                    last_corner_signature = None
                    smoother.reset()
                elif event.kind is GestureKind.PANEL_CLOSE:
                    feedback = "PANEL CLOSED"
                    last_corner_signature = None
                    smoother.reset()
                elif event.kind is GestureKind.HAND_FILL_OPEN:
                    feedback = "HAND PANELS ON"
                else:
                    feedback = "HAND PANELS CLOSED"
                feedback_until = now + 2.5

            render_started = time.perf_counter()
            split_level = 2 if manual_split_enabled else 0
            raw_points = (
                tracker.panel_for_fingers(effect_state.panel_finger_ids)
                if effect_state.panel_active
                else None
            )
            if not effect_state.panel_active:
                tracker.selected_finger_ids = []
            current_points_are_valid = (
                raw_points is not None
                and is_valid_quadrilateral(raw_points, camera_frame.shape)
            )
            render_points = raw_points if current_points_are_valid else None

            # Tracking only reads the camera frame, so render in place instead
            # of copying another full-resolution image on every preview tick.
            frame = camera_frame
            points = None
            elapsed = now - started_at
            if render_points is not None:
                corner_signature = tuple(
                    tip_id
                    for pair in tracker.selected_finger_ids
                    for tip_id in pair
                )
                if corner_signature != last_corner_signature:
                    smoother.reset()
                    last_corner_signature = corner_signature
                points = smoother.update(
                    render_points,
                    enabled=smoothing_enabled,
                    timestamp=now,
                )
            else:
                smoother.reset()

            hand_render_data = []
            if effect_state.hand_fill_active:
                for index, hand in enumerate(tracker.tracked_hands[:2]):
                    landmarks = hand_smoothers[index].update(
                        hand.landmarks,
                        enabled=smoothing_enabled,
                        timestamp=now,
                    )
                    hand_render_data.append((landmarks, hand.palm_scale))
                for unused in hand_smoothers[len(tracker.tracked_hands[:2]) :]:
                    unused.reset()
            else:
                for hand_smoother in hand_smoothers:
                    hand_smoother.reset()

            has_effect_geometry = points is not None or bool(hand_render_data)
            shared_overlay = (
                overlay_renderer.overlay_for_frame(
                    elapsed, split_level=split_level, advance=True
                )
                if has_effect_geometry
                else None
            )
            if points is not None:
                overlay_renderer.render(
                    frame,
                    points,
                    elapsed,
                    split_level=split_level,
                    advance=False,
                    overlay=shared_overlay,
                )
                draw_neon_border(frame, points, glow_enabled, border_enabled)
            if hand_render_data:
                overlay_renderer.render_hands(
                    frame,
                    hand_render_data,
                    elapsed,
                    split_level=split_level,
                    advance=False,
                    overlay=shared_overlay,
                    glow_enabled=glow_enabled,
                    border_enabled=border_enabled,
                )

            panel_visible = points is not None
            hand_fill_visible = bool(hand_render_data)
            drawing_active = panel_visible or hand_fill_visible
            if tracker.snapshot_stale and tracker.tracked_hands:
                tracking_state = "RECOVERING"
            elif drawing_active:
                tracking_state = "LOCKED"
            elif effect_state.panel_active or effect_state.hand_fill_active:
                tracking_state = "ARMED"
            else:
                tracking_state = "READY" if hand_count == 2 else "SEARCHING"

            guidance, guidance_color = tracking_guidance(
                tracker,
                hand_count,
                frame.shape,
                brightness,
                effect_state,
                panel_visible,
                hand_fill_visible,
            )
            if hud_state["visible"] and (
                guidance != "Effects tracking" or calibration_enabled
            ):
                draw_message(frame, guidance, y=44, color=guidance_color, scale=0.65)

            performance.add("render", time.perf_counter() - render_started)

            # Export preparation used to copy every full-resolution frame even
            # while idle. Only build it while recording to keep the preview fast.
            export_frame = None
            if recording and record_camera_background:
                export_frame = frame.copy()
            elif recording:
                export_frame = np.zeros_like(frame)
                if points is not None:
                    overlay_renderer.render(
                        export_frame,
                        points,
                        elapsed,
                        split_level=split_level,
                        advance=False,
                        overlay=shared_overlay,
                    )
                    draw_neon_border(
                        export_frame, points, glow_enabled, border_enabled
                    )
                if hand_render_data:
                    overlay_renderer.render_hands(
                        export_frame,
                        hand_render_data,
                        elapsed,
                        split_level=split_level,
                        advance=False,
                        overlay=shared_overlay,
                        glow_enabled=glow_enabled,
                        border_enabled=border_enabled,
                    )

            hud_started = time.perf_counter()
            if hud_state["visible"] and debug_points_enabled:
                draw_finger_labels(
                    frame,
                    tracker.tracked_fingertips,
                    drawing_active=drawing_active,
                    selected_finger_ids=tracker.selected_finger_ids,
                )

            if hud_state["visible"] and calibration_enabled:
                draw_calibration(frame, tracker, guidance, guidance_color)

            if hud_state["visible"]:
                status_color = (
                    (80, 230, 100) if tracking_state == "LOCKED" else (60, 190, 255)
                )
                draw_message(
                    frame,
                    f"[{overlay_renderer.selected_key}] {overlay_renderer.label}  "
                    f"| {tracking_state} | {hand_count} hands | {fps:4.1f} FPS",
                    y=frame.shape[0] - 24,
                    color=status_color,
                    scale=0.46,
                )
                confidence_bar(frame, tracker.tracking_confidence)
                draw_message(
                    frame,
                    f"Confidence {tracker.tracking_confidence:.0%}",
                    y=82,
                    color=(235, 235, 235),
                    scale=0.44,
                )
                draw_message(
                    frame,
                    f"{tracker.backend_name} | result "
                    f"{tracker.snapshot_age_seconds * 1000.0:.0f} ms | "
                    f"dropped {tracker.dropped_submissions}",
                    y=106,
                    color=(235, 235, 235),
                    scale=0.40,
                )

            recording_elapsed = 0.0
            if recording and hud_state["visible"]:
                recording_elapsed = now - recording_started_at
                minutes, seconds = divmod(int(recording_elapsed), 60)
                draw_message(
                    frame,
                    f"REC {minutes:02d}:{seconds:02d}",
                    y=42,
                    x=max(22, frame.shape[1] - 170),
                    color=(40, 40, 255),
                    scale=0.65,
                )

            if hud_state["visible"] and help_enabled:
                recording_label = (
                    f"ON {recording_elapsed:0.1f}s -> {recording_path.name}"
                    if recording and recording_path is not None
                    else "OFF"
                )
                lines = [
                    ("H  hide controls", (110, 230, 255)),
                    ("1-4/custom overlay  [ ] opacity  - + sensitivity", (235, 235, 235)),
                    ("4-tip touch: open panel | two-finger taps: close", (235, 235, 235)),
                    ("Hold two full fists: toggle both hand panels", (235, 235, 235)),
                    ("C calibrate   S smoothing  R reset tracking", (235, 235, 235)),
                    ("D labels      G glow       B border   X split", (235, 235, 235)),
                    ("I camera inversion", (235, 235, 235)),
                    ("SPACE record  P screenshot V camera background", (235, 235, 235)),
                    ("U hide/show HUD  ESC/Q quit", (235, 235, 235)),
                    (f"Current overlay: [{overlay_renderer.selected_key}] {overlay_renderer.label}", (255, 220, 90)),
                    (f"Opacity: {overlay_renderer.opacity:.0%}", (235, 235, 235)),
                    (
                        f"Camera inversion: "
                        f"{'ON' if overlay_renderer.inversion_enabled else 'OFF'}",
                        (235, 235, 235),
                    ),
                    (f"Tracking: {tracking_state} | Hands: {hand_count}", (235, 235, 235)),
                    (
                        f"Panel: {'ON' if effect_state.panel_active else 'OFF'} | "
                        f"Hand panels: {'ON' if effect_state.hand_fill_active else 'OFF'}",
                        (80, 230, 100),
                    ),
                    (
                        "Panel fingers: "
                        + (
                            " / ".join(
                                "+".join(str((tip_id // 4) if tip_id != 4 else 1) for tip_id in pair)
                                for pair in effect_state.panel_finger_ids
                            )
                            if effect_state.panel_finger_ids
                            else "not selected"
                        ),
                        (235, 235, 235),
                    ),
                    (f"Confidence: {tracker.tracking_confidence:.0%} | FPS: {fps:.1f}", (235, 235, 235)),
                    (
                        f"Backend: {tracker.backend_name} | "
                        f"Inference: {tracker.last_inference_seconds * 1000.0:.1f} ms | "
                        f"Age: {tracker.snapshot_age_seconds * 1000.0:.1f} ms",
                        (235, 235, 235),
                    ),
                    (f"Sensitivity: {tracker.sensitivity:.0%}", (235, 235, 235)),
                    (f"Recording: {recording_label}", (80, 100, 255) if recording else (235, 235, 235)),
                    (f"Recording camera background: {'YES' if record_camera_background else 'NO'}", (235, 235, 235)),
                ]
                draw_panel(frame, lines)
            if hud_state["visible"] and feedback and now < feedback_until:
                draw_message(
                    frame,
                    feedback,
                    y=frame.shape[0] - 55,
                    color=(70, 210, 255),
                    scale=0.55,
                )
            performance.add("hud", time.perf_counter() - hud_started)

            if recording:
                recording_write_started = time.perf_counter()
                try:
                    if not writer.isOpened():
                        raise RuntimeError("video writer closed unexpectedly")
                    if not writer.write(export_frame):
                        raise RuntimeError("video writer worker stopped")
                    performance.add(
                        "recording_enqueue",
                        time.perf_counter() - recording_write_started,
                    )
                    performance.set_gauge(
                        "recording_queue_depth", writer.queue_depth
                    )
                    performance.set_gauge(
                        "recording_max_queue_depth", writer.max_queue_depth
                    )
                except (cv2.error, RuntimeError) as exc:
                    writer.release()
                    performance.increment(
                        "recording_dropped_frames", writer.dropped_frames
                    )
                    performance.increment(
                        "recording_written_frames", writer.written_frames
                    )
                    writer = None
                    recording = False
                    feedback = f"RECORDING FAILED: {exc}"
                    feedback_until = now + 5.0
                    print(feedback)

            display_started = time.perf_counter()
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            performance.add("display", time.perf_counter() - display_started)
            performance.add("frame_total", time.perf_counter() - frame_started)
            performance.increment("displayed_frames")
            if key == 255:
                continue
            if key == 27:
                break
            character = chr(key)
            normalized = character.lower() if character.isalpha() else character

            if normalized == "q":
                break
            if overlay_renderer.has_key(normalized):
                overlay_renderer.select(normalized)
                # Forced split intentionally uses bundled sources 1 and 2 and
                # therefore hides the selected asset. An explicit selection
                # should always show exactly what the user picked.
                manual_split_enabled = False
                if not overlay_renderer.selected_asset_available:
                    feedback = f"MEDIA MISSING for overlay {normalized}"
                    feedback_until = now + 4.0
                else:
                    feedback = (
                        f"Overlay [{overlay_renderer.selected_key}] "
                        f"{overlay_renderer.label}"
                    )
                    feedback_until = now + 2.0
            elif normalized == "h":
                help_enabled = not help_enabled
            elif normalized == "c":
                calibration_enabled = not calibration_enabled
            elif normalized == "s":
                smoothing_enabled = not smoothing_enabled
            elif normalized == "r":
                tracker.reset()
                smoother.reset()
                for hand_smoother in hand_smoothers:
                    hand_smoother.reset()
                gesture_controller.reset()
            elif normalized == "g":
                glow_enabled = not glow_enabled
            elif normalized == "b":
                border_enabled = not border_enabled
            elif normalized == "d":
                debug_points_enabled = not debug_points_enabled
            elif normalized == "u":
                hud_state["visible"] = not hud_state["visible"]
            elif normalized == "x":
                manual_split_enabled = not manual_split_enabled
            elif normalized == "i":
                inversion_enabled = overlay_renderer.toggle_inversion()
                feedback = (
                    "Camera inversion ON"
                    if inversion_enabled
                    else "Camera inversion OFF"
                )
                feedback_until = now + 2.5
            elif normalized == "v":
                record_camera_background = not record_camera_background
                feedback = (
                    "Recording camera background ON"
                    if record_camera_background
                    else "Recording camera background OFF"
                )
                feedback_until = now + 2.5
            elif normalized == "[":
                overlay_renderer.adjust_opacity(-0.05)
            elif normalized == "]":
                overlay_renderer.adjust_opacity(0.05)
            elif normalized in ("-", "_"):
                tracker.adjust_sensitivity(-0.05)
                feedback = f"Detection sensitivity {tracker.sensitivity:.0%}"
                feedback_until = now + 2.0
            elif normalized in ("=", "+"):
                tracker.adjust_sensitivity(0.05)
                feedback = f"Detection sensitivity {tracker.sensitivity:.0%}"
                feedback_until = now + 2.0
            elif normalized == "p":
                path = screenshot_path(config.output_filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(path), frame):
                    feedback = f"Screenshot saved: {path}"
                else:
                    feedback = f"SCREENSHOT FAILED: {path}"
                feedback_until = now + 4.0
                print(feedback)
            elif normalized == " ":
                if recording:
                    recording = False
                    writer.release()
                    performance.increment(
                        "recording_dropped_frames", writer.dropped_frames
                    )
                    performance.increment(
                        "recording_written_frames", writer.written_frames
                    )
                    writer = None
                    feedback = f"Saved recording: {recording_path}"
                    feedback_until = now + 4.0
                    print(feedback)
                else:
                    recording_path = resolve_output_path(config.output_filename)
                    raw_writer = create_video_writer(
                        frame, camera_fps, recording_path
                    )
                    if raw_writer is None:
                        feedback = f"RECORDING FAILED: {recording_path}"
                        feedback_until = now + 5.0
                        print(feedback)
                    else:
                        writer = AsyncVideoWriter(
                            raw_writer,
                            queue_size=config.recording_queue_size,
                        )
                        recording = True
                        recording_started_at = now
                        feedback = f"Recording: {recording_path}"
                        feedback_until = now + 3.0
                        print(feedback)
    finally:
        runtime_seconds = max(time.perf_counter() - started_at, 1e-6)
        process_cpu_seconds = max(
            time.process_time() - process_cpu_started_at, 0.0
        )
        performance.set_gauge("runtime_seconds", runtime_seconds)
        performance.set_gauge(
            "inference_frequency_hz", tracker.inference_count / runtime_seconds
        )
        if writer is not None:
            writer.release()
            performance.increment(
                "recording_dropped_frames", writer.dropped_frames
            )
            performance.increment(
                "recording_written_frames", writer.written_frames
            )
        performance.increment("camera_dropped_frames", camera.dropped_frames)
        performance.increment(
            "tracking_dropped_submissions", tracker.dropped_submissions
        )
        performance.increment("hand_swap_warnings", tracker.hand_swap_count)
        overlay_renderer.close()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        runtime_report = build_runtime_report(
            performance,
            config,
            tracker,
            runtime_seconds,
            process_cpu_seconds,
        )
        report_path = None
        report_error = None
        try:
            report_path = save_runtime_report(
                runtime_report,
                config.performance_output_filename,
            )
        except OSError as exc:
            report_error = exc
        print()
        print(format_runtime_summary(runtime_report))
        if report_path is not None:
            print(f"Full JSON report: {report_path}")
        else:
            print(f"Could not save performance report: {report_error}")
        wait_for_console_exit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
