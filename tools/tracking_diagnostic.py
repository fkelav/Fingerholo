"""Summarize tracking and gesture-state transitions over a recorded video."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import is_valid_quadrilateral
from gestures import EffectState, GestureController
from hand_tracker import HandTracker
from runtime_pipeline import PerformanceMonitor
from tracking import AsyncTasksHandsProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--initial-panel", action="store_true")
    parser.add_argument("--initial-hand-fill", action="store_true")
    parser.add_argument("--gesture-metrics", action="store_true")
    parser.add_argument("--processing-width", type=int, default=480)
    parser.add_argument(
        "--backend", choices=("legacy", "tasks"), default="legacy"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hand_landmarker.task"),
    )
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(0, int(round(args.start * fps)))
    end_frame = None if args.end is None else max(start_frame, int(round(args.end * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processor = (
        AsyncTasksHandsProcessor(args.model)
        if args.backend == "tasks"
        else None
    )
    tracker = HandTracker(
        processing_width=args.processing_width,
        hands_processor=processor,
        inference_interval=1 if processor is not None else 2,
        tracking_grace_seconds=0.35,
        max_result_age_seconds=0.20,
    )
    gesture_controller = GestureController()
    effect_state = EffectState(
        panel_active=args.initial_panel,
        panel_finger_ids=((4, 8), (4, 8)) if args.initial_panel else None,
        hand_fill_active=args.initial_hand_fill,
    )
    counts = Counter()
    gesture_events = []
    transitions = []
    previous_state = None
    frame_index = start_frame
    processing_started = time.perf_counter()
    best_pinch = None
    first_fists = None
    performance = PerformanceMonitor()
    frame_rows = []

    try:
        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.stride:
                frame_index += 1
                continue
            if args.crop:
                left, top, right, bottom = args.crop
                frame = frame[top:bottom, left:right]

            if args.realtime:
                target_time = (
                    processing_started + (frame_index - start_frame) / fps
                )
                remaining = target_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
            tracking_started = time.perf_counter()
            hand_count, points, _extra_fingers = tracker.find_fingertips(
                frame,
                frame_id=frame_index,
                captured_at=tracking_started,
            )
            tracking_seconds = time.perf_counter() - tracking_started
            performance.add("tracking", tracking_seconds)
            if tracker.snapshot_is_new:
                performance.add("preprocess", tracker.preprocess_seconds)
                performance.add("inference", tracker.last_inference_seconds)
                performance.add(
                    "tracking_result_age", tracker.snapshot_age_seconds
                )
            timestamp = frame_index / fps
            if args.gesture_metrics and len(tracker.tracked_hands) == 2:
                pinch_ratios = []
                pinch_outside = []
                for hand in tracker.tracked_hands:
                    scale = max(hand.palm_scale, 1.0)
                    thumb = hand.tip_points[4]
                    index = hand.tip_points[8]
                    pinch_ratios.append(float(np.linalg.norm(thumb - index) / scale))
                    pinch_outside.append(
                        float(np.linalg.norm((thumb + index) * 0.5 - hand.palm_center) / scale)
                    )
                bunch = gesture_controller._both_fingertips_bunched(
                    tracker.tracked_hands
                )
                pinch = gesture_controller._both_thumb_index_pinches(
                    tracker.tracked_hands
                )
                fists = gesture_controller._both_fists(tracker.tracked_hands)
                bilateral_ratio = max(pinch_ratios)
                if best_pinch is None or bilateral_ratio < best_pinch[0]:
                    best_pinch = (
                        bilateral_ratio,
                        timestamp,
                        tuple(pinch_ratios),
                        tuple(pinch_outside),
                        tuple(
                            gesture_controller._hand_is_fist(hand)
                            for hand in tracker.tracked_hands
                        ),
                    )
                if fists and first_fists is None:
                    first_fists = timestamp
                selection = gesture_controller._four_tip_contact(
                    tracker.tracked_hands
                )
                if bunch or pinch or fists or selection:
                    metrics = []
                    for hand in tracker.tracked_hands:
                        tips = np.array(list(hand.tip_points.values()))
                        centroid = tips.mean(axis=0)
                        spread = np.linalg.norm(tips - centroid, axis=1).max()
                        outside = np.linalg.norm(centroid - hand.palm_center)
                        metrics.append(
                            f"spread={spread / max(hand.palm_scale, 1):.2f} "
                            f"outside={outside / max(hand.palm_scale, 1):.2f} "
                            f"extended={len(hand.extended_ids)}"
                        )
                    print(
                        f"metric {timestamp:5.2f}s bunch={bool(bunch)} "
                        f"pinch={bool(pinch)} fists={bool(fists)} "
                        f"selection={selection} | "
                        + " | ".join(metrics)
                    )
            gesture_started = time.perf_counter()
            events = (
                gesture_controller.update(
                    tracker.tracked_hands, timestamp, effect_state
                )
                if tracker.snapshot_is_new and not tracker.snapshot_stale
                else []
            )
            performance.add("gestures", time.perf_counter() - gesture_started)
            for event in events:
                effect_state.apply(event)
                gesture_events.append((timestamp, event.kind.name))
            if hand_count != 2:
                state = f"hands={hand_count}"
            elif points is None:
                state = "pose-rejected"
            elif not is_valid_quadrilateral(points, frame.shape):
                state = "shape-rejected"
            else:
                state = "panel-valid"
            counts[state] += 1
            frame_rows.append(
                {
                    "frame": frame_index,
                    "timestamp_seconds": timestamp,
                    "state": state,
                    "hand_count": hand_count,
                    "tracking_ms": tracking_seconds * 1000.0,
                    "preprocess_ms": tracker.preprocess_seconds * 1000.0,
                    "inference_ms": tracker.last_inference_seconds * 1000.0,
                    "confidence": tracker.tracking_confidence,
                    "snapshot_new": tracker.snapshot_is_new,
                    "snapshot_stale": tracker.snapshot_stale,
                }
            )
            if state != previous_state:
                transitions.append((frame_index / fps, state))
                previous_state = state
            frame_index += 1
    finally:
        tracker.close()
        capture.release()

    processing_seconds = max(time.perf_counter() - processing_started, 1e-6)
    total = sum(counts.values())
    print(f"video={args.video.name} sampled_frames={total} stride={args.stride}")
    print(f"processing_fps={total / processing_seconds:.1f}")
    print(f"tracker_inferences={tracker.inference_count}")
    if args.gesture_metrics and best_pinch is not None:
        ratio, timestamp, per_hand, outside, fist_flags = best_pinch
        print(
            f"closest_bilateral_pinch={ratio:.3f} at {timestamp:.2f}s "
            f"per_hand={per_hand} outside={outside} hand_fists={fist_flags}"
        )
        print(f"first_both_fists={first_fists}")
    for state, count in sorted(counts.items()):
        print(f"{state}: {count} ({count / max(total, 1):.1%})")
    print("transitions:")
    for timestamp, state in transitions:
        print(f"{timestamp:6.2f}s {state}")
    print("gesture events:")
    if not gesture_events:
        print("  none")
    for timestamp, event_name in gesture_events:
        print(f"{timestamp:6.2f}s {event_name}")

    report = {
        "video": str(args.video),
        "backend": tracker.backend_name,
        "realtime": args.realtime,
        "processing_width": args.processing_width,
        "stride": args.stride,
        "sampled_frames": total,
        "processing_seconds": processing_seconds,
        "processing_fps": total / processing_seconds,
        "tracker_inferences": tracker.inference_count,
        "dropped_submissions": tracker.dropped_submissions,
        "states": dict(counts),
        "transitions": [
            {"timestamp_seconds": timestamp, "state": state}
            for timestamp, state in transitions
        ],
        "gesture_events": [
            {"timestamp_seconds": timestamp, "event": event_name}
            for timestamp, event_name in gesture_events
        ],
        "performance": performance.to_dict(),
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"json_report={args.json_output}")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
            fieldnames = (
                list(frame_rows[0].keys())
                if frame_rows
                else [
                    "frame",
                    "timestamp_seconds",
                    "state",
                    "hand_count",
                    "tracking_ms",
                    "preprocess_ms",
                    "inference_ms",
                    "confidence",
                    "snapshot_new",
                    "snapshot_stale",
                ]
            )
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frame_rows)
        print(f"csv_report={args.csv_output}")


if __name__ == "__main__":
    main()
