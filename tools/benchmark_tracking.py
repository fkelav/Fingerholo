"""Run the offline tracking diagnostic at several inference widths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run_width(args, width: int) -> dict:
    diagnostic = Path(__file__).with_name("tracking_diagnostic.py")
    with tempfile.TemporaryDirectory(prefix="fingerholo_benchmark_") as folder:
        report_path = Path(folder) / f"{width}.json"
        command = [
            sys.executable,
            str(diagnostic),
            str(args.video),
            "--stride",
            str(args.stride),
            "--processing-width",
            str(width),
            "--backend",
            args.backend,
            "--model",
            str(args.model),
            "--json-output",
            str(report_path),
        ]
        if args.realtime:
            command.append("--realtime")
        if args.start:
            command.extend(("--start", str(args.start)))
        if args.end is not None:
            command.extend(("--end", str(args.end)))
        if args.crop:
            command.append("--crop")
            command.extend(str(value) for value in args.crop)
        subprocess.run(command, check=True)
        return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=(320, 480, 640))
    parser.add_argument("--stride", type=int, default=1)
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
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM")
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/tracking_width_benchmark.json"),
    )
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    reports = [run_width(args, width) for width in args.widths]
    output = {
        "video": str(args.video),
        "backend": args.backend,
        "realtime": args.realtime,
        "widths": list(args.widths),
        "reports": reports,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"json_report={args.json_output}")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for report in reports:
            tracking = report["performance"]["stages"].get("tracking", {})
            inference = report["performance"]["stages"].get("inference", {})
            result_age = report["performance"]["stages"].get(
                "tracking_result_age", {}
            )
            rows.append(
                {
                    "processing_width": report["processing_width"],
                    "processing_fps": report["processing_fps"],
                    "tracking_p50_ms": tracking.get("p50_ms", 0.0),
                    "tracking_p95_ms": tracking.get("p95_ms", 0.0),
                    "inference_p50_ms": inference.get("p50_ms", 0.0),
                    "inference_p95_ms": inference.get("p95_ms", 0.0),
                    "result_age_p50_ms": result_age.get("p50_ms", 0.0),
                    "result_age_p95_ms": result_age.get("p95_ms", 0.0),
                    "panel_valid": report["states"].get("panel-valid", 0),
                    "sampled_frames": report["sampled_frames"],
                }
            )
        with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv_report={args.csv_output}")


if __name__ == "__main__":
    main()
