"""Evaluate the objective TypeScript rewrite gate from two metric reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = (
    "p95_result_age_ms",
    "display_fps",
    "average_cpu_percent",
    "tracking_accuracy",
    "gesture_accuracy",
)


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED if data.get(name) is None]
    if missing:
        raise ValueError(
            f"{path} needs measured values for: {', '.join(missing)}"
        )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_metrics", type=Path)
    parser.add_argument("web_metrics", type=Path)
    args = parser.parse_args()
    python = load_metrics(args.python_metrics)
    web = load_metrics(args.web_metrics)

    latency_gain = 1.0 - web["p95_result_age_ms"] / python["p95_result_age_ms"]
    cpu_gain = 1.0 - web["average_cpu_percent"] / python["average_cpu_percent"]
    tracking_delta = web["tracking_accuracy"] - python["tracking_accuracy"]
    gesture_delta = web["gesture_accuracy"] - python["gesture_accuracy"]
    checks = {
        "latency_improvement_at_least_30_percent": latency_gain >= 0.30,
        "display_sustains_30_fps": web["display_fps"] >= 30.0,
        "cpu_reduction_at_least_25_percent": cpu_gain >= 0.25,
        "tracking_within_two_points": tracking_delta >= -0.02,
        "gestures_within_two_points": gesture_delta >= -0.02,
        "offline_assets": bool(web.get("offline_assets")),
        "gpu_active": bool(web.get("gpu_active")),
    }
    report = {
        "approved": all(checks.values()),
        "checks": checks,
        "comparisons": {
            "latency_improvement": latency_gain,
            "cpu_reduction": cpu_gain,
            "tracking_accuracy_delta": tracking_delta,
            "gesture_accuracy_delta": gesture_delta,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["approved"] else 1)


if __name__ == "__main__":
    main()
