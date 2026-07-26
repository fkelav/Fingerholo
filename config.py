"""Configuration loading and validation for the webcam application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass
class AppConfig:
    camera_index: int = 0
    resolution: tuple[int, int] = (1280, 720)
    processing_width: int = 480
    opacity: float = 0.68
    smoothing_amount: float = 0.30
    output_filename: str = "output/finger_hologram_{timestamp}.mp4"
    default_overlay: str = "1"
    asset_directory: str = "assets"
    detection_sensitivity: float = 0.55
    tracking_grace_seconds: float = 0.35
    max_tracking_result_age_seconds: float = 0.20
    tracking_backend: str = "auto"
    hand_model_path: str = "models/hand_landmarker.task"
    recording_queue_size: int = 2
    performance_output_filename: str = (
        "artifacts/performance_{timestamp}.json"
    )
    record_camera_background: bool = True

    def validate(self) -> "AppConfig":
        self.camera_index = int(self.camera_index)
        self.resolution = parse_resolution(self.resolution)
        self.processing_width = max(160, int(self.processing_width))
        self.opacity = float(min(1.0, max(0.10, self.opacity)))
        self.smoothing_amount = float(min(1.0, max(0.0, self.smoothing_amount)))
        self.default_overlay = str(self.default_overlay)
        if self.default_overlay not in {"1", "2", "3", "4"}:
            raise ValueError("default_overlay must be one of 1, 2, 3, or 4")
        self.detection_sensitivity = float(
            min(0.95, max(0.10, self.detection_sensitivity))
        )
        self.tracking_grace_seconds = float(
            min(2.0, max(0.0, self.tracking_grace_seconds))
        )
        self.max_tracking_result_age_seconds = float(
            min(2.0, max(0.01, self.max_tracking_result_age_seconds))
        )
        self.tracking_backend = str(self.tracking_backend).lower()
        if self.tracking_backend not in {"auto", "tasks", "legacy"}:
            raise ValueError("tracking_backend must be auto, tasks, or legacy")
        self.hand_model_path = str(self.hand_model_path)
        if not self.hand_model_path:
            raise ValueError("hand_model_path cannot be empty")
        self.recording_queue_size = min(
            16, max(1, int(self.recording_queue_size))
        )
        if not self.output_filename:
            raise ValueError("output_filename cannot be empty")
        self.performance_output_filename = str(
            self.performance_output_filename
        )
        if not self.performance_output_filename:
            raise ValueError("performance_output_filename cannot be empty")
        return self

    def to_dict(self) -> dict:
        result = asdict(self)
        result["resolution"] = list(self.resolution)
        return result


def parse_resolution(value) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().split("x", 1)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        parts = value
    else:
        raise ValueError("resolution must look like 1280x720 or [1280, 720]")
    try:
        width, height = (int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolution values must be integers") from exc
    if width < 160 or height < 120:
        raise ValueError("resolution must be at least 160x120")
    return width, height


def load_config(path: str | Path | None) -> AppConfig:
    """Load JSON settings; unknown keys are rejected to catch typos."""
    if path is None:
        return AppConfig().validate()
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a JSON object")
    allowed = {field.name for field in fields(AppConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown configuration option(s): {', '.join(unknown)}")
    return AppConfig(**data).validate()
