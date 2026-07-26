"""Download and verify the pinned MediaPipe task models used at runtime."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_SHA256 = (
    "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(destination: Path, force: bool = False) -> Path:
    destination = destination.resolve()
    if destination.is_file() and not force:
        if sha256_file(destination) == HAND_MODEL_SHA256:
            print(f"Model already verified: {destination}")
            return destination
        raise RuntimeError(
            f"Existing model failed checksum verification: {destination}. "
            "Pass --force to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="hand_landmarker_", suffix=".task", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        print(f"Downloading {HAND_MODEL_URL}")
        urllib.request.urlretrieve(HAND_MODEL_URL, temporary_path)
        actual = sha256_file(temporary_path)
        if actual != HAND_MODEL_SHA256:
            raise RuntimeError(
                f"Downloaded model checksum mismatch: expected "
                f"{HAND_MODEL_SHA256}, got {actual}"
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Saved verified model: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "hand_landmarker.task",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_model(args.output, force=args.force)


if __name__ == "__main__":
    main()
