"""Transcode an OpenCV-readable clip to browser-compatible VP8 WebM."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def transcode(source: Path, output: Path) -> tuple[int, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open input video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not 1.0 <= fps <= 240.0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"invalid input dimensions: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(
            "OpenCV could not create VP8 WebM; use an FFmpeg build with VP8 "
            "support or provide an H.264/WebM fixture directly."
        )

    frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        raise RuntimeError(f"input contained no decodable frames: {source}")
    return frames, fps, (width, height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("gpu_benchmark/public/fixture.webm"),
    )
    args = parser.parse_args()
    frames, fps, size = transcode(args.source, args.output)
    print(
        f"fixture={args.output} frames={frames} fps={fps:.3f} "
        f"size={size[0]}x{size[1]}"
    )


if __name__ == "__main__":
    main()
