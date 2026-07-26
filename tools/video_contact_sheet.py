"""Create timestamped contact sheets for visual regression videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def make_sheet(
    video_path: Path,
    output_path: Path,
    columns: int,
    rows: int,
    start: float,
    end: float | None,
    crop: tuple[int, int, int, int] | None,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration = max((frame_count - 1) / fps, 0.0)
    end = duration if end is None else min(end, duration)
    sample_count = columns * rows
    times = [start + (end - start) * index / max(sample_count - 1, 1) for index in range(sample_count)]

    cells: list = []
    for timestamp in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        if crop is not None:
            left, top, right, bottom = crop
            frame = frame[top:bottom, left:right]
        target_width = 320
        scale = target_width / frame.shape[1]
        frame = cv2.resize(frame, (target_width, int(frame.shape[0] * scale)))
        cv2.rectangle(frame, (0, 0), (112, 26), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{timestamp:05.1f}s",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(frame)

    capture.release()
    if not cells:
        raise RuntimeError(f"No frames decoded from {video_path}")

    cell_height = max(cell.shape[0] for cell in cells)
    while len(cells) < sample_count:
        cells.append(cells[-1].copy())
    normalized = [
        cv2.copyMakeBorder(cell, 0, cell_height - cell.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for cell in cells
    ]
    sheet_rows = [cv2.hconcat(normalized[start : start + columns]) for start in range(0, sample_count, columns)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.vconcat(sheet_rows)):
        raise RuntimeError(f"Could not write {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    make_sheet(
        args.video,
        args.output,
        args.columns,
        args.rows,
        args.start,
        args.end,
        tuple(args.crop) if args.crop else None,
    )


if __name__ == "__main__":
    main()
