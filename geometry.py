"""Geometry helpers shared by tracking and rendering."""

from __future__ import annotations

import cv2
import numpy as np


def _as_quadrilateral(points):
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return None
    return polygon


def is_simple_convex_quadrilateral(points, minimum_area=1.0):
    """Return whether four cyclic points form a usable perspective target."""
    polygon = _as_quadrilateral(points)
    if polygon is None:
        return False

    area = abs(float(cv2.contourArea(polygon)))
    if area < minimum_area:
        return False

    # getPerspectiveTransform assumes a simple, consistently wound polygon.
    # Concave and self-crossing inputs are what create the triangular tears.
    rounded = np.rint(polygon).astype(np.int32)
    return bool(cv2.isContourConvex(rounded))


def build_fingertip_quadrilateral(first_pair, second_pair):
    """Connect two fingertip pairs without producing a crossed polygon.

    Each pair is ``(thumb, index)`` for one hand. There are two possible ways
    to connect the pairs; only a simple convex candidate can be perspective
    warped safely. The larger valid candidate is the stable outer surface.
    """
    first = np.asarray(first_pair, dtype=np.float32)
    second = np.asarray(second_pair, dtype=np.float32)
    if (
        first.shape != (2, 2)
        or second.shape != (2, 2)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        return None

    # Start at the first hand's upper/fingertip-side corner. The renderer maps
    # the source image's top edge to the first two points, so starting at the
    # thumb-side edge would vertically invert recognisable custom images.
    candidates = (
        np.array([first[1], second[1], second[0], first[0]], dtype=np.float32),
        np.array([first[1], second[0], second[1], first[0]], dtype=np.float32),
    )
    valid = [
        candidate
        for candidate in candidates
        if is_simple_convex_quadrilateral(candidate, minimum_area=4.0)
    ]
    if not valid:
        return None
    return max(valid, key=lambda polygon: abs(cv2.contourArea(polygon)))


def is_valid_quadrilateral(points, frame_shape):
    """Reject shapes that would make the perspective render fold or explode."""
    polygon = _as_quadrilateral(points)
    if polygon is None:
        return False

    frame_height, frame_width = frame_shape[:2]
    minimum_area = max(80.0, frame_width * frame_height * 0.00015)
    if not is_simple_convex_quadrilateral(polygon, minimum_area=minimum_area):
        return False

    edge_lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    if float(edge_lengths.min()) < 2.0:
        return False

    # At least one cross-hand edge must have a meaningful span. This preserves
    # intentionally pinched strips while rejecting two hands clasped together.
    cross_hand_span = max(float(edge_lengths[0]), float(edge_lengths[2]))
    return cross_hand_span >= max(24.0, frame_width * 0.035)
