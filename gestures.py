"""Instant, release-gated gestures and independently latched effect state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from itertools import combinations

import numpy as np

from hand_tracker import FINGERTIP_IDS, WRIST


FIST_CONFIRM_SECONDS = 0.15


class GestureKind(Enum):
    PANEL_OPEN = auto()
    PANEL_CLOSE = auto()
    HAND_FILL_OPEN = auto()
    HAND_FILL_CLOSE = auto()


@dataclass(frozen=True)
class GestureEvent:
    kind: GestureKind
    panel_finger_ids: tuple[tuple[int, int], tuple[int, int]] | None = None


@dataclass
class EffectState:
    panel_active: bool = False
    panel_finger_ids: tuple[tuple[int, int], tuple[int, int]] | None = None
    hand_fill_active: bool = False

    def apply(self, event: GestureEvent) -> None:
        if event.kind is GestureKind.PANEL_OPEN:
            self.panel_active = True
            self.panel_finger_ids = event.panel_finger_ids
        elif event.kind is GestureKind.PANEL_CLOSE:
            self.panel_active = False
            self.panel_finger_ids = None
        elif event.kind is GestureKind.HAND_FILL_OPEN:
            self.hand_fill_active = True
        elif event.kind is GestureKind.HAND_FILL_CLOSE:
            self.hand_fill_active = False


class _HoldGate:
    """Fire on the first candidate frame and re-arm only after release."""

    def __init__(self, confirmation_seconds=0.0):
        self.confirmation_seconds = max(0.0, float(confirmation_seconds))
        self.signature = None
        self.candidate_since = None
        self.armed = True

    def update(self, candidate, now: float):
        if candidate is None or candidate is False:
            self.signature = None
            self.candidate_since = None
            self.armed = True
            return None
        signature = candidate if candidate is not True else True
        if not self.armed:
            return None
        if signature != self.signature:
            self.signature = signature
            self.candidate_since = now
        if now - self.candidate_since < self.confirmation_seconds:
            return None
        self.armed = False
        return candidate

    def reset(self) -> None:
        self.signature = None
        self.candidate_since = None
        self.armed = True


class GestureController:
    """Recognize the four reference gestures using palm-normalized geometry."""

    def __init__(self):
        self._gates = {kind: _HoldGate() for kind in GestureKind}
        self._gates[GestureKind.HAND_FILL_OPEN] = _HoldGate(FIST_CONFIRM_SECONDS)
        self._panel_open_blocked_until_close_release = False
        self._panel_close_blocked_until_contact_release = False

    def reset(self) -> None:
        for gate in self._gates.values():
            gate.reset()
        self._panel_open_blocked_until_close_release = False
        self._panel_close_blocked_until_contact_release = False

    def update(self, hands, now: float, state: EffectState) -> list[GestureEvent]:
        hands = tuple(hands)
        panel_selection = self._four_tip_contact(hands)
        paired_finger_taps = self._both_thumb_index_pinches(hands)
        simple_cross_touch = self._cross_hand_fingertips_touch(hands)
        # Support the demonstrated thumb/index taps on both hands as well as a
        # simple opposing-tip touch. A four-tip cluster remains exclusively the
        # open/select gesture, so the broad cross-touch detector cannot cancel it.
        panel_close = paired_finger_taps or (
            simple_cross_touch if panel_selection is None else None
        )
        hand_panel_toggle = self._both_fists(hands)

        if not panel_close:
            self._panel_open_blocked_until_close_release = False
        if not (panel_selection or paired_finger_taps or simple_cross_touch):
            self._panel_close_blocked_until_contact_release = False

        events: list[GestureEvent] = []

        # One shared gate makes the fist pose a true toggle: holding the pose
        # cannot open and immediately close the hand panels on the next frame.
        fill_toggled = self._gates[GestureKind.HAND_FILL_OPEN].update(
            hand_panel_toggle, now
        )
        if fill_toggled:
            kind = (
                GestureKind.HAND_FILL_CLOSE
                if state.hand_fill_active
                else GestureKind.HAND_FILL_OPEN
            )
            events.append(GestureEvent(kind))

        close_panel_candidate = (
            panel_close
            if state.panel_active
            and not self._panel_close_blocked_until_contact_release
            else None
        )
        close_panel_fired = self._gates[GestureKind.PANEL_CLOSE].update(
            close_panel_candidate, now
        )
        if close_panel_fired:
            events.append(GestureEvent(GestureKind.PANEL_CLOSE))
            self._panel_open_blocked_until_close_release = True

        open_panel_candidate = (
            panel_selection
            if not state.panel_active
            and not panel_close
            and not close_panel_fired
            and not self._panel_open_blocked_until_close_release
            else None
        )
        selected = self._gates[GestureKind.PANEL_OPEN].update(
            open_panel_candidate, now
        )
        if selected:
            events.append(GestureEvent(GestureKind.PANEL_OPEN, selected))
            self._panel_close_blocked_until_contact_release = True

        return events

    @staticmethod
    def _mean_palm_scale(hands) -> float:
        return float(np.mean([max(hand.palm_scale, 1.0) for hand in hands]))

    @classmethod
    def _four_tip_contact(cls, hands):
        if len(hands) != 2:
            return None
        scale = cls._mean_palm_scale(hands)
        midpoint = (hands[0].palm_center + hands[1].palm_center) * 0.5
        best = None

        for first_ids in combinations(FINGERTIP_IDS, 2):
            if any(
                hands[0].scores.get(tip_id, 0.0) < 55.0
                for tip_id in first_ids
            ):
                continue
            for second_ids in combinations(FINGERTIP_IDS, 2):
                if any(
                    hands[1].scores.get(tip_id, 0.0) < 55.0
                    for tip_id in second_ids
                ):
                    continue
                points = np.array(
                    [
                        hands[0].tip_points[first_ids[0]],
                        hands[0].tip_points[first_ids[1]],
                        hands[1].tip_points[second_ids[0]],
                        hands[1].tip_points[second_ids[1]],
                    ],
                    dtype=np.float32,
                )
                distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
                diameter = float(distances.max())
                center_offset = float(np.linalg.norm(points.mean(axis=0) - midpoint))
                outward = all(
                    np.linalg.norm(point - hand.landmarks[WRIST]) >= 0.68 * hand.palm_scale
                    for hand, ids in zip(hands, (first_ids, second_ids))
                    for point in (hand.tip_points[ids[0]], hand.tip_points[ids[1]])
                )
                if (
                    not outward
                    or diameter > 0.95 * scale
                    or center_offset > 1.25 * scale
                ):
                    continue
                score = diameter + center_offset * 0.15
                candidate = (score, (tuple(first_ids), tuple(second_ids)))
                if best is None or candidate[0] < best[0]:
                    best = candidate
        return None if best is None else best[1]

    @staticmethod
    def _hand_is_fist(hand) -> bool:
        """Recognize a fist while tolerating MediaPipe's noisy thumb angle."""
        scale = max(float(hand.palm_scale), 1.0)

        def closed(tip_id, score_limit, distance_limit):
            return (
                hand.scores.get(tip_id, 999.0) < score_limit
                and np.linalg.norm(hand.tip_points[tip_id] - hand.palm_center)
                <= distance_limit * scale
            )

        # A thumb wrapped across the fingers can look almost perfectly straight
        # to MediaPipe, so its bend score is not a reliable fist signal. Check
        # that the thumb tip is tucked beside the index PIP joint instead.
        thumb_tucked = (
            np.linalg.norm(hand.landmarks[4] - hand.landmarks[6])
            <= 0.45 * scale
        )

        # Index is mandatory so a thumb/index pointing pose cannot be mistaken
        # for a fist. Accept two of middle/ring/pinky because MediaPipe can mark
        # one occluded curled finger as extended.
        index_closed = closed(8, 122.0, 1.40)
        outer_closed = sum(closed(tip_id, 122.0, 1.40) for tip_id in (12, 16, 20))
        return thumb_tucked and index_closed and outer_closed >= 2

    @classmethod
    def _both_fists(cls, hands):
        return True if len(hands) == 2 and all(cls._hand_is_fist(hand) for hand in hands) else None

    @staticmethod
    def _hand_is_thumb_index_pinch(hand) -> bool:
        scale = max(float(hand.palm_scale), 1.0)
        thumb = hand.tip_points[4]
        index = hand.tip_points[8]
        contact = (thumb + index) * 0.5
        return (
            np.linalg.norm(thumb - index) <= 0.42 * scale
            and np.linalg.norm(contact - hand.palm_center) >= 0.45 * scale
        )

    @classmethod
    def _both_thumb_index_pinches(cls, hands):
        return (
            True
            if len(hands) == 2
            and all(cls._hand_is_thumb_index_pinch(hand) for hand in hands)
            else None
        )

    @staticmethod
    def _hand_is_bunched(hand) -> bool:
        tips = np.array([hand.tip_points[tip_id] for tip_id in FINGERTIP_IDS])
        scale = max(hand.palm_scale, 1.0)
        centroid = tips.mean(axis=0)
        spread = float(np.linalg.norm(tips - centroid, axis=1).max())
        outside_palm = float(np.linalg.norm(centroid - hand.palm_center)) >= 0.62 * scale
        fingers_still_outward = len(hand.extended_ids) >= 4
        return (
            spread <= 0.72 * scale
            and outside_palm
            and fingers_still_outward
            and not GestureController._hand_is_fist(hand)
        )

    @classmethod
    def _both_fingertips_bunched(cls, hands):
        return True if len(hands) == 2 and all(cls._hand_is_bunched(hand) for hand in hands) else None

    @staticmethod
    def _both_hands_open(hands) -> bool:
        if len(hands) != 2:
            return False
        for hand in hands:
            tips = np.array([hand.tip_points[tip_id] for tip_id in FINGERTIP_IDS])
            distances = np.linalg.norm(tips[:, None] - tips[None, :], axis=2)
            if len(hand.extended_ids) < 4 or distances.max() < 1.45 * hand.palm_scale:
                return False
        return True

    @classmethod
    def _cross_hand_fingertips_touch(cls, hands):
        if len(hands) != 2:
            return None
        scale = cls._mean_palm_scale(hands)
        midpoint = (hands[0].palm_center + hands[1].palm_center) * 0.5
        for first_id in FINGERTIP_IDS:
            first = hands[0].tip_points[first_id]
            for second_id in FINGERTIP_IDS:
                second = hands[1].tip_points[second_id]
                contact_center = (first + second) * 0.5
                if (
                    np.linalg.norm(first - second) <= 0.58 * scale
                    and np.linalg.norm(contact_center - midpoint) <= 1.0 * scale
                ):
                    return True
        return None
