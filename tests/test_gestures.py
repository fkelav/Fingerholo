from dataclasses import replace
import unittest

import numpy as np

from gestures import EffectState, GestureController, GestureEvent, GestureKind
from hand_tracker import FINGERTIP_IDS, TrackedHand


def make_hand(palm_x, tip_points, scores=None, palm_y=120.0, scale=40.0):
    palm_center = np.array([palm_x, palm_y], dtype=np.float32)
    landmarks = np.tile(palm_center, (21, 1)).astype(np.float32)
    landmarks[0] = palm_center + np.array([0.0, scale], dtype=np.float32)
    landmarks[5] = palm_center + np.array([-12.0, -4.0], dtype=np.float32)
    landmarks[9] = palm_center + np.array([0.0, -6.0], dtype=np.float32)
    landmarks[13] = palm_center + np.array([10.0, -4.0], dtype=np.float32)
    landmarks[17] = palm_center + np.array([18.0, 2.0], dtype=np.float32)
    tips = {}
    for index, tip_id in enumerate(FINGERTIP_IDS):
        point = np.array(
            tip_points.get(tip_id, (palm_x + (index - 2) * 18.0, 55.0)),
            dtype=np.float32,
        )
        tips[tip_id] = point
        landmarks[tip_id] = point
    score_map = {tip_id: 140.0 for tip_id in FINGERTIP_IDS}
    if scores:
        score_map.update(scores)
    return TrackedHand(
        slot=0,
        landmarks=landmarks,
        center=palm_center.copy(),
        palm_center=palm_center,
        palm_scale=scale,
        tip_points=tips,
        scores=score_map,
        extended_ids=tuple(
            tip_id for tip_id, score in score_map.items() if score >= 120.0
        ),
    )


def four_tip_contact():
    inactive = {4: 20.0, 16: 20.0, 20: 20.0}
    left = make_hand(
        100,
        {8: (147, 96), 12: (149, 103)},
        scores=inactive,
    )
    right = make_hand(
        200,
        {8: (152, 97), 12: (151, 104)},
        scores=inactive,
    )
    return left, right


def fists():
    hands = []
    for x in (100, 200):
        points = {
            tip_id: (x + offset, 120 + abs(offset) * 0.2)
            for tip_id, offset in zip(FINGERTIP_IDS, (-16, -8, 0, 8, 16))
        }
        hands.append(
            make_hand(x, points, scores={tip_id: 35.0 for tip_id in FINGERTIP_IDS})
        )
    return tuple(hands)


def thumb_index_out_with_other_fingers_curled():
    hands = []
    for x in (100, 200):
        points = {
            4: (x - 30, 58),
            8: (x - 10, 48),
            12: (x, 120),
            16: (x + 8, 122),
            20: (x + 16, 124),
        }
        scores = {tip_id: 35.0 for tip_id in FINGERTIP_IDS}
        scores.update({4: 140.0, 8: 140.0})
        hands.append(make_hand(x, points, scores=scores))
    return tuple(hands)


def fists_with_one_noisy_outer_finger():
    return tuple(
        replace(hand, scores={**hand.scores, 20: 135.0}, extended_ids=(20,))
        for hand in fists()
    )


def fists_with_straight_scoring_tucked_thumbs():
    return tuple(
        replace(hand, scores={**hand.scores, 4: 195.0}, extended_ids=(4,))
        for hand in fists()
    )


def thumbs_up_with_other_fingers_curled():
    hands = []
    for x in (100, 200):
        points = {
            4: (x - 38, 54),
            8: (x - 8, 120),
            12: (x, 122),
            16: (x + 8, 122),
            20: (x + 16, 124),
        }
        scores = {tip_id: 35.0 for tip_id in FINGERTIP_IDS}
        scores[4] = 195.0
        hands.append(make_hand(x, points, scores=scores))
    return tuple(hands)


def cross_hand_touch():
    inactive = {4: 20.0, 12: 20.0, 16: 20.0, 20: 20.0}
    left = make_hand(100, {8: (149, 90)}, scores=inactive)
    right = make_hand(200, {8: (151, 90)}, scores=inactive)
    return left, right


def thumb_index_pinches():
    hands = []
    for x in (100, 200):
        hands.append(
            make_hand(
                x,
                {
                    4: (x - 5, 82),
                    8: (x + 5, 82),
                    12: (x, 48),
                    16: (x + 17, 55),
                    20: (x + 30, 68),
                },
            )
        )
    return tuple(hands)


def relaxed_four_tip_contact():
    inactive = {4: 20.0, 16: 20.0, 20: 20.0}
    selected = {8: 60.0, 12: 60.0}
    left = make_hand(
        100,
        {8: (135, 94), 12: (145, 101)},
        scores={**inactive, **selected},
    )
    right = make_hand(
        200,
        {8: (155, 96), 12: (165, 103)},
        scores={**inactive, **selected},
    )
    return left, right


def bunched_hands():
    hands = []
    for x in (100, 200):
        points = {
            tip_id: (x + offset, 84 + abs(offset) * 0.15)
            for tip_id, offset in zip(FINGERTIP_IDS, (-6, -3, 0, 3, 6))
        }
        hands.append(make_hand(x, points))
    return tuple(hands)


class GestureControllerTests(unittest.TestCase):
    def test_effect_state_keeps_panel_and_hand_fill_independent(self):
        state = EffectState()
        state.apply(GestureEvent(GestureKind.PANEL_OPEN, ((4, 8), (12, 16))))
        state.apply(GestureEvent(GestureKind.HAND_FILL_OPEN))
        state.apply(GestureEvent(GestureKind.PANEL_CLOSE))

        self.assertFalse(state.panel_active)
        self.assertIsNone(state.panel_finger_ids)
        self.assertTrue(state.hand_fill_active)

    def test_four_tip_contact_instantly_selects_exact_fingers(self):
        controller = GestureController()
        state = EffectState()
        hands = four_tip_contact()

        events = controller.update(hands, 1.0, state)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, GestureKind.PANEL_OPEN)
        self.assertEqual(events[0].panel_finger_ids, ((8, 12), (8, 12)))

    def test_four_tip_contact_accepts_a_wider_lower_confidence_contact(self):
        events = GestureController().update(
            relaxed_four_tip_contact(), 0.0, EffectState()
        )

        self.assertEqual([event.kind for event in events], [GestureKind.PANEL_OPEN])

    def test_gesture_requires_release_before_it_can_fire_again(self):
        controller = GestureController()
        state = EffectState()
        hands = four_tip_contact()

        self.assertTrue(controller.update(hands, 0.0, state))
        self.assertEqual(controller.update(hands, 0.60, state), [])
        controller.update([], 0.70, state)
        self.assertTrue(controller.update(hands, 1.0, state))

    def test_open_contact_must_release_before_it_can_close_panel(self):
        controller = GestureController()
        state = EffectState()

        open_events = controller.update(four_tip_contact(), 0.0, state)
        state.apply(open_events[0])
        self.assertEqual(controller.update(cross_hand_touch(), 0.1, state), [])

        controller.update([], 0.2, state)
        close_events = controller.update(cross_hand_touch(), 0.3, state)
        self.assertEqual(
            [event.kind for event in close_events], [GestureKind.PANEL_CLOSE]
        )

    def test_cross_hand_fingertip_touch_closes_panel(self):
        controller = GestureController()
        state = EffectState(panel_active=True, panel_finger_ids=((8, 12), (8, 12)))
        hands = cross_hand_touch()

        events = controller.update(hands, 0.0, state)

        self.assertEqual([event.kind for event in events], [GestureKind.PANEL_CLOSE])

    def test_thumb_index_taps_on_both_hands_close_panel(self):
        state = EffectState(panel_active=True, panel_finger_ids=((8, 12), (8, 12)))

        events = GestureController().update(thumb_index_pinches(), 0.0, state)

        self.assertEqual([event.kind for event in events], [GestureKind.PANEL_CLOSE])

    def test_cross_hand_touch_cannot_immediately_reopen_panel(self):
        controller = GestureController()
        state = EffectState(panel_active=True, panel_finger_ids=((8, 12), (8, 12)))
        hands = cross_hand_touch()

        events = controller.update(hands, 0.0, state)
        state.apply(events[0])
        self.assertEqual(controller.update(hands, 0.1, state), [])

    def test_bunched_fingertips_do_not_open_hand_panels(self):
        controller = GestureController()
        state = EffectState()
        hands = bunched_hands()

        events = controller.update(hands, 0.0, state)

        self.assertNotIn(GestureKind.HAND_FILL_OPEN, [event.kind for event in events])
        self.assertFalse(GestureController._both_fists(hands))

    def test_two_fists_open_hand_panels(self):
        controller = GestureController()
        state = EffectState()
        hands = fists()

        self.assertEqual(controller.update(hands, 0.0, state), [])
        events = controller.update(hands, 0.31, state)

        self.assertEqual([event.kind for event in events], [GestureKind.HAND_FILL_OPEN])

    def test_fists_tolerate_one_noisy_outer_finger(self):
        controller = GestureController()
        state = EffectState()
        hands = fists_with_one_noisy_outer_finger()

        self.assertEqual(controller.update(hands, 0.0, state), [])
        events = controller.update(hands, 0.16, state)

        self.assertEqual([event.kind for event in events], [GestureKind.HAND_FILL_OPEN])

    def test_fists_accept_straight_scoring_thumbs_when_they_are_tucked(self):
        controller = GestureController()
        state = EffectState()
        hands = fists_with_straight_scoring_tucked_thumbs()

        self.assertEqual(controller.update(hands, 0.0, state), [])
        events = controller.update(hands, 0.16, state)

        self.assertEqual([event.kind for event in events], [GestureKind.HAND_FILL_OPEN])

    def test_thumbs_up_cannot_trigger_hand_panels(self):
        controller = GestureController()
        state = EffectState()
        hands = thumbs_up_with_other_fingers_curled()

        self.assertEqual(controller.update(hands, 0.0, state), [])
        events = controller.update(hands, 1.0, state)

        self.assertNotIn(GestureKind.HAND_FILL_OPEN, [event.kind for event in events])
        self.assertFalse(GestureController._both_fists(hands))

    def test_fists_close_only_hand_panels_when_both_effects_are_active(self):
        controller = GestureController()
        state = EffectState(
            panel_active=True,
            panel_finger_ids=((8, 12), (8, 12)),
            hand_fill_active=True,
        )

        controller.update(fists(), 0.0, state)
        events = controller.update(fists(), 0.31, state)

        self.assertEqual([event.kind for event in events], [GestureKind.HAND_FILL_CLOSE])

    def test_fist_toggle_requires_release_before_closing_hand_panels(self):
        controller = GestureController()
        state = EffectState()
        fist_pose = fists()

        self.assertEqual(controller.update(fist_pose, 0.0, state), [])
        open_events = controller.update(fist_pose, 0.31, state)
        state.apply(open_events[0])
        self.assertEqual(controller.update(fist_pose, 0.4, state), [])

        open_hands = (make_hand(100, {}), make_hand(200, {}))
        controller.update(open_hands, 0.5, state)
        self.assertEqual(controller.update(fist_pose, 0.6, state), [])
        events = controller.update(fist_pose, 0.91, state)
        self.assertEqual([event.kind for event in events], [GestureKind.HAND_FILL_CLOSE])

    def test_thumb_and_index_out_cannot_trigger_hand_panels(self):
        controller = GestureController()
        state = EffectState()
        hands = thumb_index_out_with_other_fingers_curled()

        self.assertEqual(controller.update(hands, 0.0, state), [])
        events = controller.update(hands, 1.0, state)

        self.assertNotIn(GestureKind.HAND_FILL_OPEN, [event.kind for event in events])
        self.assertFalse(GestureController._both_fists(hands))

    def test_tracking_loss_does_not_clear_latched_state(self):
        controller = GestureController()
        state = EffectState(
            panel_active=True,
            panel_finger_ids=((4, 8), (12, 16)),
            hand_fill_active=True,
        )

        self.assertEqual(controller.update([], 3.0, state), [])
        self.assertTrue(state.panel_active)
        self.assertTrue(state.hand_fill_active)


if __name__ == "__main__":
    unittest.main()
