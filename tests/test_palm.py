"""Tests for open-palm pause gesture."""

from __future__ import annotations

import unittest

from src.hand_tracker import is_fist, is_open_palm


def _hand(
    *,
    index_tip: tuple[float, float],
    middle_tip: tuple[float, float],
    ring_tip: tuple[float, float],
    pinky_tip: tuple[float, float],
) -> list[tuple[float, float, float]]:
    pts = [(0.0, 0.0, 0.0)] * 21
    pts[0] = (100.0, 200.0, 0.0)  # wrist
    pts[5] = (80.0, 140.0, 0.0)  # index MCP
    pts[9] = (100.0, 130.0, 0.0)  # middle MCP; palm size = 70
    pts[13] = (118.0, 140.0, 0.0)
    pts[17] = (132.0, 152.0, 0.0)
    pts[8] = (*index_tip, 0.0)
    pts[12] = (*middle_tip, 0.0)
    pts[16] = (*ring_tip, 0.0)
    pts[20] = (*pinky_tip, 0.0)
    return pts


def _curled() -> list[tuple[float, float, float]]:
    return _hand(
        index_tip=(82.0, 148.0),
        middle_tip=(100.0, 138.0),
        ring_tip=(118.0, 148.0),
        pinky_tip=(132.0, 158.0),
    )


def _open_palm() -> list[tuple[float, float, float]]:
    return _hand(
        index_tip=(80.0, 40.0),
        middle_tip=(100.0, 35.0),
        ring_tip=(120.0, 40.0),
        pinky_tip=(135.0, 50.0),
    )


class PalmDetectionTests(unittest.TestCase):
    def test_open_palm_pauses(self) -> None:
        self.assertTrue(is_open_palm(_open_palm()))

    def test_pointing_index_is_not_a_palm(self) -> None:
        pts = _curled()
        pts[8] = (70.0, 40.0, 0.0)
        self.assertFalse(is_open_palm(pts))

    def test_fist_is_not_a_palm(self) -> None:
        self.assertFalse(is_open_palm(_curled()))
        self.assertTrue(is_fist(_curled()))

    def test_open_palm_is_not_a_fist(self) -> None:
        self.assertFalse(is_fist(_open_palm()))

    def test_pointing_index_is_not_a_fist(self) -> None:
        pts = _curled()
        pts[8] = (70.0, 40.0, 0.0)
        self.assertFalse(is_fist(pts))

    def test_short_landmark_list_is_not_a_palm(self) -> None:
        self.assertFalse(is_open_palm([]))
        self.assertFalse(is_open_palm([(0.0, 0.0, 0.0)] * 10))
