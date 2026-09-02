"""Tests for swipe detection."""

from __future__ import annotations

import unittest

from src.models import FingertipPoint
from src.swipe_detector import SwipeDetector


def _points(path: list[tuple[float, float]], dt: float = 1 / 30) -> list[FingertipPoint]:
    return [
        FingertipPoint(x=x, y=y, timestamp=i * dt, confidence=1.0)
        for i, (x, y) in enumerate(path)
    ]


class SwipeDetectorTests(unittest.TestCase):
    def test_slow_motion_stays_inactive(self) -> None:
        # ~150 px/s crawl across 5 frames.
        pts = _points([(i * 5.0, 0.0) for i in range(5)])
        result = SwipeDetector(min_velocity=500, hold_ms=0).update(pts)
        self.assertFalse(result.active)
        self.assertLess(result.velocity, 500)

    def test_fast_swipe_activates(self) -> None:
        # ~1500 px/s slash.
        pts = _points([(i * 50.0, 0.0) for i in range(5)])
        result = SwipeDetector(min_velocity=500, hold_ms=0).update(pts)
        self.assertTrue(result.active)
        self.assertGreaterEqual(result.velocity, 500)
        self.assertEqual(result.start, (0.0, 0.0))
        self.assertEqual(result.end, (200.0, 0.0))

    def test_curved_path_uses_path_length(self) -> None:
        # Right then down: start→end is short, path length is long.
        pts = _points([(0, 0), (100, 0), (100, 100)])
        result = SwipeDetector(min_velocity=500, hold_ms=0).update(pts)
        # Path = 200 over 2/30 s ≈ 3000 px/s.
        self.assertGreater(result.velocity, 2500)
        self.assertTrue(result.active)

    def test_hold_keeps_blade_briefly(self) -> None:
        detector = SwipeDetector(min_velocity=500, hold_ms=100)
        fast = _points([(i * 50.0, 0.0) for i in range(4)])
        self.assertTrue(detector.update(fast).active)
        # Fast window ends at 0.1s; still samples within the 100 ms hold.
        still = [
            FingertipPoint(x=150.0, y=0.0, timestamp=0.12, confidence=1.0),
            FingertipPoint(x=150.5, y=0.0, timestamp=0.15, confidence=1.0),
        ]
        held = detector.update(still, now=0.15)
        self.assertTrue(held.active)
        # After the hold window expires, idle again.
        later = [
            FingertipPoint(x=151.0, y=0.0, timestamp=0.25, confidence=1.0),
            FingertipPoint(x=151.5, y=0.0, timestamp=0.28, confidence=1.0),
        ]
        self.assertFalse(detector.update(later, now=0.28).active)

    def test_reset_clears_hold(self) -> None:
        detector = SwipeDetector(min_velocity=500, hold_ms=500)
        fast = _points([(i * 50.0, 0.0) for i in range(4)])
        self.assertTrue(detector.update(fast).active)
        detector.reset()
        still = [
            FingertipPoint(x=0.0, y=0.0, timestamp=1.0, confidence=1.0),
            FingertipPoint(x=1.0, y=0.0, timestamp=1.03, confidence=1.0),
        ]
        self.assertFalse(detector.update(still, now=1.03).active)

    def test_single_point_is_inactive(self) -> None:
        result = SwipeDetector().update(
            [FingertipPoint(x=1.0, y=2.0, timestamp=0.0, confidence=1.0)]
        )
        self.assertFalse(result.active)
        self.assertIsNone(result.start)


if __name__ == "__main__":
    unittest.main()
