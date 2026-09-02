"""Tests for the blade trail."""

from __future__ import annotations

import math
import unittest

from src.blade import Blade, catmull_rom


class BladeTests(unittest.TestCase):
    def test_expires_old_points(self) -> None:
        blade = Blade(history_ms=100, min_step_px=0.0)
        blade.add(0, 0, timestamp=0.0)
        blade.add(10, 0, timestamp=0.05)
        blade.add(20, 0, timestamp=0.2)
        self.assertEqual(blade.polyline(), [(20, 0)])

    def test_skips_tiny_steps(self) -> None:
        blade = Blade(history_ms=1000, min_step_px=2.0)
        blade.add(0, 0, timestamp=0.0)
        blade.add(0.5, 0.5, timestamp=0.01)
        self.assertEqual(len(blade.polyline()), 1)


class CatmullRomTests(unittest.TestCase):
    def test_passes_through_samples(self) -> None:
        pts = [(0.0, 0.0), (100.0, 50.0), (200.0, 0.0), (300.0, 80.0)]
        curve = catmull_rom(pts, segments=4)
        for p in pts:
            self.assertTrue(any(math.hypot(c[0] - p[0], c[1] - p[1]) < 1e-6 for c in curve))

    def test_straight_line_stays_straight(self) -> None:
        pts = [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0), (150.0, 150.0)]
        for x, y in catmull_rom(pts, segments=5):
            self.assertAlmostEqual(x, y, places=6)

    def test_short_input_passes_through(self) -> None:
        self.assertEqual(catmull_rom([(0.0, 0.0), (1.0, 1.0)], 8), [(0.0, 0.0), (1.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
