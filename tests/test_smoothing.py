"""Tests for fingertip smoothing filters."""

from __future__ import annotations

import math
import random
import unittest

from src.smoothing import AnchoredTip, ExponentialSmoother, OneEuroFilter


class ExponentialSmootherTests(unittest.TestCase):
    def test_first_sample_passes_through(self) -> None:
        s = ExponentialSmoother(alpha=0.4)
        self.assertEqual(s.update(10.0, 20.0), (10.0, 20.0))

    def test_blends_toward_new_value(self) -> None:
        s = ExponentialSmoother(alpha=0.5)
        s.update(0.0, 0.0)
        x, y = s.update(10.0, 10.0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 5.0)


class OneEuroFilterTests(unittest.TestCase):
    def test_reduces_jitter_when_still(self) -> None:
        rng = random.Random(0)
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
        outputs = []
        t = 0.0
        for _ in range(120):
            t += 1 / 30
            noise_x = rng.uniform(-6, 6)
            noise_y = rng.uniform(-6, 6)
            outputs.append(f.update(500 + noise_x, 300 + noise_y, t))
        xs = [p[0] for p in outputs[30:]]
        spread = max(xs) - min(xs)
        self.assertLess(spread, 6.0)

    def test_tracks_fast_motion_closely(self) -> None:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
        t = 0.0
        x = 0.0
        out = (0.0, 0.0)
        for _ in range(30):
            t += 1 / 30
            x += 40.0  # 1200 px/s swipe
            out = f.update(x, 0.0, t)
        lag = x - out[0]
        self.assertLess(lag, 40.0)

    def test_follows_sharp_turn_quickly(self) -> None:
        f = OneEuroFilter()
        t = 0.0
        path = [(i * 45.0, 0.0) for i in range(12)]
        path += [(11 * 45.0, (i + 1) * 45.0) for i in range(12)]
        lags = []
        for x, y in path:
            t += 1 / 30
            ox, oy = f.update(x, y, t)
            lags.append(math.hypot(ox - x, oy - y))
        self.assertLess(max(lags[12:18]), 12.0)

    def test_reset_clears_history(self) -> None:
        f = OneEuroFilter()
        f.update(0.0, 0.0, 0.0)
        f.update(100.0, 100.0, 0.033)
        f.reset()
        self.assertEqual(f.update(5.0, 6.0, 1.0), (5.0, 6.0))

    def test_advance_keeps_resume_continuous(self) -> None:
        f = OneEuroFilter()
        t = 0.0
        x = 0.0
        for _ in range(10):
            t += 1 / 30
            x += 40.0
            f.update(x, 0.0, t)
        # Two dropped frames: coast forward, then a real sample arrives on-path.
        t += 2 / 30
        x += 80.0
        f.advance(x, 0.0, t)
        t += 1 / 30
        x += 40.0
        out_x, _ = f.update(x, 0.0, t)
        self.assertLess(abs(out_x - x), 40.0)

    def test_zero_dt_is_safe(self) -> None:
        f = OneEuroFilter()
        f.update(1.0, 1.0, 1.0)
        x, y = f.update(50.0, 50.0, 1.0)
        self.assertFalse(math.isnan(x) or math.isnan(y))


class AnchoredTipTests(unittest.TestCase):
    def test_rejects_tip_dip_at_speed(self) -> None:
        a = AnchoredTip(slow_cutoff=4.0, fast_cutoff=0.5, full_speed=1000.0)
        dt = 1 / 30
        offset = (0.0, -120.0)  # finger pointing up from the palm
        out = (0.0, 0.0)
        for i in range(12):
            palm = (i * 50.0, 400.0)
            tip = (palm[0] + offset[0], palm[1] + offset[1])
            if i == 8:
                tip = (tip[0], tip[1] + 80.0)  # blurred tip lands near the palm
            out = a.update(palm, tip, speed=1500.0, dt=dt)
            if i == 8:
                self.assertLess(abs(out[1] - (palm[1] + offset[1])), 12.0)
        self.assertAlmostEqual(out[0], 11 * 50.0, places=6)

    def test_follows_finger_at_rest(self) -> None:
        a = AnchoredTip(slow_cutoff=4.0, fast_cutoff=0.5, full_speed=1000.0)
        palm = (300.0, 300.0)
        a.update(palm, (300.0, 180.0), speed=0.0, dt=1 / 30)
        out = (0.0, 0.0)
        for _ in range(15):
            out = a.update(palm, (360.0, 200.0), speed=0.0, dt=1 / 30)
        self.assertLess(math.hypot(out[0] - 360.0, out[1] - 200.0), 3.0)


if __name__ == "__main__":
    unittest.main()
