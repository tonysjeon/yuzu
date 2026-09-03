"""Tests for fruit projectiles and spawning."""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

from game.fruit import Fruit, _flesh_albedo
from game.fruit_manager import FruitManager


class FruitPhysicsTests(unittest.TestCase):
    def test_gravity_pulls_fruit_down(self) -> None:
        fruit = Fruit(
            x=100,
            y=400,
            velocity_x=0,
            velocity_y=-900,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="yuzu",
            color=(220, 70, 70),
        )
        fruit.update(0.1, gravity=1100)
        # Velocity is integrated before position (semi-implicit Euler).
        self.assertAlmostEqual(fruit.velocity_y, -790.0, places=3)
        self.assertAlmostEqual(fruit.y, 400 + (-790) * 0.1, places=3)

    def test_inactive_fruit_does_not_move(self) -> None:
        fruit = Fruit(
            x=50,
            y=50,
            velocity_x=100,
            velocity_y=-100,
            radius=10,
            sliced=False,
            active=False,
            fruit_type="orange",
            color=(240, 150, 50),
        )
        fruit.update(1.0)
        self.assertEqual((fruit.x, fruit.y), (50, 50))


class FruitManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = FruitManager(
            min_interval=1.0,
            max_interval=1.0,
            gravity=1100,
            rng=random.Random(0),
        )

    def test_spawns_after_interval(self) -> None:
        for _ in range(30):
            self.manager.update(1 / 60, width=800, height=600)
        self.assertEqual(len(self.manager.active_fruits), 0)
        for _ in range(35):
            self.manager.update(1 / 60, width=800, height=600)
        self.assertEqual(len(self.manager.active_fruits), 1)

    def test_spawn_starts_below_screen(self) -> None:
        fruit = self.manager.spawn(width=800, height=600)
        self.assertGreater(fruit.y, 600)
        self.assertLess(fruit.velocity_y, 0)
        self.assertIn(fruit.fruit_type, ("yuzu", "orange", "watermelon"))

    def test_fruit_follows_arc_then_is_culled(self) -> None:
        fruit = self.manager.spawn(width=800, height=600)
        for _ in range(300):
            self.manager.update(1 / 60, width=800, height=600)
            if fruit not in self.manager.fruits:
                break
        self.assertNotIn(fruit, self.manager.fruits)

    def test_arc_peaks_above_spawn(self) -> None:
        fruit = self.manager.spawn(width=800, height=600)
        start_y = fruit.y
        peak = start_y
        for _ in range(180):
            fruit.update(1 / 60, gravity=1100)
            peak = min(peak, fruit.y)
        self.assertLess(peak, start_y - 250)
        self.assertGreater(peak, -fruit.radius)

    def test_toss_stays_inside_screen(self) -> None:
        width, height = 800, 600
        for seed in range(20):
            manager = FruitManager(min_interval=10, max_interval=10, rng=random.Random(seed))
            fruit = manager.spawn(width, height)
            pad = fruit.radius + 28
            for _ in range(240):
                fruit.update(1 / 60, gravity=1100)
                self.assertGreaterEqual(fruit.x, pad - 2)
                self.assertLessEqual(fruit.x, width - pad + 2)
                if fruit.y - fruit.radius > height + 40:
                    break

    def test_arcs_stay_near_center(self) -> None:
        width, height = 800, 600
        center = width * 0.5
        for seed in range(20):
            fruit = FruitManager(
                min_interval=10, max_interval=10, rng=random.Random(seed)
            ).spawn(width, height)
            self.assertLess(abs(fruit.x - center), width * 0.22)
            peak_x = fruit.x
            for _ in range(180):
                fruit.update(1 / 60, gravity=1100)
                if fruit.velocity_y >= 0:
                    break
                peak_x = fruit.x
            self.assertLess(abs(peak_x - center), width * 0.18)

    def test_fruit_sprites_exist(self) -> None:
        from pathlib import Path

        folder = Path(__file__).resolve().parents[1] / "assets" / "fruits"
        for kind in ("yuzu", "orange", "watermelon"):
            self.assertTrue((folder / f"{kind}.png").is_file(), kind)

    def test_hud_fonts_exist(self) -> None:
        from pathlib import Path

        folder = Path(__file__).resolve().parents[1] / "assets" / "fonts"
        for name in ("Bungee-Regular.ttf", "Outfit-ExtraBold.ttf", "Outfit-Medium.ttf"):
            self.assertTrue((folder / name).is_file(), name)

    def test_falling_unsliced_fruit_counts_as_miss(self) -> None:
        fruit = Fruit(
            x=100,
            y=700,
            velocity_x=0,
            velocity_y=200,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="yuzu",
            color=(220, 70, 70),
        )
        self.manager.fruits = [fruit]
        self.manager._time_to_spawn = 10.0
        misses = self.manager.update(1 / 60, width=800, height=600)
        self.assertEqual(misses, 1)
        self.assertEqual(self.manager.fruits, [])

    def test_rising_fruit_below_screen_is_not_a_miss(self) -> None:
        fruit = Fruit(
            x=100,
            y=700,
            velocity_x=0,
            velocity_y=-400,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="yuzu",
            color=(220, 70, 70),
        )
        self.manager.fruits = [fruit]
        self.manager._time_to_spawn = 10.0
        misses = self.manager.update(1 / 60, width=800, height=600)
        self.assertEqual(misses, 0)
        self.assertEqual(self.manager.fruits, [])

    def test_sliced_halves_falling_off_are_not_misses(self) -> None:
        fruit = Fruit(
            x=100,
            y=100,
            velocity_x=0,
            velocity_y=0,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="yuzu",
            color=(220, 70, 70),
        )
        self.manager.fruits = [fruit]
        self.manager._time_to_spawn = 10.0
        self.manager.slice_with_blade([(0.0, 100.0), (200.0, 100.0)])
        for piece in self.manager.pieces:
            piece.y = 700
            piece.velocity_y = 200
        misses = self.manager.update(1 / 60, width=800, height=600, spawn=False)
        self.assertEqual(misses, 0)
        self.assertEqual(self.manager.pieces, [])

    def test_spawn_false_does_not_spawn(self) -> None:
        self.manager._time_to_spawn = 0.0
        misses = self.manager.update(1 / 60, width=800, height=600, spawn=False)
        self.assertEqual(misses, 0)
        self.assertEqual(self.manager.active_fruits, [])

    def test_spawn_intervals_shrink_as_progress_rises(self) -> None:
        self.manager.set_progress(0.0)
        early = [self.manager._next_interval() for _ in range(40)]
        self.manager.set_progress(1.0)
        late = [self.manager._next_interval() for _ in range(40)]
        self.assertGreater(sum(early) / len(early), sum(late) / len(late))
        self.assertLessEqual(max(late), self.manager.max_interval_end + 1e-6)

    def test_late_round_can_double_spawn(self) -> None:
        manager = FruitManager(
            min_interval=0.01,
            max_interval=0.01,
            min_interval_end=0.01,
            max_interval_end=0.01,
            rng=random.Random(1),
        )
        manager.set_progress(1.0)
        manager._time_to_spawn = 0.0
        manager.update(1 / 60, width=800, height=600)
        self.assertGreaterEqual(len(manager.fruits), 2)


class CutVisualTests(unittest.TestCase):
    def test_watermelon_half_matches_whole_fruit_size(self) -> None:
        fruit = Fruit(
            x=200,
            y=200,
            velocity_x=0,
            velocity_y=0,
            radius=73,
            sliced=False,
            active=True,
            fruit_type="watermelon",
            color=(70, 180, 90),
        )
        left, right = fruit.split(1.0, 0.0)
        self.assertEqual(left.display_size, fruit.sprite_size())
        self.assertEqual(right.display_size, fruit.sprite_size())
        self.assertIsNotNone(left.sprite)
        assert left.sprite is not None
        # Must not draw at half the 256px raster (the old undersized look).
        self.assertGreater(left.display_size, left.sprite.get_width() * 0.55)

    def test_slice_spawns_juice_splatter(self) -> None:
        manager = FruitManager(min_interval=10, max_interval=10, rng=random.Random(0))
        fruit = Fruit(
            x=200,
            y=80,
            velocity_x=0,
            velocity_y=0,
            radius=40,
            sliced=False,
            active=True,
            fruit_type="watermelon",
            color=(70, 180, 90),
        )
        manager.fruits = [fruit]
        manager.slice_with_blade([(0.0, 80.0), (400.0, 80.0)])
        self.assertGreater(len(manager.splatter.drops), 8)

    def test_splatter_expires(self) -> None:
        manager = FruitManager(min_interval=10, max_interval=10, rng=random.Random(0))
        manager.splatter.burst(100, 100, 1.0, 0.0, "yuzu", 20)
        self.assertGreater(len(manager.splatter.drops), 0)
        for _ in range(90):
            manager.splatter.update(1 / 60)
        self.assertEqual(manager.splatter.drops, [])


class WatermelonFleshTests(unittest.TestCase):
    def test_cut_face_is_red_with_green_rind_and_dark_seeds(self) -> None:
        n = 96
        u = np.linspace(-1.0, 1.0, n, dtype=np.float32)[:, None]
        v = np.linspace(-1.0, 1.0, n, dtype=np.float32)[None, :]
        rgb = _flesh_albedo("watermelon", u, v)
        heart = rgb[n // 2, n // 2]
        self.assertGreater(float(heart[0]), float(heart[1]))
        self.assertGreater(float(heart[0]), 140.0)
        rind = rgb[n - 2, n // 2]
        self.assertGreater(float(rind[1]), float(rind[0]))
        dark = rgb.min(axis=(0, 1))
        self.assertLess(float(dark[0]), 50.0)

    def test_flesh_has_angular_variation_unlike_flat_clipart(self) -> None:
        n = 96
        u = np.linspace(-1.0, 1.0, n, dtype=np.float32)[:, None]
        v = np.linspace(-1.0, 1.0, n, dtype=np.float32)[None, :]
        rgb = _flesh_albedo("watermelon", u, v)
        # Mid-radius ring should vary around the circle (fibers / seeds).
        r = 0.45
        samples = []
        for i in range(24):
            a = i * (2.0 * math.pi / 24)
            x = int((r * math.cos(a) * 0.5 + 0.5) * (n - 1))
            y = int((r * math.sin(a) * 0.5 + 0.5) * (n - 1))
            samples.append(float(rgb[x, y, 0]))
        self.assertGreater(max(samples) - min(samples), 25.0)


if __name__ == "__main__":
    unittest.main()
