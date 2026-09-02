"""Tests for fruit projectiles and spawning."""

from __future__ import annotations

import random
import unittest

from game.fruit import Fruit
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
            fruit_type="apple",
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
        self.assertIn(fruit.fruit_type, ("apple", "orange", "watermelon"))

    def test_fruit_follows_arc_then_is_culled(self) -> None:
        fruit = self.manager.spawn(width=800, height=600)
        # Integrate until it falls back below the cull line.
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
        # Stay in the upper half of the window, still on-screen.
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

    def test_fruit_sprites_exist(self) -> None:
        from pathlib import Path

        folder = Path(__file__).resolve().parents[1] / "assets" / "fruits"
        for kind in ("apple", "orange", "watermelon"):
            self.assertTrue((folder / f"{kind}.png").is_file(), kind)


if __name__ == "__main__":
    unittest.main()
