"""Tests for blade-versus-fruit collision."""

from __future__ import annotations

import math
import unittest

from game.collision import blade_hits_fruit, point_segment_distance, segment_hits_circle
from game.fruit import Fruit
from game.fruit_manager import FruitManager


class SegmentCircleTests(unittest.TestCase):
    def test_endpoint_inside_circle_hits(self) -> None:
        self.assertTrue(segment_hits_circle(0, 0, 10, 0, 0, 0, 5))

    def test_segment_grazing_circle_hits(self) -> None:
        # Horizontal swipe above the fruit; closest point is the midpoint.
        self.assertTrue(segment_hits_circle(0, 0, 100, 0, 50, 8, 10))

    def test_segment_missing_circle_does_not_hit(self) -> None:
        self.assertFalse(segment_hits_circle(0, 0, 100, 0, 50, 40, 10))

    def test_distance_clamps_to_endpoint(self) -> None:
        dist = point_segment_distance(200, 0, 0, 0, 10, 0)
        self.assertAlmostEqual(dist, 190.0, places=3)


class BladeHitsFruitTests(unittest.TestCase):
    def test_swipe_through_fruit_without_sample_inside(self) -> None:
        # Samples jump from left of the fruit to the right of it.
        points = [(0.0, 0.0), (80.0, 0.0)]
        self.assertTrue(blade_hits_fruit(points, 40.0, 0.0, 12.0, blade_radius=0.0))

    def test_idle_single_point_does_not_hit(self) -> None:
        self.assertFalse(blade_hits_fruit([(40.0, 0.0)], 40.0, 0.0, 12.0))

    def test_parallel_miss(self) -> None:
        points = [(0.0, 40.0), (80.0, 40.0)]
        self.assertFalse(blade_hits_fruit(points, 40.0, 0.0, 12.0, blade_radius=0.0))


class SliceWithBladeTests(unittest.TestCase):
    def test_sliced_fruit_is_removed_and_counted(self) -> None:
        manager = FruitManager(min_interval=10, max_interval=10)
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
        manager.fruits = [fruit]
        n = manager.slice_with_blade([(0.0, 100.0), (200.0, 100.0)])
        self.assertEqual(n, 1)
        self.assertTrue(fruit.sliced)
        self.assertEqual(manager.active_fruits, [])
        self.assertEqual(len(manager.pieces), 2)
        left, right = manager.pieces
        self.assertGreater(math.hypot(left.x - right.x, left.y - right.y), 4.0)

    def test_miss_leaves_fruit(self) -> None:
        manager = FruitManager(min_interval=10, max_interval=10)
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
        manager.fruits = [fruit]
        n = manager.slice_with_blade([(0.0, 0.0), (40.0, 0.0)])
        self.assertEqual(n, 0)
        self.assertFalse(fruit.sliced)
        self.assertEqual(len(manager.active_fruits), 1)


class SliceBombTests(unittest.TestCase):
    def test_sliced_bomb_does_not_split_or_score(self) -> None:
        manager = FruitManager(min_interval=10, max_interval=10)
        bomb = Fruit(
            x=100,
            y=100,
            velocity_x=0,
            velocity_y=0,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="bomb",
            color=(40, 42, 48),
        )
        manager.fruits = [bomb]
        n = manager.slice_with_blade([(0.0, 100.0), (200.0, 100.0)])
        self.assertEqual(n, 0)
        self.assertTrue(manager.hit_bomb)
        self.assertTrue(bomb.sliced)
        self.assertEqual(manager.pieces, [])
        self.assertEqual(manager.active_fruits, [])


if __name__ == "__main__":
    unittest.main()
