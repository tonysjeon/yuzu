"""Tests for coordinate mapping."""

from __future__ import annotations

import unittest

from src.coordinate_mapper import CoordinateMapper


class CoordinateMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(
            camera_width=1000,
            camera_height=1000,
            game_width=1000,
            game_height=1000,
            active_region={
                "left": 0.10,
                "right": 0.90,
                "top": 0.10,
                "bottom": 0.90,
            },
        )

    def test_active_region_center_maps_to_game_center(self) -> None:
        x, y = self.mapper.map(500, 500)
        self.assertAlmostEqual(x, 499.5, places=3)
        self.assertAlmostEqual(y, 499.5, places=3)

    def test_active_region_corners_map_to_game_corners(self) -> None:
        top_left = self.mapper.map(100, 100)
        top_right = self.mapper.map(900, 100)
        bottom_left = self.mapper.map(100, 900)
        bottom_right = self.mapper.map(900, 900)

        self.assertAlmostEqual(top_left[0], 0.0, places=3)
        self.assertAlmostEqual(top_left[1], 0.0, places=3)
        self.assertAlmostEqual(top_right[0], 999.0, places=3)
        self.assertAlmostEqual(top_right[1], 0.0, places=3)
        self.assertAlmostEqual(bottom_left[0], 0.0, places=3)
        self.assertAlmostEqual(bottom_left[1], 999.0, places=3)
        self.assertAlmostEqual(bottom_right[0], 999.0, places=3)
        self.assertAlmostEqual(bottom_right[1], 999.0, places=3)

    def test_out_of_bounds_values_are_clamped(self) -> None:
        outside = self.mapper.map(-50, 2000)
        self.assertAlmostEqual(outside[0], 0.0, places=3)
        self.assertAlmostEqual(outside[1], 999.0, places=3)

    def test_different_camera_and_game_sizes(self) -> None:
        mapper = CoordinateMapper(
            camera_width=1280,
            camera_height=720,
            game_width=640,
            game_height=360,
            active_region={
                "left": 0.0,
                "right": 1.0,
                "top": 0.0,
                "bottom": 1.0,
            },
        )
        x, y = mapper.map(640, 360)
        self.assertAlmostEqual(x, 319.5, places=3)
        self.assertAlmostEqual(y, 179.5, places=3)


if __name__ == "__main__":
    unittest.main()
