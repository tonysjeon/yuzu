"""Tests for the dojo wall background."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from game.background import build_dojo_wall


class DojoWallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((64, 48))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_matches_requested_size(self) -> None:
        wall = build_dojo_wall(320, 240)
        self.assertEqual(wall.get_size(), (320, 240))

    def test_center_is_warm_wood(self) -> None:
        wall = build_dojo_wall(320, 240)
        r, g, b, *_ = wall.get_at((160, 110))
        self.assertGreater(r, g)
        self.assertGreater(g, b)
        self.assertGreater(r, 18)
        self.assertLess(r, 110)

    def test_corners_are_darker_than_center(self) -> None:
        wall = build_dojo_wall(320, 240)
        center = wall.get_at((160, 110))
        corner = wall.get_at((8, 8))
        self.assertGreater(center[0] + center[1], corner[0] + corner[1])

    def test_planks_run_vertically(self) -> None:
        wall = build_dojo_wall(320, 240)
        row = [wall.get_at((x, 120))[0] for x in range(16, 304, 8)]
        self.assertGreater(max(row) - min(row), 8)
