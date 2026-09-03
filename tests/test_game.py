"""Tests for timed rounds and game-over."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from game.fruit import Fruit
from game.game import Game, combo_multiplier, combo_name, format_clock
from src import config


def _falling_fruit(y: float = 400) -> Fruit:
    return Fruit(
        x=100,
        y=y,
        velocity_x=0,
        velocity_y=400,
        radius=20,
        sliced=False,
        active=True,
        fruit_type="yuzu",
        color=(220, 70, 70),
    )


class ClockFormatTests(unittest.TestCase):
    def test_formats_minutes_and_seconds(self) -> None:
        self.assertEqual(format_clock(60), "1:00")
        self.assertEqual(format_clock(47), "0:47")
        self.assertEqual(format_clock(0), "0:00")
        self.assertEqual(format_clock(-1), "0:00")


class TimerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game(width=400, height=300)
        self.game.fruits._time_to_spawn = 10.0

    def tearDown(self) -> None:
        self.game.quit()

    def test_clock_block_width_does_not_change_with_digits(self) -> None:
        font = self.game._score_font
        self.assertIsNotNone(font)
        assert font is not None
        wide = self.game._ninja_clock(font, "0:00")
        narrow = self.game._ninja_clock(font, "1:11")
        self.assertEqual(wide.get_width(), narrow.get_width())

    def test_starts_with_a_full_round(self) -> None:
        self.assertAlmostEqual(self.game.time_left, float(config.ROUND_SECONDS))
        self.assertFalse(self.game.game_over)
        self.assertEqual(self.game.high_score, 0)

    def test_misses_do_not_end_the_round(self) -> None:
        self.game.fruits.fruits = [_falling_fruit() for _ in range(3)]
        self.game.update()
        self.assertFalse(self.game.game_over)
        self.assertGreater(self.game.time_left, 0.0)

    def test_timer_expiring_ends_the_round(self) -> None:
        self.game.time_left = 0.0
        self.game.update()
        self.assertTrue(self.game.game_over)
        self.assertEqual(self.game.time_left, 0.0)

    def test_game_over_updates_session_best(self) -> None:
        self.game.high_score = 7
        self.game.score = 12
        self.game.time_left = 0.0
        self.game.update()
        self.assertTrue(self.game.game_over)
        self.assertEqual(self.game.score, 12)
        self.assertEqual(self.game.high_score, 12)

    def test_game_over_keeps_existing_session_best(self) -> None:
        self.game.high_score = 20
        self.game.score = 3
        self.game.time_left = 0.0
        self.game.update()
        self.assertEqual(self.game.high_score, 20)

    def test_restart_keeps_session_best(self) -> None:
        self.game.score = 9
        self.game.time_left = 0.0
        self.game.update()
        self.assertTrue(self.game.game_over)
        self.game.restart()
        self.assertFalse(self.game.game_over)
        self.assertAlmostEqual(self.game.time_left, float(config.ROUND_SECONDS))
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.game.fruits.fruits, [])
        self.assertEqual(self.game.high_score, 9)

    def test_no_slicing_after_time_expires(self) -> None:
        leftover = Fruit(
            x=200,
            y=80,
            velocity_x=0,
            velocity_y=0,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="orange",
            color=(240, 150, 50),
        )
        self.game.fruits.fruits = [leftover]
        self.game.time_left = 0.0
        self.game.blade_active = True
        self.game.blade_segments = [(0.0, 80.0), (400.0, 80.0)]
        self.game.update()
        self.assertTrue(self.game.game_over)
        self.assertFalse(leftover.sliced)
        self.assertEqual(self.game.score, 0)

    def test_slicing_still_scores_during_the_round(self) -> None:
        fruit = Fruit(
            x=200,
            y=80,
            velocity_x=0,
            velocity_y=0,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="yuzu",
            color=(220, 70, 70),
        )
        self.game.fruits.fruits = [fruit]
        self.game.fruits._time_to_spawn = 10.0
        self.game.blade_active = True
        self.game.blade_segments = [(0.0, 80.0), (400.0, 80.0)]
        self.game.update()
        self.assertFalse(self.game.game_over)
        self.assertEqual(self.game.score, 1)
        self.assertTrue(fruit.sliced)


class PauseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game(width=400, height=300)
        self.game.fruits._time_to_spawn = 10.0

    def tearDown(self) -> None:
        self.game.quit()

    def test_held_palm_pauses_after_debounce(self) -> None:
        for _ in range(config.PALM_PAUSE_FRAMES - 1):
            self.game.set_palm(True)
        self.assertFalse(self.game.paused)
        self.game.set_palm(True)
        self.assertTrue(self.game.paused)

    def test_pointing_resumes_after_debounce(self) -> None:
        self.game.paused = True
        for _ in range(config.PALM_RESUME_FRAMES - 1):
            self.game.set_palm(False)
        self.assertTrue(self.game.paused)
        self.game.set_palm(False)
        self.assertFalse(self.game.paused)

    def test_lost_hand_does_not_resume(self) -> None:
        self.game.paused = True
        self.game.set_palm(None)
        self.assertTrue(self.game.paused)

    def test_pause_freezes_timer_and_fruit(self) -> None:
        fruit = _falling_fruit(y=200)
        self.game.fruits.fruits = [fruit]
        self.game.paused = True
        before = self.game.time_left
        y = fruit.y
        self.game.update()
        self.assertAlmostEqual(self.game.time_left, before)
        self.assertAlmostEqual(fruit.y, y)

    def test_pause_does_not_slice(self) -> None:
        fruit = Fruit(
            x=200,
            y=80,
            velocity_x=0,
            velocity_y=0,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="yuzu",
            color=(220, 70, 70),
        )
        self.game.fruits.fruits = [fruit]
        self.game.paused = True
        self.game.blade_active = True
        self.game.blade_segments = [(0.0, 80.0), (400.0, 80.0)]
        self.game.update()
        self.assertFalse(fruit.sliced)
        self.assertEqual(self.game.score, 0)

    def test_palm_ignored_after_time_expires(self) -> None:
        self.game.time_left = 0.0
        self.game.update()
        self.assertTrue(self.game.game_over)
        for _ in range(config.PALM_PAUSE_FRAMES):
            self.game.set_palm(True)
        self.assertFalse(self.game.paused)

    def test_restart_clears_pause(self) -> None:
        self.game.paused = True
        self.game.restart()
        self.assertFalse(self.game.paused)


class ComboTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game(width=400, height=300)
        self.game.fruits._time_to_spawn = 10.0

    def tearDown(self) -> None:
        self.game.quit()

    def _fruit(self, x: float, y: float = 80) -> Fruit:
        return Fruit(
            x=x,
            y=y,
            velocity_x=0,
            velocity_y=0,
            radius=20,
            sliced=False,
            active=True,
            fruit_type="yuzu",
            color=(220, 70, 70),
        )

    def test_multiplier_ramps_every_three_hits(self) -> None:
        self.assertEqual(combo_multiplier(0), 1)
        self.assertEqual(combo_multiplier(2), 1)
        self.assertEqual(combo_multiplier(3), 2)
        self.assertEqual(combo_multiplier(6), 3)
        self.assertEqual(combo_multiplier(99), config.MAX_MULTIPLIER)

    def test_three_in_one_swipe_builds_streak_and_x2(self) -> None:
        self.game.fruits.fruits = [self._fruit(80), self._fruit(200), self._fruit(320)]
        self.game.blade_active = True
        self.game.blade_segments = [(0.0, 80.0), (400.0, 80.0)]
        self.game.update()
        self.assertEqual(self.game.streak, 3)
        self.assertEqual(self.game.multiplier, 2)
        self.assertEqual(self.game.score, 4)

    def test_swipe_combo_awards_bonus_when_slash_ends(self) -> None:
        self.game.fruits.fruits = [self._fruit(80), self._fruit(200), self._fruit(320)]
        self.game.blade_active = True
        self.game.blade_segments = [(0.0, 80.0), (400.0, 80.0)]
        self.game.update()
        self.game.blade_active = False
        self.game.update()
        self.assertEqual(self.game.score, 4 + 3 * 2)
        self.assertTrue(any("NICE" in item.text for item in self.game._floaters))

    def test_combo_names_follow_streak(self) -> None:
        self.assertIsNone(combo_name(2))
        self.assertEqual(combo_name(3), "NICE")
        self.assertEqual(combo_name(5), "AWESOME")
        self.assertEqual(combo_name(8), "INSANE")
        self.assertEqual(combo_name(15), "YUZU NINJA")

    def test_streak_resets_after_the_window(self) -> None:
        self.game.streak = 5
        self.game.multiplier = 2
        self.game._combo_timer = 0.0
        self.game._tick_combo(0.02)
        self.assertEqual(self.game.streak, 0)
        self.assertEqual(self.game.multiplier, 1)

    def test_restart_clears_streak(self) -> None:
        self.game.streak = 9
        self.game.multiplier = 4
        self.game.restart()
        self.assertEqual(self.game.streak, 0)
        self.assertEqual(self.game.multiplier, 1)
