"""Spawn and integrate active fruits."""

from __future__ import annotations

import math
import random

from src import config

from game.collision import blade_cut_axis
from game.fruit import FRUIT_STYLES, Fruit, FruitHalf
from game.splatter import Splatter


class FruitManager:
    """Timed fruit spawns with simple projectile physics."""

    def __init__(
        self,
        min_interval: float = config.FRUIT_MIN_SPAWN_INTERVAL,
        max_interval: float = config.FRUIT_MAX_SPAWN_INTERVAL,
        min_interval_end: float = config.FRUIT_MIN_SPAWN_INTERVAL_END,
        max_interval_end: float = config.FRUIT_MAX_SPAWN_INTERVAL_END,
        gravity: float = config.GRAVITY,
        rng: random.Random | None = None,
    ) -> None:
        if max_interval < min_interval:
            raise ValueError("max_interval must be >= min_interval")
        if max_interval_end < min_interval_end:
            raise ValueError("max_interval_end must be >= min_interval_end")
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.min_interval_end = min_interval_end
        self.max_interval_end = max_interval_end
        self.gravity = gravity
        self._rng = rng if rng is not None else random.Random()
        self.fruits: list[Fruit] = []
        self.pieces: list[FruitHalf] = []
        self.splatter = Splatter(self._rng)
        self.last_slice_at: tuple[float, float] | None = None
        self._progress = 0.0
        self._time_to_spawn = self._next_interval()

    def set_progress(self, progress: float) -> None:
        """Round difficulty from 0 (start) to 1 (time's up)."""
        self._progress = max(0.0, min(float(progress), 1.0))

    def _lerp(self, a: float, b: float) -> float:
        return a + (b - a) * self._progress

    def _next_interval(self) -> float:
        lo = self._lerp(self.min_interval, self.min_interval_end)
        hi = self._lerp(self.max_interval, self.max_interval_end)
        if hi < lo:
            lo, hi = hi, lo
        return self._rng.uniform(lo, hi)

    def spawn(self, width: int, height: int) -> Fruit:
        """Toss one fruit from below the screen so its arc stays in view."""
        kind = self._rng.choice(list(FRUIT_STYLES.keys()))
        color, radius = FRUIT_STYLES[kind]
        pad = radius + config.FRUIT_SIDE_PAD
        center = width * 0.5
        spawn_span = max(width * config.FRUIT_SPAWN_HALF_SPAN, 1.0)
        x = self._rng.uniform(center - spawn_span, center + spawn_span)
        x = min(max(x, pad), width - pad)
        y = height + config.FRUIT_SPAWN_MARGIN_Y + radius

        peak_frac = self._rng.uniform(config.FRUIT_PEAK_MIN, config.FRUIT_PEAK_MAX)
        peak_y = peak_frac * height
        travel = max(y - peak_y, 80.0)
        speed = math.sqrt(2.0 * self.gravity * travel)
        velocity_y = -speed

        # Aim the top of the arc at the middle of the screen.
        flight_up = max(speed / self.gravity, 1e-3)
        peak_span = width * config.FRUIT_PEAK_HALF_SPAN
        target_x = self._rng.uniform(center - peak_span, center + peak_span)
        target_x = min(max(target_x, pad), width - pad)
        velocity_x = (target_x - x) / flight_up

        # Still clamp so the fruit cannot leave the side pads on the way down.
        flight = max(2.0 * speed / self.gravity, 1e-3)
        vx_min = -(x - pad) / flight
        vx_max = (width - pad - x) / flight
        if vx_min <= vx_max:
            velocity_x = min(max(velocity_x, vx_min), vx_max)
        else:
            velocity_x = 0.0

        fruit = Fruit(
            x=x,
            y=y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            radius=radius,
            sliced=False,
            active=True,
            fruit_type=kind,
            color=color,
            rotation=self._rng.uniform(0.0, 360.0),
            rotation_speed=self._rng.uniform(-220.0, 220.0),
        )
        self.fruits.append(fruit)
        return fruit

    def update(self, dt: float, width: int, height: int, spawn: bool = True) -> int:
        """Advance physics. Return how many unsliced fruits fell off-screen."""
        dt = max(0.0, min(dt, 0.05))
        if spawn:
            self._time_to_spawn -= dt
            if self._time_to_spawn <= 0.0:
                self.spawn(width, height)
                if (
                    self._progress >= config.FRUIT_DOUBLE_SPAWN_START
                    and self._rng.random() < config.FRUIT_DOUBLE_SPAWN_CHANCE
                ):
                    self.spawn(width, height)
                self._time_to_spawn = self._next_interval()

        for fruit in self.fruits:
            fruit.update(dt, gravity=self.gravity)
        for piece in self.pieces:
            piece.update(dt, gravity=self.gravity)
        self.splatter.update(dt, gravity=self.gravity)

        misses = 0
        keep: list[Fruit] = []
        cull = height + config.FRUIT_SPAWN_MARGIN_Y
        for fruit in self.fruits:
            if not fruit.active:
                continue
            if fruit.y - fruit.radius > cull:
                fruit.active = False
                # Spawned below the screen while rising is not a miss.
                if fruit.velocity_y > 0 and not fruit.sliced:
                    misses += 1
                continue
            keep.append(fruit)
        self.fruits = keep

        keep_pieces: list[FruitHalf] = []
        for piece in self.pieces:
            if not piece.active:
                continue
            if piece.y - piece.radius > height + config.FRUIT_SPAWN_MARGIN_Y:
                piece.active = False
                continue
            keep_pieces.append(piece)
        self.pieces = keep_pieces
        return misses

    def slice_with_blade(self, points: list[tuple[float, float]]) -> int:
        """Split fruits the blade grazes into two flying halves."""
        sliced = 0
        keep: list[Fruit] = []
        for fruit in self.fruits:
            if not fruit.active or fruit.sliced:
                continue
            axis = blade_cut_axis(points, fruit.x, fruit.y, fruit.radius)
            if axis is None:
                keep.append(fruit)
                continue
            fruit.sliced = True
            fruit.active = False
            self.pieces.extend(fruit.split(*axis))
            self.splatter.burst(
                fruit.x, fruit.y, axis[0], axis[1], fruit.fruit_type, fruit.radius
            )
            self.last_slice_at = (fruit.x, fruit.y)
            sliced += 1
        self.fruits = keep
        return sliced

    def clear(self) -> None:
        self.fruits.clear()
        self.pieces.clear()
        self.splatter.clear()
        self.last_slice_at = None
        self._progress = 0.0
        self._time_to_spawn = self._next_interval()

    @property
    def active_fruits(self) -> list[Fruit]:
        return [fruit for fruit in self.fruits if fruit.active]
