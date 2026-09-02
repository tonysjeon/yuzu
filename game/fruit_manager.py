"""Spawn and integrate active fruits."""

from __future__ import annotations

import math
import random

from src import config

from game.fruit import FRUIT_STYLES, Fruit


class FruitManager:
    """Timed fruit spawns with simple projectile physics."""

    def __init__(
        self,
        min_interval: float = config.FRUIT_MIN_SPAWN_INTERVAL,
        max_interval: float = config.FRUIT_MAX_SPAWN_INTERVAL,
        gravity: float = config.GRAVITY,
        rng: random.Random | None = None,
    ) -> None:
        if max_interval < min_interval:
            raise ValueError("max_interval must be >= min_interval")
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.gravity = gravity
        self._rng = rng if rng is not None else random.Random()
        self.fruits: list[Fruit] = []
        self._time_to_spawn = self._next_interval()

    def _next_interval(self) -> float:
        return self._rng.uniform(self.min_interval, self.max_interval)

    def spawn(self, width: int, height: int) -> Fruit:
        """Toss one fruit from below the screen so its arc stays in view."""
        kind = self._rng.choice(list(FRUIT_STYLES.keys()))
        color, radius = FRUIT_STYLES[kind]
        pad = radius + config.FRUIT_SIDE_PAD
        inner = max(width - 2 * pad, 1.0)
        # Spawn in the middle 70% so even a long arc has room to drift.
        x = pad + inner * self._rng.uniform(0.15, 0.85)
        y = height + config.FRUIT_SPAWN_MARGIN_Y + radius

        peak_frac = self._rng.uniform(config.FRUIT_PEAK_MIN, config.FRUIT_PEAK_MAX)
        peak_y = peak_frac * height
        travel = max(y - peak_y, 80.0)
        speed = math.sqrt(2.0 * self.gravity * travel)
        velocity_y = -speed

        # Horizontal speed is limited so the fruit cannot leave the side pads
        # before it falls back to the spawn height.
        flight = max(2.0 * speed / self.gravity, 1e-3)
        vx_min = -(x - pad) / flight
        vx_max = (width - pad - x) / flight
        if vx_min > vx_max:
            velocity_x = 0.0
        else:
            velocity_x = self._rng.uniform(vx_min, vx_max)

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

    def update(self, dt: float, width: int, height: int) -> None:
        dt = max(0.0, min(dt, 0.05))
        self._time_to_spawn -= dt
        if self._time_to_spawn <= 0.0:
            self.spawn(width, height)
            self._time_to_spawn = self._next_interval()

        for fruit in self.fruits:
            fruit.update(dt, gravity=self.gravity)

        # Drop fruits that have fallen back below the screen.
        keep: list[Fruit] = []
        for fruit in self.fruits:
            if not fruit.active:
                continue
            if fruit.y - fruit.radius > height + config.FRUIT_SPAWN_MARGIN_Y:
                fruit.active = False
                continue
            keep.append(fruit)
        self.fruits = keep

    def clear(self) -> None:
        self.fruits.clear()
        self._time_to_spawn = self._next_interval()

    @property
    def active_fruits(self) -> list[Fruit]:
        return [fruit for fruit in self.fruits if fruit.active]
