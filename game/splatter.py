"""Juice spray when a fruit is sliced."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from src import config

_JUICE: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "yuzu": ((255, 236, 110), (255, 196, 28)),
    "orange": ((255, 176, 64), (255, 112, 16)),
    "watermelon": ((255, 64, 86), (196, 20, 44)),
}


@dataclass
class _Drop:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: tuple[int, int, int]
    life: float
    max_life: float
    stretch: float


class Splatter:
    """Droplets and streaks that burst along a cut."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self.drops: list[_Drop] = []

    def burst(
        self,
        x: float,
        y: float,
        cut_dx: float,
        cut_dy: float,
        kind: str,
        radius: float,
    ) -> None:
        length = math.hypot(cut_dx, cut_dy) or 1.0
        tx, ty = cut_dx / length, cut_dy / length
        nx, ny = -ty, tx
        bright, dark = _JUICE.get(kind, _JUICE["yuzu"])
        count = 28 if kind == "watermelon" else 24
        for i in range(count):
            along = self._rng.uniform(-1.0, 1.0)
            out = self._rng.choice((-1.0, 1.0)) * self._rng.uniform(0.25, 1.15)
            speed = self._rng.uniform(280.0, 780.0)
            vx = (tx * along * 0.9 + nx * out * 0.7) * speed
            vy = (ty * along * 0.9 + ny * out * 0.7) * speed - self._rng.uniform(60.0, 200.0)
            life = self._rng.uniform(0.32, 0.62)
            streak = i < 8
            self.drops.append(
                _Drop(
                    x=x + tx * along * radius * 0.45,
                    y=y + ty * along * radius * 0.45,
                    vx=vx,
                    vy=vy,
                    radius=self._rng.uniform(3.2, 8.5) * (1.2 if kind == "watermelon" else 1.0),
                    color=bright if i % 2 else dark,
                    life=life,
                    max_life=life,
                    stretch=self._rng.uniform(1.8, 3.2) if streak else 1.0,
                )
            )

    def update(self, dt: float, gravity: float = config.GRAVITY) -> None:
        dt = max(0.0, min(dt, 0.05))
        keep: list[_Drop] = []
        for drop in self.drops:
            drop.life -= dt
            if drop.life <= 0.0:
                continue
            drop.vy += gravity * dt * 0.62
            drop.x += drop.vx * dt
            drop.y += drop.vy * dt
            keep.append(drop)
        self.drops = keep

    def draw(self, surface: pygame.Surface) -> None:
        for drop in self.drops:
            t = max(drop.life / drop.max_life, 0.0)
            alpha = int(255 * (0.35 + 0.65 * t))
            r = max(int(drop.radius * (0.55 + 0.45 * t)), 2)
            if drop.stretch > 1.05:
                speed = math.hypot(drop.vx, drop.vy) or 1.0
                ux, uy = drop.vx / speed, drop.vy / speed
                length = max(int(r * drop.stretch * (0.7 + 0.3 * t)), r + 2)
                w, h = length * 2 + 4, r * 2 + 4
                blob = pygame.Surface((w, h), pygame.SRCALPHA)
                pygame.draw.ellipse(blob, (*drop.color, alpha), (0, 0, w, h))
                angle = -math.degrees(math.atan2(uy, ux))
                rotated = pygame.transform.rotate(blob, angle)
                surface.blit(rotated, rotated.get_rect(center=(int(drop.x), int(drop.y))))
            else:
                blob = pygame.Surface((r * 2 + 3, r * 2 + 3), pygame.SRCALPHA)
                pygame.draw.circle(blob, (*drop.color, alpha), (r + 1, r + 1), r)
                surface.blit(blob, (int(drop.x) - r - 1, int(drop.y) - r - 1))

    def clear(self) -> None:
        self.drops.clear()
