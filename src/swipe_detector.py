"""Detect intentional slicing swipes from fingertip motion."""

from __future__ import annotations

import math

from src import config
from src.models import FingertipPoint, SwipeResult


class SwipeDetector:
    """Activate the blade only when fingertip speed exceeds a threshold.

    Velocity is path length over the recent point window (not just start→end),
    so a curved swipe still registers. A short hold keeps the blade from
    flickering off between CV frames during a continuous slash.
    """

    def __init__(
        self,
        min_velocity: float = config.MIN_SWIPE_VELOCITY,
        hold_ms: float = config.SWIPE_HOLD_MS,
    ) -> None:
        self.min_velocity = min_velocity
        self._hold_s = hold_ms / 1000.0
        self._last_active_at = -1.0

    def update(self, points: list[FingertipPoint], now: float | None = None) -> SwipeResult:
        if len(points) < 2:
            return SwipeResult(active=False, velocity=0.0, start=None, end=None)

        first = points[0]
        last = points[-1]
        dt = last.timestamp - first.timestamp
        if dt <= 1e-6:
            return SwipeResult(
                active=False,
                velocity=0.0,
                start=(first.x, first.y),
                end=(last.x, last.y),
            )

        path = 0.0
        for a, b in zip(points, points[1:]):
            path += math.hypot(b.x - a.x, b.y - a.y)
        velocity = path / dt

        stamp = last.timestamp if now is None else now
        if velocity >= self.min_velocity:
            self._last_active_at = stamp
            active = True
        else:
            active = (
                self._last_active_at >= 0.0
                and (stamp - self._last_active_at) <= self._hold_s
            )

        return SwipeResult(
            active=active,
            velocity=velocity,
            start=(first.x, first.y),
            end=(last.x, last.y),
        )

    def reset(self) -> None:
        self._last_active_at = -1.0
