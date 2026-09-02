"""Blade trajectory from recent fingertip motion."""

from __future__ import annotations

import math

from src import config
from src.models import FingertipPoint


class Blade:
    """Keep a short trailing path of fingertip points."""

    def __init__(
        self,
        history_ms: float = config.BLADE_HISTORY_MS,
        min_step_px: float = config.BLADE_MIN_STEP_PX,
    ) -> None:
        self._history_s = history_ms / 1000.0
        self._min_step = min_step_px
        self._points: list[FingertipPoint] = []

    def add(
        self,
        x: float,
        y: float,
        timestamp: float,
        confidence: float = 1.0,
    ) -> None:
        if self._points:
            last = self._points[-1]
            if math.hypot(x - last.x, y - last.y) < self._min_step:
                # Same CV frame re-sampled by a faster game loop; just age out.
                self.expire(timestamp)
                return
        self._points.append(
            FingertipPoint(
                x=x,
                y=y,
                timestamp=timestamp,
                confidence=confidence,
            )
        )
        self.expire(timestamp)

    def expire(self, now: float) -> None:
        """Drop points older than the history window."""
        cutoff = now - self._history_s
        self._points = [point for point in self._points if point.timestamp >= cutoff]

    def clear(self) -> None:
        self._points.clear()

    @property
    def points(self) -> list[FingertipPoint]:
        return list(self._points)

    def polyline(self) -> list[tuple[float, float]]:
        """Return recent points as (x, y) pairs for rendering."""
        return [(point.x, point.y) for point in self._points]

    def curve(self, segments: int = config.BLADE_CURVE_SEGMENTS) -> list[tuple[float, float]]:
        """Return a smooth Catmull-Rom curve through the recent points.

        At 30 fps a 150 ms trail is only a handful of samples; drawing them
        as straight segments makes even a clean swipe look angular.
        """
        return catmull_rom(self.polyline(), segments)


def catmull_rom(
    points: list[tuple[float, float]],
    segments: int,
) -> list[tuple[float, float]]:
    """Interpolate a centripetal-free (uniform) Catmull-Rom spline."""
    if len(points) < 3 or segments < 2:
        return list(points)

    # Duplicate the endpoints so the curve passes through them.
    pts = [points[0], *points, points[-1]]
    out: list[tuple[float, float]] = [points[0]]
    for i in range(1, len(pts) - 2):
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for step in range(1, segments + 1):
            t = step / segments
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                2 * x1
                + (-x0 + x2) * t
                + (2 * x0 - 5 * x1 + 4 * x2 - x3) * t2
                + (-x0 + 3 * x1 - 3 * x2 + x3) * t3
            )
            y = 0.5 * (
                2 * y1
                + (-y0 + y2) * t
                + (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2
                + (-y0 + 3 * y1 - 3 * y2 + y3) * t3
            )
            out.append((x, y))
    return out
