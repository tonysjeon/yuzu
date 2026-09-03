"""Blade-segment versus fruit-circle collision."""

from __future__ import annotations

import math

from src import config


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Shortest distance from point P to segment AB."""
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / length_sq
    t = min(max(t, 0.0), 1.0)
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def segment_hits_circle(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    """True when segment AB comes within ``radius`` of circle center C."""
    return point_segment_distance(cx, cy, ax, ay, bx, by) <= radius


def blade_cut_axis(
    points: list[tuple[float, float]],
    fruit_x: float,
    fruit_y: float,
    fruit_radius: float,
    blade_radius: float = config.BLADE_HIT_RADIUS,
) -> tuple[float, float] | None:
    """Direction of the first blade segment that grazes the fruit, or None."""
    if len(points) < 2:
        return None
    hit_radius = fruit_radius + blade_radius
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        if segment_hits_circle(ax, ay, bx, by, fruit_x, fruit_y, hit_radius):
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                return (1.0, 0.0)
            return (dx / length, dy / length)
    return None


def blade_hits_fruit(
    points: list[tuple[float, float]],
    fruit_x: float,
    fruit_y: float,
    fruit_radius: float,
    blade_radius: float = config.BLADE_HIT_RADIUS,
) -> bool:
    """True if any consecutive blade samples graze the fruit circle.

    Uses the segment, not the fingertip-in-circle test, so a fast swipe that
    never samples inside the fruit still counts.
    """
    return blade_cut_axis(points, fruit_x, fruit_y, fruit_radius, blade_radius) is not None
