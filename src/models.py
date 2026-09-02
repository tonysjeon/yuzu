"""Shared data types."""

from dataclasses import dataclass


@dataclass
class FingertipPoint:
    """A single fingertip observation in game-screen coordinates."""

    x: float
    y: float
    timestamp: float
    confidence: float


@dataclass
class SwipeResult:
    """Whether recent fingertip motion counts as an intentional swipe."""

    active: bool
    velocity: float
    start: tuple[float, float] | None
    end: tuple[float, float] | None
