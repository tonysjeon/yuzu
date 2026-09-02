"""Shared data types."""

from dataclasses import dataclass


@dataclass
class FingertipPoint:
    """A single fingertip observation in game-screen coordinates."""

    x: float
    y: float
    timestamp: float
    confidence: float
