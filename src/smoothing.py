"""Fingertip smoothing filters."""

from __future__ import annotations

import math

from src import config


class ExponentialSmoother:
    """Fixed-alpha exponential smoothing applied to X and Y independently."""

    def __init__(self, alpha: float = config.SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._x is None or self._y is None:
            self._x, self._y = x, y
            return x, y
        self._x = self.alpha * x + (1.0 - self.alpha) * self._x
        self._y = self.alpha * y + (1.0 - self.alpha) * self._y
        return self._x, self._y

    def reset(self) -> None:
        self._x = None
        self._y = None


class _LowPass:
    def __init__(self) -> None:
        self.value: float | None = None

    def update(self, x: float, alpha: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Velocity-adaptive smoothing: steady when slow, responsive when fast.

    ``min_cutoff`` sets how much jitter is removed at rest; ``beta`` scales how
    quickly the filter opens up as the point speeds up. ``lateral_scale``
    (0..1) keeps the filter tighter *across* the direction of travel than
    along it, so a straight swipe stays straight instead of picking up the
    sideways wobble in the landmark.
    """

    def __init__(
        self,
        min_cutoff: float = config.ONE_EURO_MIN_CUTOFF,
        beta: float = config.ONE_EURO_BETA,
        d_cutoff: float = config.ONE_EURO_D_CUTOFF,
        lateral_scale: float = config.ONE_EURO_LATERAL_SCALE,
        lateral_gate_px: float = config.ONE_EURO_LATERAL_GATE_PX,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.lateral_scale = lateral_scale
        self.lateral_gate_px = max(lateral_gate_px, 1e-3)
        self._x: float | None = None
        self._y: float | None = None
        self._dx = _LowPass()
        self._dy = _LowPass()
        self._last_t: float | None = None
        self._last_raw: tuple[float, float] | None = None

    def update(self, x: float, y: float, t: float) -> tuple[float, float]:
        if self._last_t is None or self._last_raw is None or self._x is None or self._y is None:
            self._last_t = t
            self._last_raw = (x, y)
            self._x, self._y = x, y
            return x, y

        dt = t - self._last_t
        if dt <= 1e-6:
            return self._x, self._y
        self._last_t = t

        raw_dx = (x - self._last_raw[0]) / dt
        raw_dy = (y - self._last_raw[1]) / dt
        self._last_raw = (x, y)

        alpha_d = _alpha(self.d_cutoff, dt)
        dx = self._dx.update(raw_dx, alpha_d)
        dy = self._dy.update(raw_dy, alpha_d)
        speed = math.hypot(dx, dy)

        along_alpha = _alpha(self.min_cutoff + self.beta * speed, dt)
        ex = x - self._x
        ey = y - self._y

        if speed < 1e-6:
            self._x += along_alpha * ex
            self._y += along_alpha * ey
            return self._x, self._y

        # Split the innovation into along-track and cross-track components.
        ux, uy = dx / speed, dy / speed
        par = ex * ux + ey * uy
        px, py = par * ux, par * uy
        qx, qy = ex - px, ey - py

        lateral_alpha = _alpha(
            self.min_cutoff + self.beta * speed * self.lateral_scale, dt
        )
        # Small cross-track deviations are landmark wobble; large ones mean the
        # swipe is actually turning, so fall back to the responsive alpha.
        turn = min(math.hypot(qx, qy) / self.lateral_gate_px, 1.0)
        lateral_alpha += (along_alpha - lateral_alpha) * turn
        self._x += along_alpha * px + lateral_alpha * qx
        self._y += along_alpha * py + lateral_alpha * qy
        return self._x, self._y

    @property
    def speed(self) -> float:
        """Current smoothed speed estimate in units/s (0 before two samples)."""
        if self._dx.value is None or self._dy.value is None:
            return 0.0
        return math.hypot(self._dx.value, self._dy.value)

    def advance(self, x: float, y: float, t: float) -> None:
        """Move the filter to an externally predicted position.

        Used while coasting through dropped frames so that when real samples
        return, smoothing continues from where the cursor actually is instead
        of snapping back to the last detected point.
        """
        self._x, self._y = x, y
        self._last_raw = (x, y)
        self._last_t = t

    def reset(self) -> None:
        self._x = None
        self._y = None
        self._dx.reset()
        self._dy.reset()
        self._last_t = None
        self._last_raw = None


class AnchoredTip:
    """Stabilize a fingertip against a sturdier anchor such as the palm center.

    The tip is expressed as ``anchor + offset`` and only the offset is
    low-passed. The offset cutoff drops as speed rises: when the hand is
    moving fast its pose barely changes, so a fingertip landmark that suddenly
    jumps toward the palm (motion blur) is treated as noise, while the tip
    still translates with the palm at full speed.
    """

    def __init__(
        self,
        slow_cutoff: float = config.TIP_ANCHOR_CUTOFF_SLOW,
        fast_cutoff: float = config.TIP_ANCHOR_CUTOFF_FAST,
        full_speed: float = config.TIP_ANCHOR_SPEED_FULL,
    ) -> None:
        self.slow_cutoff = slow_cutoff
        self.fast_cutoff = fast_cutoff
        self.full_speed = max(full_speed, 1e-6)
        self._ox = _LowPass()
        self._oy = _LowPass()

    def update(
        self,
        anchor: tuple[float, float],
        tip: tuple[float, float],
        speed: float,
        dt: float,
    ) -> tuple[float, float]:
        ox = tip[0] - anchor[0]
        oy = tip[1] - anchor[1]
        if self._ox.value is None or dt <= 1e-6:
            self._ox.update(ox, 1.0)
            self._oy.update(oy, 1.0)
            return tip
        k = min(max(speed / self.full_speed, 0.0), 1.0)
        cutoff = self.slow_cutoff + (self.fast_cutoff - self.slow_cutoff) * k
        alpha = _alpha(cutoff, dt)
        fx = self._ox.update(ox, alpha)
        fy = self._oy.update(oy, alpha)
        return anchor[0] + fx, anchor[1] + fy

    def reset(self) -> None:
        self._ox.reset()
        self._oy.reset()
