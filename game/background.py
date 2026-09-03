"""Fruit Ninja-style wooden dojo wall."""

from __future__ import annotations

import math

import numpy as np
import pygame


def _upsample(small: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear upsample a (sw, sh) float array to (width, height)."""
    sw, sh = small.shape
    xs = np.linspace(0, sw - 1, width, dtype=np.float32)
    ys = np.linspace(0, sh - 1, height, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, sw - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    tx = (xs - x0).astype(np.float32)[:, None]
    ty = (ys - y0).astype(np.float32)[None, :]
    a = small[x0[:, None], y0[None, :]]
    b = small[x1[:, None], y0[None, :]]
    c = small[x0[:, None], y1[None, :]]
    d = small[x1[:, None], y1[None, :]]
    return ((a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty)


def _noise(rng: np.random.Generator, sw: int, sh: int, width: int, height: int) -> np.ndarray:
    return _upsample(rng.standard_normal((max(sw, 2), max(sh, 2))).astype(np.float32), width, height)


def _streaks(phase: np.ndarray, sharpness: float) -> np.ndarray:
    """Thin bright/dark lines from a continuous phase field (0..1)."""
    frac = phase - np.floor(phase)
    return np.exp(-((frac - 0.5) * sharpness) ** 2)


def _plank_edges(width: int, n_planks: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.uniform(0.8, 1.25, n_planks).astype(np.float32)
    fracs = raw / raw.sum()
    return np.concatenate(([0.0], np.cumsum(fracs) * width))


def _slashes(surface: pygame.Surface, rng: np.random.Generator) -> None:
    """Deep sword cuts: a dark groove with a lit edge, some in V pairs."""
    width, height = surface.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)

    def cut(x0: float, y0: float, ang: float, length: float, depth: int) -> None:
        x1 = x0 + math.cos(ang) * length
        y1 = y0 + math.sin(ang) * length
        hx, hy = -math.sin(ang), math.cos(ang)
        # Groove tapers: thick dark core, thin dark edge, warm lit lip.
        pygame.draw.line(overlay, (5, 2, 1, depth), (x0, y0), (x1, y1), 3)
        pygame.draw.line(overlay, (3, 1, 0, depth), (x0 - hx, y0 - hy), (x1 - hx, y1 - hy), 1)
        pygame.draw.line(
            overlay,
            (168, 108, 58, max(depth // 3, 22)),
            (x0 + hx * 2.2, y0 + hy * 2.2),
            (x1 + hx * 2.2, y1 + hy * 2.2),
            1,
        )

    n = int(8 + width * height / 110000)
    for _ in range(n):
        x0 = float(rng.uniform(-40, width + 40))
        y0 = float(rng.uniform(-40, height + 40))
        ang = rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.1)
        length = float(rng.uniform(120, min(width, height) * 0.55))
        depth = int(rng.integers(120, 200))
        cut(x0, y0, ang, length, depth)
        if rng.random() < 0.45:
            cut(
                x0 + rng.uniform(-8, 8),
                y0 + rng.uniform(-8, 8),
                -ang + rng.uniform(-0.25, 0.25),
                length * rng.uniform(0.5, 0.9),
                depth,
            )

    for _ in range(5):
        cx = int(rng.integers(40, max(41, width - 40)))
        cy = int(rng.integers(40, max(41, height - 40)))
        for _s in range(10):
            rad = int(rng.integers(3, 14))
            pygame.draw.circle(
                overlay,
                (96, 14, 10, int(rng.integers(14, 34))),
                (cx + int(rng.integers(-30, 31)), cy + int(rng.integers(-30, 31))),
                rad,
            )
    surface.blit(overlay, (0, 0))


def build_dojo_wall(width: int, height: int, seed: int = 11) -> pygame.Surface:
    """Vertical boards with crisp grain, staggered seams, knots, and cuts."""
    width = max(int(width), 2)
    height = max(int(height), 2)
    rng = np.random.default_rng(seed)

    n_planks = max(int(round(width / 118)), 6)
    edges = _plank_edges(width, n_planks, rng)
    x = np.arange(width, dtype=np.float32)[:, None]
    y = np.arange(height, dtype=np.float32)[None, :]
    plank_i = np.clip(np.searchsorted(edges[1:], np.arange(width), side="right"), 0, n_planks - 1)[:, None]
    left = edges[plank_i]
    right = edges[plank_i + 1]
    span = np.maximum(right - left, 1.0)
    local = (x - left) / span
    dist_edge = np.minimum(x - left, right - x)

    plank_tone = rng.uniform(0.86, 1.14, n_planks).astype(np.float32)
    tone = plank_tone[plank_i]

    # Smooth low-frequency warp so grain lines wander like real fibers.
    warp = _noise(rng, n_planks * 2, max(height // 40, 6), width, height)
    wobble = _noise(rng, n_planks * 4, max(height // 14, 12), width, height)
    phase_a = x * 0.16 + warp * 2.2 + wobble * 0.5 + plank_i * 3.1
    phase_b = x * 0.31 + warp * 1.4 + wobble * 0.9 + plank_i * 1.7
    lines_a = _streaks(phase_a / (2 * math.pi), 5.5)
    lines_b = _streaks(phase_b / (2 * math.pi), 8.0)
    # Full-resolution speckle keeps the surface crisp instead of blurry.
    speckle = rng.standard_normal((width, height)).astype(np.float32)
    fine = _noise(rng, width // 3, height // 3, width, height)
    grain = -0.16 * lines_a - 0.10 * lines_b + 0.05 * fine + 0.022 * speckle + 0.06 * wobble

    # Board edges: dark gap plus a bevel highlight on the lit side.
    gap = np.clip(1.0 - dist_edge / 2.2, 0.0, 1.0)
    bevel = np.clip(1.0 - (x - left - 2.0) / 3.0, 0.0, 1.0) * (x - left >= 2.0)
    belly = 0.05 * np.sin(np.clip(local, 0.0, 1.0) * math.pi)

    # Staggered horizontal seams where boards butt end to end.
    seams = np.zeros((width, height), dtype=np.float32)
    for i in range(n_planks):
        col = (x >= edges[i]) & (x < edges[i + 1])
        for _ in range(int(rng.integers(1, 3))):
            sy = float(rng.uniform(0.08, 0.92) * height)
            band = np.clip(1.0 - np.abs(y - sy) / 2.0, 0.0, 1.0)
            lit = np.clip(1.0 - np.abs(y - sy - 3.0) / 1.5, 0.0, 1.0) * 0.5
            seams += col * (band - lit)

    knots = np.zeros((width, height), dtype=np.float32)
    for i in range(n_planks):
        if rng.random() < 0.3:
            continue
        px = float(0.5 * (edges[i] + edges[i + 1]) + rng.uniform(-0.2, 0.2) * float(edges[i + 1] - edges[i]))
        py = float(rng.uniform(0.1, 0.9) * height)
        rx = float(rng.uniform(6, 13))
        ry = float(rng.uniform(14, 34))
        dx = (x - px) / rx
        dy = (y - py) / ry
        rr = np.sqrt(dx * dx + dy * dy)
        core = np.clip(1.0 - rr, 0.0, 1.0)
        rings = _streaks(rr * 2.2, 6.0) * np.clip(1.0 - rr / 2.6, 0.0, 1.0)
        knots += 0.28 * core + 0.10 * rings

    nx = (x / width - 0.5) / 0.60
    ny = (y / height - 0.46) / 0.80
    spot = np.exp(-0.5 * (nx * nx + ny * ny))
    light = 0.40 + 0.42 * spot
    vignette = np.clip(
        1.05 - 0.85 * ((x / width - 0.5) ** 2 * 2.0 + (y / height - 0.5) ** 2 * 1.4),
        0.24,
        1.0,
    )

    shade = (tone + grain + belly + 0.08 * bevel - 0.55 * gap - 0.30 * seams - knots) * light * vignette
    shade = np.clip(shade, 0.0, 1.6)
    rgb = np.empty((width, height, 3), dtype=np.float32)
    rgb[..., 0] = 18.0 + 92.0 * shade
    rgb[..., 1] = 9.0 + 46.0 * shade
    rgb[..., 2] = 4.0 + 20.0 * shade
    np.clip(rgb, 0, 255, out=rgb)
    surface = pygame.surfarray.make_surface(rgb.astype(np.uint8))
    _slashes(surface, rng)
    if pygame.display.get_init() and pygame.display.get_surface() is not None:
        surface = surface.convert()
    return surface
