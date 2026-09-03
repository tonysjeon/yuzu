"""Single fruit projectile and sliced halves."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from src import config

# Visual size is the collision radius; sprites include stem/leaf overflow.
FRUIT_STYLES: dict[str, tuple[tuple[int, int, int], float]] = {
    "yuzu": ((245, 200, 40), 52.0),
    "orange": ((240, 150, 50), 55.0),
    "watermelon": ((70, 180, 90), 73.0),
}

# Interior of a cut fruit: flesh, shaded flesh, rind, and accent.
_FLESH: dict[str, dict[str, tuple[int, int, int]]] = {
    "yuzu": {
        "flesh": (255, 236, 150),
        "shade": (232, 200, 80),
        "rind": (242, 196, 28),
        "accent": (252, 246, 220),
    },
    "orange": {
        "flesh": (255, 168, 48),
        "shade": (220, 108, 24),
        "rind": (235, 96, 22),
        "accent": (255, 210, 90),
    },
    "watermelon": {
        "flesh": (226, 48, 72),
        "shade": (148, 18, 42),
        "rind": (46, 132, 64),
        "accent": (32, 24, 20),
    },
}

_SPRITE_DIR = Path(__file__).resolve().parent.parent / "assets" / "fruits"
_sprites: dict[str, pygame.Surface | None] = {}
_scaled: dict[tuple[str, int], pygame.Surface] = {}


def _load_sprite(kind: str) -> pygame.Surface | None:
    if kind in _sprites:
        return _sprites[kind]
    path = _SPRITE_DIR / f"{kind}.png"
    if not path.is_file():
        _sprites[kind] = None
        return None
    image = pygame.image.load(str(path))
    if pygame.display.get_init() and pygame.display.get_surface() is not None:
        image = image.convert_alpha()
    _sprites[kind] = image
    return image


def fruit_sprite(kind: str, size: int) -> pygame.Surface | None:
    """Scaled fruit art for HUD icons and projectiles."""
    return _sprite_at(kind, size)


# Sprites are rotated at 2x and filtered back down so edges stay smooth.
_SUPERSAMPLE = 2
# Half-cap raster size cap (supersampled). Big enough to stay sharp, small
# enough that a slice doesn't hitch.
_HALF_MAX_PX = 256
_HALF_ANGLE_STEPS = 24
_FLESH_TEX_SIZE = 256
_FLESH_TEX_EXTENT = 1.15

_flesh_tex: dict[str, np.ndarray] = {}
_flesh_alpha: np.ndarray | None = None
_skin_arr_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_half_cache: dict[tuple[str, int, int], pygame.Surface] = {}


def _draw_rotated(
    surface: pygame.Surface,
    hires: pygame.Surface,
    center: tuple[int, int],
    rotation: float,
    scale: float | None = None,
) -> None:
    zoom = (1.0 / _SUPERSAMPLE) if scale is None else scale
    drawn = pygame.transform.rotozoom(hires, rotation, zoom)
    surface.blit(drawn, drawn.get_rect(center=center))


_icons: dict[tuple[str, int], pygame.Surface] = {}


def _crop_opaque(surf: pygame.Surface, pad: int = 2) -> pygame.Surface:
    """Tight box around visible pixels so a cut half fills the HUD badge."""
    alpha = pygame.surfarray.array_alpha(surf)
    xs, ys = np.where(alpha > 10)
    if xs.size == 0:
        return surf
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, surf.get_width())
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, surf.get_height())
    return surf.subsurface((x0, y0, x1 - x0, y1 - y0)).copy()


def cut_fruit_icon(kind: str, size: int) -> pygame.Surface | None:
    """HUD badge: a turned cut half, same hemisphere as a sliced fruit."""
    key = (kind, size)
    cached = _icons.get(key)
    if cached is not None:
        return cached
    raster = max(size * 3, 160)
    skin = _sprite_at(kind, raster)
    if skin is None:
        return None
    # Same cap as in-game halves. Tip it, then turn the cut face toward the
    # top-left so it reads against the score in the HUD corner.
    half = _render_half_cap(kind, skin, -0.82, -0.28)
    turned = pygame.transform.rotozoom(_crop_opaque(half), -32.0, 1.0)
    turned = pygame.transform.rotate(turned, 180)
    turned = _crop_opaque(turned, pad=1)
    tw, th = turned.get_size()
    scale = size / max(tw, th, 1)
    icon = pygame.transform.smoothscale(
        turned, (max(int(round(tw * scale)), 1), max(int(round(th * scale)), 1))
    )
    try:
        icon = icon.convert_alpha()
    except pygame.error:
        pass
    _icons[key] = icon
    return icon


def _sprite_at(kind: str, size: int) -> pygame.Surface | None:
    key = (kind, size)
    cached = _scaled.get(key)
    if cached is not None:
        return cached
    source = _load_sprite(kind)
    if source is None or size < 2:
        return None
    scaled = pygame.transform.smoothscale(source, (size, size))
    try:
        scaled = scaled.convert_alpha()
    except pygame.error:
        pass
    _scaled[key] = scaled
    return scaled


def _lerp(
    a: tuple[int, int, int] | np.ndarray,
    b: tuple[int, int, int] | np.ndarray,
    t: float | np.ndarray,
) -> np.ndarray:
    tt = np.asarray(t, dtype=np.float32)
    if tt.ndim == 1:
        tt = tt[..., None]
    elif tt.ndim == 2:
        tt = tt[..., None]
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return aa * (1.0 - tt) + bb * tt


def _skin_arrays(source: pygame.Surface) -> tuple[np.ndarray, np.ndarray]:
    key = id(source)
    cached = _skin_arr_cache.get(key)
    if cached is not None:
        return cached
    if not pygame.get_init():
        pygame.init()
    try:
        src = source.convert_alpha()
    except pygame.error:
        src = source
    arrays = (
        pygame.surfarray.array3d(src).astype(np.float32),
        pygame.surfarray.array_alpha(src).astype(np.float32),
    )
    _skin_arr_cache[key] = arrays
    return arrays


def _sample_sprite(
    rgb: np.ndarray,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample a sprite at sphere-front UVs in [-1, 1]."""
    h, w = rgb.shape[1], rgb.shape[0]
    x = (u * 0.5 + 0.5) * (w - 1)
    y = (v * 0.5 + 0.5) * (h - 1)
    x0 = np.clip(np.floor(x).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    tx = (x - x0)[..., None]
    ty = (y - y0)[..., None]
    color = (rgb[x0, y0] * (1 - tx) + rgb[x1, y0] * tx) * (1 - ty) + (
        rgb[x0, y1] * (1 - tx) + rgb[x1, y1] * tx
    ) * ty
    txa = x - x0
    tya = y - y0
    a = (alpha[x0, y0] * (1 - txa) + alpha[x1, y0] * txa) * (1 - tya) + (
        alpha[x0, y1] * (1 - txa) + alpha[x1, y1] * txa
    ) * tya
    return color, a


def _yuzu_flesh(fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
    """Yuzu cross-section: thick pith, pale segments, many large white seeds."""
    r2 = fu * fu + fv * fv
    radial = np.sqrt(np.clip(r2, 0.0, 1.0))
    ang = np.arctan2(fv, fu)
    n_seg = 11
    turns = (ang + math.pi) / (2.0 * math.pi) * n_seg
    mem = np.abs(turns - np.round(turns))
    membrane = np.clip(1.0 - mem / 0.055, 0.0, 1.0) * np.clip(
        (radial - 0.12) / 0.10, 0.0, 1.0
    )

    juice = np.array((255, 236, 168), dtype=np.float32)
    juice_dark = np.array((236, 210, 110), dtype=np.float32)
    pith = np.array((252, 248, 236), dtype=np.float32)
    seed = np.array((250, 244, 220), dtype=np.float32)
    seed_edge = np.array((214, 196, 150), dtype=np.float32)

    vr = np.mod(radial * 14.0 + 0.12 * np.sin(ang * n_seg), 1.0)
    va = np.mod(ang * 6.5 / math.pi + radial * 3.0, 1.0)
    cell = ((vr - 0.5) / 0.40) ** 2 + ((va - 0.5) / 0.44) ** 2
    vesicle = np.clip(1.0 - cell, 0.0, 1.0)
    rgb = _lerp(juice_dark, juice, 0.40 + 0.60 * vesicle)
    rgb = _lerp(rgb, pith, 0.80 * membrane)
    col = np.clip(1.0 - (radial / 0.14) ** 2, 0.0, 1.0)
    rgb = _lerp(rgb, pith, col)

    # Large teardrop seeds clustered toward the center of each segment.
    for i in range(n_seg):
        a = -math.pi + (i + 0.5) * (2.0 * math.pi / n_seg)
        sx, sy = 0.28 * math.cos(a), 0.28 * math.sin(a)
        ca, sa = math.cos(a), math.sin(a)
        lu = (fu - sx) * ca + (fv - sy) * sa
        lv = -(fu - sx) * sa + (fv - sy) * ca
        blob = (lu / 0.085) ** 2 + (lv / 0.048) ** 2
        inside = blob < 1.0
        rgb[inside] = _lerp(seed_edge, seed, 1.0 - blob[inside] ** 0.6)
        highlight = (lu / 0.04) ** 2 + ((lv + 0.012) / 0.022) ** 2
        rgb[highlight < 1.0] = (255, 252, 240)

    pith_ring = np.clip((radial - 0.70) / 0.12, 0.0, 1.0) * np.clip(
        (0.90 - radial) / 0.08, 0.0, 1.0
    )
    rgb = _lerp(rgb, pith, pith_ring)
    zest = np.clip((radial - 0.88) / 0.12, 0.0, 1.0)
    zest_noise = 0.5 + 0.5 * np.sin(ang * 22.0) * np.sin(radial * 36.0)
    rgb = _lerp(rgb, (242, 196, 28), zest)
    rgb = rgb + (zest * (zest_noise - 0.5) * 18.0)[..., None]
    return np.clip(rgb, 0, 255)


def _orange_flesh(fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
    """Cross-section: zest, white pith, radial segments, juice vesicles."""
    r2 = fu * fu + fv * fv
    radial = np.sqrt(np.clip(r2, 0.0, 1.0))
    ang = np.arctan2(fv, fu)
    n_seg = 10
    # Distance to nearest segment membrane.
    turns = (ang + math.pi) / (2.0 * math.pi) * n_seg
    mem = np.abs(turns - np.round(turns))
    membrane = np.clip(1.0 - mem / 0.06, 0.0, 1.0) * np.clip((radial - 0.10) / 0.12, 0.0, 1.0)

    juice = np.array((255, 148, 32), dtype=np.float32)
    juice_dark = np.array((230, 110, 18), dtype=np.float32)
    # Vesicles: polar cells that glisten.
    vr = np.mod(radial * 16.0 + 0.15 * np.sin(ang * n_seg), 1.0)
    va = np.mod(ang * 7.5 / math.pi + radial * 4.0, 1.0)
    cell = ((vr - 0.5) / 0.38) ** 2 + ((va - 0.5) / 0.42) ** 2
    vesicle = np.clip(1.0 - cell, 0.0, 1.0)
    highlight = vesicle * np.clip(0.55 - vr, 0.0, 1.0)
    rgb = _lerp(juice_dark, juice, 0.45 + 0.55 * vesicle)
    rgb = rgb + highlight[..., None] * 40.0

    pith = np.array((248, 244, 232), dtype=np.float32)
    rgb = _lerp(rgb, pith, 0.85 * membrane)
    # Central columella.
    col = np.clip(1.0 - (radial / 0.12) ** 2, 0.0, 1.0)
    rgb = _lerp(rgb, pith, col)
    # White pith ring under the zest.
    pith_ring = np.clip((radial - 0.78) / 0.10, 0.0, 1.0) * np.clip((0.93 - radial) / 0.06, 0.0, 1.0)
    rgb = _lerp(rgb, pith, pith_ring)
    # Thin textured zest at the very edge.
    zest = np.clip((radial - 0.91) / 0.09, 0.0, 1.0)
    zest_noise = 0.5 + 0.5 * np.sin(ang * 28.0) * np.sin(radial * 40.0)
    rgb = _lerp(rgb, (255, 118, 22), zest)
    rgb = rgb + (zest * (zest_noise - 0.5) * 24.0)[..., None]
    return np.clip(rgb, 0, 255)


def _watermelon_flesh(fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
    """Cross-section: juicy red flesh, irregular seeds, white pith, green rind."""
    r2 = fu * fu + fv * fv
    radial = np.sqrt(np.clip(r2, 0.0, 1.0))
    ang = np.arctan2(fv, fu)

    heart = np.array((176, 32, 52), dtype=np.float32)
    flesh = np.array((220, 52, 72), dtype=np.float32)
    juice = np.array((242, 98, 108), dtype=np.float32)
    pith = np.array((250, 246, 232), dtype=np.float32)
    rind_pale = np.array((198, 214, 150), dtype=np.float32)
    rind = np.array((58, 138, 70), dtype=np.float32)
    rind_dark = np.array((30, 86, 44), dtype=np.float32)
    seed = np.array((26, 18, 16), dtype=np.float32)
    seed_edge = np.array((64, 40, 32), dtype=np.float32)
    seed_hl = np.array((118, 88, 70), dtype=np.float32)

    # Soft cartesian blotches — watermelon isn't segmented like citrus.
    n1 = np.sin(fu * 7.3 + 1.1) * np.sin(fv * 6.8 - 0.4)
    n2 = np.sin(fu * 13.7 - fv * 3.2) * np.sin(fv * 12.1 + fu * 2.8)
    n3 = np.sin(fu * 19.0 + fv * 17.5) * np.sin(fv * 21.0 - fu * 15.0)
    mottle = np.clip(0.5 + 0.28 * n1 + 0.22 * n2 + 0.12 * n3, 0.0, 1.0)
    rgb = _lerp(heart, flesh, np.clip(0.30 + radial * 0.70, 0.0, 1.0))
    rgb = _lerp(rgb, juice, 0.55 * mottle)
    rgb = rgb + ((mottle - 0.5) * 16.0)[..., None]

    # Teardrop seeds pointing inward, scattered in a messy band.
    n_seeds = 15
    for i in range(n_seeds):
        if i % 5 == 3:
            continue
        a = -math.pi + (i + 0.28 * math.sin(i * 2.15)) * (2.0 * math.pi / n_seeds)
        rr = 0.36 + 0.12 * math.sin(i * 1.7 + 0.8)
        if i % 4 == 1:
            rr += 0.09
        sx, sy = rr * math.cos(a), rr * math.sin(a)
        ca, sa = math.cos(a), math.sin(a)
        lu = (fu - sx) * ca + (fv - sy) * sa
        lv = -(fu - sx) * sa + (fv - sy) * ca
        length = 0.068 + 0.016 * math.sin(i * 2.9)
        width = 0.028 + 0.008 * math.cos(i * 2.1)
        blob = (lu / length) ** 2 + (lv / width) ** 2
        inside = blob < 1.0
        rgb[inside] = _lerp(seed_edge, seed, np.clip(blob[inside] ** 0.45, 0.0, 1.0))
        highlight = ((lu + 0.016) / (length * 0.36)) ** 2 + (
            (lv - 0.005) / (width * 0.40)
        ) ** 2
        rgb[inside & (highlight < 1.0)] = seed_hl

    white = np.clip((radial - 0.78) / 0.05, 0.0, 1.0) * np.clip(
        (0.89 - radial) / 0.04, 0.0, 1.0
    )
    rgb = _lerp(rgb, pith, white)
    pale = np.clip((radial - 0.87) / 0.04, 0.0, 1.0) * np.clip(
        (0.93 - radial) / 0.03, 0.0, 1.0
    )
    rgb = _lerp(rgb, rind_pale, pale)
    green = np.clip((radial - 0.91) / 0.09, 0.0, 1.0)
    stripe = 0.5 + 0.5 * np.sin(ang * 11.0 + 0.6 * np.sin(ang * 3.0))
    rgb = _lerp(rgb, rind, green)
    rgb = _lerp(rgb, rind_dark, green * (0.25 + 0.55 * stripe))
    return np.clip(rgb, 0, 255)


def _flesh_albedo(kind: str, fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
    """Interior texture in the cut-plane disc, coords in roughly [-1, 1]."""
    if kind == "yuzu":
        return _yuzu_flesh(fu, fv)
    if kind == "orange":
        return _orange_flesh(fu, fv)
    if kind == "watermelon":
        return _watermelon_flesh(fu, fv)

    palette = _FLESH.get(kind, _FLESH["yuzu"])
    r2 = fu * fu + fv * fv
    radial = np.sqrt(np.clip(r2, 0.0, 1.0))
    light = np.clip(0.55 + 0.45 * (1.0 - radial), 0.0, 1.0)
    rgb = _lerp(palette["shade"], palette["flesh"], light)
    rgb = _lerp(rgb, palette["flesh"], np.clip(1.0 - radial * 0.35, 0, 1))
    seeds = (
        (-0.42, -0.08),
        (0.38, 0.18),
        (0.08, -0.40),
        (-0.18, 0.38),
        (0.40, -0.22),
        (-0.32, 0.22),
    )
    for sx, sy in seeds:
        blob = ((fu - sx) / 0.08) ** 2 + ((fv - sy) / 0.045) ** 2
        rgb[blob < 1.0] = palette["accent"]
    rind = np.clip((radial - 0.82) / 0.18, 0.0, 1.0)
    rgb = _lerp(rgb, palette["rind"], rind)
    return np.clip(rgb, 0, 255)


def _flesh_texture(kind: str) -> np.ndarray:
    cached = _flesh_tex.get(kind)
    if cached is not None:
        return cached
    n = _FLESH_TEX_SIZE
    extent = _FLESH_TEX_EXTENT
    u = np.linspace(-extent, extent, n, dtype=np.float32)[:, None]
    v = np.linspace(-extent, extent, n, dtype=np.float32)[None, :]
    rgb = _flesh_albedo(kind, u, v).astype(np.float32)
    _flesh_tex[kind] = rgb
    return rgb


def _sample_flesh(kind: str, fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
    global _flesh_alpha
    tex = _flesh_texture(kind)
    n = tex.shape[0]
    if _flesh_alpha is None or _flesh_alpha.shape[0] != n:
        _flesh_alpha = np.full((n, n), 255.0, dtype=np.float32)
    color, _ = _sample_sprite(
        tex, _flesh_alpha, fu / _FLESH_TEX_EXTENT, fv / _FLESH_TEX_EXTENT
    )
    return color


def warm_cut_cache() -> None:
    """Build flesh textures once so the first slice doesn't hitch."""
    for kind in FRUIT_STYLES:
        _flesh_texture(kind)


def _render_half_cap(
    kind: str,
    skin: pygame.Surface,
    nx: float,
    ny: float,
) -> pygame.Surface:
    """Tilted hemisphere: circular flesh face + curved skin shell."""
    nlen = math.hypot(nx, ny) or 1.0
    nx, ny = nx / nlen, ny / nlen
    ang = math.atan2(ny, nx)
    bucket = int(round(ang / (2.0 * math.pi) * _HALF_ANGLE_STEPS)) % _HALF_ANGLE_STEPS
    size = min(max(skin.get_width(), skin.get_height(), 32), _HALF_MAX_PX)
    cache_key = (kind, size, bucket)
    cached = _half_cache.get(cache_key)
    if cached is not None:
        return cached
    step = 2.0 * math.pi / _HALF_ANGLE_STEPS
    nx, ny = math.cos(bucket * step), math.sin(bucket * step)

    # Face looks at the camera and slightly toward the gap between halves.
    tilt = math.radians(50.0)
    ct, st = math.cos(tilt), math.sin(tilt)
    fnx, fny, fnz = -nx * ct, -ny * ct, st
    flen = math.sqrt(fnx * fnx + fny * fny + fnz * fnz)
    fnx, fny, fnz = fnx / flen, fny / flen, fnz / flen
    tx, ty, tz = -ny, nx, 0.0
    bx = fny * tz - fnz * ty
    by = fnz * tx - fnx * tz
    bz = fnx * ty - fny * tx
    blen = math.sqrt(bx * bx + by * by + bz * bz) or 1.0
    bx, by, bz = bx / blen, by / blen, bz / blen

    extent = 1.05
    u = np.linspace(-extent, extent, size, dtype=np.float32)[:, None]
    v = np.linspace(-extent, extent, size, dtype=np.float32)[None, :]
    r2 = u * u + v * v
    disc = r2 <= 1.0
    z_sph = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    # Volume sits behind the cut so the camera sees flesh, not the front pole.
    vol = fnx * u + fny * v + fnz * z_sph <= 0.06
    skin_hit = disc & vol

    z_cut = -(fnx * u + fny * v) / max(fnz, 1e-4)
    cut_r2 = u * u + v * v + z_cut * z_cut
    flesh_hit = cut_r2 <= 1.02

    z_skin = np.where(skin_hit, z_sph, -10.0)
    z_flesh = np.where(flesh_hit, z_cut, -10.0)
    use_flesh = flesh_hit & (z_flesh >= z_skin - 0.02)
    use_skin = skin_hit & ~use_flesh

    sprite_rgb, sprite_a = _skin_arrays(skin)
    sampled, samp_a = _sample_sprite(sprite_rgb, sprite_a, u, v)
    ndotl = np.clip(-0.35 * u - 0.45 * v + 0.82 * z_sph, 0.22, 1.0)
    skin_rgb = sampled * (0.42 + 0.58 * ndotl[..., None])

    fu = u * tx + v * ty + z_cut * tz
    fv = u * bx + v * by + z_cut * bz
    flesh_rgb = _sample_flesh(kind, fu, fv)

    out = np.zeros((size, size, 3), dtype=np.float32)
    out[use_skin] = skin_rgb[use_skin]
    out[use_flesh] = flesh_rgb[use_flesh]

    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[use_skin] = np.clip(samp_a[use_skin], 180, 255)
    alpha[use_flesh] = 255
    visible = use_flesh | use_skin
    edge = visible & (r2 > 0.93)
    alpha[edge] *= np.clip((1.0 - r2[edge]) / 0.07, 0.0, 1.0)

    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, np.clip(out, 0, 255).astype(np.uint8))
    pygame.surfarray.pixels_alpha(surf)[:, :] = np.clip(alpha, 0, 255).astype(np.uint8)
    try:
        surf = surf.convert_alpha()
    except pygame.error:
        pass
    _half_cache[cache_key] = surf
    return surf


@dataclass
class FruitHalf:
    """One side of a sliced fruit, still under gravity."""

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    radius: float
    sprite: pygame.Surface | None
    rotation: float
    rotation_speed: float
    display_size: int
    active: bool = True

    def update(self, dt: float, gravity: float = config.GRAVITY) -> None:
        if not self.active:
            return
        self.velocity_y += gravity * dt
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        self.rotation += self.rotation_speed * dt

    def draw(self, surface: pygame.Surface) -> None:
        if not self.active:
            return
        center = (int(self.x), int(self.y))
        if self.sprite is None:
            pygame.draw.circle(surface, (200, 80, 80), center, int(self.radius * 0.7))
            return
        scale = self.display_size / max(self.sprite.get_width(), 1)
        _draw_rotated(surface, self.sprite, center, self.rotation, scale=scale)


@dataclass
class Fruit:
    """A fruit under gravity, drawn from a sprite when available."""

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    radius: float
    sliced: bool
    active: bool
    fruit_type: str
    color: tuple[int, int, int]
    rotation: float = 0.0
    rotation_speed: float = 0.0

    @property
    def is_bomb(self) -> bool:
        return self.fruit_type == "bomb"

    def sprite_size(self) -> int:
        return max(int(self.radius * 2.4), 8)

    def update(self, dt: float, gravity: float = config.GRAVITY) -> None:
        if not self.active:
            return
        self.velocity_y += gravity * dt
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        self.rotation += self.rotation_speed * dt

    def draw(self, surface: pygame.Surface) -> None:
        if not self.active:
            return
        hires = _sprite_at(self.fruit_type, self.sprite_size() * _SUPERSAMPLE)
        center = (int(self.x), int(self.y))
        if hires is None:
            pygame.draw.circle(surface, self.color, center, int(self.radius))
            return
        _draw_rotated(surface, hires, center, self.rotation)

    def split(self, cut_dx: float, cut_dy: float) -> tuple[FruitHalf, FruitHalf]:
        """Cut along the blade and return two halves flying apart."""
        length = math.hypot(cut_dx, cut_dy)
        if length <= 1e-6:
            cut_dx, cut_dy = 1.0, 0.0
            length = 1.0
        tx, ty = cut_dx / length, cut_dy / length
        nx, ny = -ty, tx

        # Halves are rasterized (capped) then drawn at the same size as the whole fruit.
        size = self.sprite_size()
        sprite = _sprite_at(self.fruit_type, size * _SUPERSAMPLE)
        left_sprite: pygame.Surface | None = None
        right_sprite: pygame.Surface | None = None
        if sprite is not None:
            left_sprite = _render_half_cap(self.fruit_type, sprite, nx, ny)
            right_sprite = _render_half_cap(self.fruit_type, sprite, -nx, -ny)

        impulse = config.SLICE_SEPARATION_SPEED
        spin = config.SLICE_SPIN
        offset = max(self.radius * 0.22, 8.0)
        left = FruitHalf(
            x=self.x + nx * offset,
            y=self.y + ny * offset,
            velocity_x=self.velocity_x + nx * impulse,
            velocity_y=self.velocity_y + ny * impulse,
            radius=self.radius,
            sprite=left_sprite,
            rotation=0.0,
            rotation_speed=self.rotation_speed + spin,
            display_size=size,
        )
        right = FruitHalf(
            x=self.x - nx * offset,
            y=self.y - ny * offset,
            velocity_x=self.velocity_x - nx * impulse,
            velocity_y=self.velocity_y - ny * impulse,
            radius=self.radius,
            sprite=right_sprite,
            rotation=0.0,
            rotation_speed=self.rotation_speed - spin,
            display_size=size,
        )
        return left, right
