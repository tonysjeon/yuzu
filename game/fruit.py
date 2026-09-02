"""Single fruit projectile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from src import config

# Visual size is the collision radius; sprites include stem/leaf overflow.
FRUIT_STYLES: dict[str, tuple[tuple[int, int, int], float]] = {
    "apple": ((220, 70, 70), 38.0),
    "orange": ((240, 150, 50), 42.0),
    "watermelon": ((70, 180, 90), 56.0),
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


def _sprite_at(kind: str, size: int) -> pygame.Surface | None:
    key = (kind, size)
    cached = _scaled.get(key)
    if cached is not None:
        return cached
    source = _load_sprite(kind)
    if source is None or size < 2:
        return None
    scaled = pygame.transform.smoothscale(source, (size, size))
    _scaled[key] = scaled
    return scaled


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
        size = max(int(self.radius * 2.4), 8)
        sprite = _sprite_at(self.fruit_type, size)
        center = (int(self.x), int(self.y))
        if sprite is None:
            pygame.draw.circle(surface, self.color, center, int(self.radius))
            return
        drawn = pygame.transform.rotate(sprite, self.rotation)
        surface.blit(drawn, drawn.get_rect(center=center))
