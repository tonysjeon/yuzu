"""Map webcam coordinates into game-screen coordinates."""

from __future__ import annotations

from src import config


class CoordinateMapper:
    """Scale camera pixels through an active region onto the game surface."""

    def __init__(
        self,
        camera_width: int,
        camera_height: int,
        game_width: int = config.GAME_WIDTH,
        game_height: int = config.GAME_HEIGHT,
        active_region: dict[str, float] | None = None,
    ) -> None:
        region = active_region if active_region is not None else config.ACTIVE_REGION
        self.camera_width = max(int(camera_width), 1)
        self.camera_height = max(int(camera_height), 1)
        self.game_width = max(int(game_width), 1)
        self.game_height = max(int(game_height), 1)
        self.active_region = {
            "left": float(region["left"]),
            "right": float(region["right"]),
            "top": float(region["top"]),
            "bottom": float(region["bottom"]),
        }
        self._validate_region()

    def _validate_region(self) -> None:
        left = self.active_region["left"]
        right = self.active_region["right"]
        top = self.active_region["top"]
        bottom = self.active_region["bottom"]
        if not (0.0 <= left < right <= 1.0):
            raise ValueError("ACTIVE_REGION left/right must satisfy 0 <= left < right <= 1")
        if not (0.0 <= top < bottom <= 1.0):
            raise ValueError("ACTIVE_REGION top/bottom must satisfy 0 <= top < bottom <= 1")

    def set_game_size(self, width: int, height: int) -> None:
        self.game_width = max(int(width), 1)
        self.game_height = max(int(height), 1)

    def set_camera_size(self, width: int, height: int) -> None:
        self.camera_width = max(int(width), 1)
        self.camera_height = max(int(height), 1)

    @property
    def active_pixel_bounds(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom) in camera pixels."""
        left = self.active_region["left"] * self.camera_width
        right = self.active_region["right"] * self.camera_width
        top = self.active_region["top"] * self.camera_height
        bottom = self.active_region["bottom"] * self.camera_height
        return left, top, right, bottom

    def map(self, camera_x: float, camera_y: float) -> tuple[float, float]:
        """Map a camera-space point into clamped game-screen coordinates."""
        left, top, right, bottom = self.active_pixel_bounds
        region_w = max(right - left, 1e-6)
        region_h = max(bottom - top, 1e-6)

        nx = (camera_x - left) / region_w
        ny = (camera_y - top) / region_h
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)

        screen_x = nx * (self.game_width - 1)
        screen_y = ny * (self.game_height - 1)
        return screen_x, screen_y
