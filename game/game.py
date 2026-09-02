"""Pygame game window and loop helpers."""

from __future__ import annotations

import pygame

from game.fruit_manager import FruitManager
from src import config


class Game:
    """Play surface with fruits, fingertip marker, and blade trail."""

    def __init__(
        self,
        width: int = config.GAME_WIDTH,
        height: int = config.GAME_HEIGHT,
        target_fps: int = config.TARGET_FPS,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("yuzu")
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps
        self.running = True
        self.fingertip: tuple[float, float] | None = None
        self.pointer_finger: str | None = None
        self.blade_points: list[tuple[float, float]] = []
        self.blade_active = False
        self.swipe_velocity = 0.0
        self.fruits = FruitManager()
        self._font = pygame.font.SysFont("helvetica", 22)
        self._set_mode(width, height)

    def _set_mode(self, width: int, height: int) -> None:
        width = max(width, 320)
        height = max(height, 240)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(
            (width, height),
            pygame.RESIZABLE,
        )

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self._set_mode(event.w, event.h)
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_q,
                pygame.K_ESCAPE,
            ):
                self.running = False

    def set_fingertip(
        self,
        point: tuple[float, float] | None,
        pointer_finger: str | None = None,
    ) -> None:
        self.fingertip = point
        self.pointer_finger = pointer_finger if point is not None else None

    def set_blade_points(
        self,
        points: list[tuple[float, float]],
        active: bool = False,
        velocity: float = 0.0,
    ) -> None:
        self.blade_points = points
        self.blade_active = active
        self.swipe_velocity = velocity

    def update(self) -> None:
        # get_time is ms since the previous tick(); first frame may be 0.
        dt_ms = self.clock.get_time()
        dt = (1.0 / self.target_fps) if dt_ms <= 0 else dt_ms / 1000.0
        self.fruits.update(dt, self.width, self.height)

    def render(self) -> None:
        self.screen.fill((18, 18, 22))

        if config.DEBUG:
            # Corner guides: active-region corners should reach these.
            margin = 18
            color = (60, 60, 70)
            corners = [
                (margin, margin),
                (self.width - margin, margin),
                (margin, self.height - margin),
                (self.width - margin, self.height - margin),
            ]
            for cx, cy in corners:
                pygame.draw.circle(self.screen, color, (cx, cy), 6, 1)
            pygame.draw.rect(
                self.screen,
                color,
                (margin, margin, self.width - 2 * margin, self.height - 2 * margin),
                1,
            )

        for fruit in self.fruits.active_fruits:
            fruit.draw(self.screen)

        if self.blade_active and len(self.blade_points) >= 2:
            pts = [(int(x), int(y)) for x, y in self.blade_points]
            pygame.draw.lines(
                self.screen,
                (255, 255, 255),
                False,
                pts,
                config.BLADE_THICKNESS,
            )
            # Round joints so the thick polyline doesn't show seams at bends.
            radius = max(config.BLADE_THICKNESS // 2, 1)
            for p in pts:
                pygame.draw.circle(self.screen, (255, 255, 255), p, radius)

        if self.fingertip is not None:
            x, y = int(self.fingertip[0]), int(self.fingertip[1])
            tip_color = (255, 220, 40) if self.blade_active else (120, 120, 130)
            pygame.draw.circle(self.screen, tip_color, (x, y), 14)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 4)

        if self.fingertip is None:
            text = "no hand"
        else:
            finger = self.pointer_finger or "tip"
            state = "swipe" if self.blade_active else "idle"
            text = f"{finger} tip · {state}"
            if config.DEBUG:
                text += f" · {self.swipe_velocity:.0f} px/s"
        label = self._font.render(text, True, (200, 200, 200))
        self.screen.blit(label, (16, 14))
        pygame.display.flip()

    def tick(self) -> None:
        self.clock.tick(self.target_fps)

    def quit(self) -> None:
        pygame.quit()
