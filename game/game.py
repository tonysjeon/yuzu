"""Pygame game window and loop helpers."""

from __future__ import annotations

import pygame

from src import config


class Game:
    """Minimal play surface: show the fingertip as a circle."""

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

    def update(self) -> None:
        """Placeholder for fruit / blade updates in later milestones."""

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

        if self.fingertip is not None:
            x, y = int(self.fingertip[0]), int(self.fingertip[1])
            pygame.draw.circle(self.screen, (255, 220, 40), (x, y), 14)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 4)

        if self.fingertip is None:
            text = "no hand"
        else:
            finger = self.pointer_finger or "tip"
            text = f"{finger} tip"
        label = self._font.render(text, True, (200, 200, 200))
        self.screen.blit(label, (16, 14))
        pygame.display.flip()

    def tick(self) -> None:
        self.clock.tick(self.target_fps)

    def quit(self) -> None:
        pygame.quit()
