"""Pygame game window and loop helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from game.background import build_dojo_wall
from game.fruit import cut_fruit_icon, warm_cut_cache
from game.fruit_manager import FruitManager
from src import config

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def format_clock(seconds: float) -> str:
    """Remaining time as m:ss, matching Fruit Ninja's timer."""
    secs = max(0, int(math.ceil(max(seconds, 0.0) - 1e-9)))
    minutes, rest = divmod(secs, 60)
    return f"{minutes}:{rest:02d}"


def combo_multiplier(streak: int) -> int:
    """Score multiplier from a fruit streak. 1x until 3, then +1x every 3 hits."""
    return min(config.MAX_MULTIPLIER, 1 + max(streak, 0) // 3)


# Highest matching rank wins. Thresholds are Fruit Ninja-style shoutouts.
_COMBO_NAMES: tuple[tuple[int, str], ...] = (
    (3, "NICE"),
    (4, "GREAT"),
    (5, "AWESOME"),
    (6, "AMAZING"),
    (8, "INSANE"),
    (10, "UNBELIEVABLE"),
    (12, "LEGENDARY"),
    (15, "YUZU NINJA"),
)


def combo_name(count: int) -> str | None:
    """Shoutout for a streak or same-swipe combo, or None below 3."""
    label: str | None = None
    for threshold, name in _COMBO_NAMES:
        if count >= threshold:
            label = name
    return label


@dataclass
class _Floater:
    text: str
    x: float
    y: float
    life: float
    max_life: float
    kind: str


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
        self.blade_segments: list[tuple[float, float]] = []
        self.fruits = FruitManager()
        self.score = 0
        self.time_left = float(config.ROUND_SECONDS)
        self.game_over = False
        self.paused = False
        self._palm_frames = 0
        self._play_frames = 0
        self.streak = 0
        self.multiplier = 1
        self._combo_timer = 0.0
        self._swipe_hits = 0
        self._prev_blade_active = False
        self._streak_pulse = 1.0
        self._floaters: list[_Floater] = []
        # Session best only — closing the game clears it.
        self.high_score = 0
        self._score_font: pygame.font.Font | None = None
        self._best_font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None
        self._overlay_font: pygame.font.Font | None = None
        self._hint_font: pygame.font.Font | None = None
        self._yuzu_icon: pygame.Surface | None = None
        self._background: pygame.Surface | None = None
        self._clock_block_w = 0
        self._text_cache: dict[tuple, pygame.Surface] = {}
        self._fx: pygame.Surface | None = None
        self._set_mode(width, height)

    def _pick_font(self, names: tuple[str, ...], size: int, bold: bool = False) -> pygame.font.Font:
        for name in names:
            path = pygame.font.match_font(name, bold=bold) or pygame.font.match_font(name)
            if not path:
                continue
            try:
                return pygame.font.Font(path, size)
            except pygame.error:
                continue
        return pygame.font.SysFont("helvetica", size, bold=bold)

    def _file_font(self, filename: str, size: int) -> pygame.font.Font | None:
        path = _FONT_DIR / filename
        if not path.is_file():
            return None
        try:
            return pygame.font.Font(str(path), size)
        except pygame.error:
            return None

    def _display_font(self, size: int, bold: bool = True) -> pygame.font.Font:
        for name in ("Bungee-Regular.ttf", "Outfit-ExtraBold.ttf" if bold else "Outfit-Medium.ttf"):
            font = self._file_font(name, size)
            if font is not None:
                return font
        return self._pick_font(
            (
                "arial black",
                "impact",
                "avenir next",
                "helvetica neue",
                "futura",
            ),
            size,
            bold=bold,
        )

    def _set_mode(self, width: int, height: int) -> None:
        width = max(width, 320)
        height = max(height, 240)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(
            (width, height),
            pygame.RESIZABLE,
        )
        self._background = build_dojo_wall(width, height)
        self._fx = pygame.Surface((width, height), pygame.SRCALPHA)
        self._text_cache = {}
        number_size = max(int(height * 0.095), 46)
        best_size = max(int(height * 0.034), 18)
        title_size = max(int(height * 0.12), 52)
        overlay_size = max(int(height * 0.05), 26)
        hint_size = max(int(height * 0.03), 16)
        icon_size = max(int(number_size * 0.82), 36)
        self._score_font = self._display_font(number_size, bold=True)
        self._best_font = self._display_font(best_size, bold=True)
        self._title_font = self._display_font(title_size, bold=True)
        self._overlay_font = self._display_font(overlay_size, bold=True)
        self._hint_font = self._file_font("Outfit-Medium.ttf", hint_size) or self._display_font(
            hint_size, bold=False
        )
        self._yuzu_icon = cut_fruit_icon("yuzu", icon_size)
        self._clock_block_w = 0
        if self._score_font is not None:
            self._clock_block_w = self._ninja_text(self._score_font, "0:00").get_width()
        warm_cut_cache()

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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                if not self.game_over:
                    self._toggle_pause()
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_r,
                pygame.K_SPACE,
                pygame.K_RETURN,
            ):
                if self.game_over:
                    self.restart()

    def restart(self) -> None:
        self.fruits.clear()
        self.score = 0
        self.time_left = float(config.ROUND_SECONDS)
        self.game_over = False
        self.paused = False
        self._palm_frames = 0
        self._play_frames = 0
        self.streak = 0
        self.multiplier = 1
        self._combo_timer = 0.0
        self._swipe_hits = 0
        self._prev_blade_active = False
        self._streak_pulse = 1.0
        self._floaters = []
        self.blade_points = []
        self.blade_segments = []
        self.blade_active = False

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self._palm_frames = 0
        self._play_frames = 0
        if self.paused:
            self.blade_points = []
            self.blade_segments = []
            self.blade_active = False

    def set_palm(self, palm: bool | None) -> None:
        """Debounced open palm → pause, pointing hand → resume. None leaves state as-is."""
        if self.game_over or palm is None:
            return
        if palm:
            self._palm_frames += 1
            self._play_frames = 0
            if not self.paused and self._palm_frames >= config.PALM_PAUSE_FRAMES:
                self.paused = True
                self.blade_points = []
                self.blade_segments = []
                self.blade_active = False
        else:
            self._play_frames += 1
            self._palm_frames = 0
            if self.paused and self._play_frames >= config.PALM_RESUME_FRAMES:
                self.paused = False

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
        segments: list[tuple[float, float]] | None = None,
    ) -> None:
        self.blade_points = points
        self.blade_active = active
        self.swipe_velocity = velocity
        self.blade_segments = segments if segments is not None else points

    def update(self) -> None:
        # get_time is ms since the previous tick(); first frame may be 0.
        dt_ms = self.clock.get_time()
        dt = (1.0 / self.target_fps) if dt_ms <= 0 else dt_ms / 1000.0
        if self.paused and not self.game_over:
            return
        if self.game_over or self.time_left <= 0.0:
            if not self.game_over:
                self._end_game()
            self.fruits.update(dt, self.width, self.height, spawn=False)
            return
        elapsed = max(0.0, config.ROUND_SECONDS - self.time_left)
        self.fruits.set_progress(elapsed / max(config.ROUND_SECONDS, 1e-6))
        self.fruits.update(dt, self.width, self.height)
        if self.blade_active and len(self.blade_segments) >= 2:
            hits = self.fruits.slice_with_blade(self.blade_segments)
            if hits:
                self._register_hits(hits, self.fruits.last_slice_at)
        if self._prev_blade_active and not self.blade_active:
            self._finish_swipe()
        self._prev_blade_active = self.blade_active
        self._tick_combo(dt)
        self._tick_floaters(dt)
        self.time_left = max(0.0, self.time_left - dt)
        if self.time_left <= 0.0:
            self._end_game()

    def _register_hits(self, hits: int, pos: tuple[float, float] | None) -> None:
        self._combo_timer = config.COMBO_WINDOW
        self._swipe_hits += hits
        self._streak_pulse = 1.22
        for _ in range(hits):
            previous_name = combo_name(self.streak)
            self.streak += 1
            new_mult = combo_multiplier(self.streak)
            at = pos or (self.width * 0.5, self.height * 0.38)
            if new_mult > self.multiplier:
                self.multiplier = new_mult
                self._streak_pulse = 1.32
            name = combo_name(self.streak)
            if name and name != previous_name:
                self._add_floater(name, at[0], at[1], "combo")
            self.score += self.multiplier

    def _finish_swipe(self) -> None:
        n = self._swipe_hits
        self._swipe_hits = 0
        if n < config.SWIPE_COMBO_MIN:
            return
        self.score += n * self.multiplier
        x, y = self.fruits.last_slice_at or (self.width * 0.5, self.height * 0.4)
        label = combo_name(n) or f"COMBO  x{n}"
        self._add_floater(label, x, y, "combo")

    def _tick_combo(self, dt: float) -> None:
        if self.streak <= 0:
            return
        self._combo_timer -= dt
        if self._combo_timer > 0.0:
            return
        self.streak = 0
        self.multiplier = 1
        self._streak_pulse = 1.0

    def _add_floater(self, text: str, x: float, y: float, kind: str) -> None:
        life = 0.95 if kind == "combo" else 0.7
        self._floaters.append(_Floater(text=text, x=x, y=y, life=life, max_life=life, kind=kind))

    def _tick_floaters(self, dt: float) -> None:
        keep: list[_Floater] = []
        for floater in self._floaters:
            floater.life -= dt
            floater.y -= 90.0 * dt
            if floater.life > 0.0:
                keep.append(floater)
        self._floaters = keep
        self._streak_pulse += (1.0 - self._streak_pulse) * min(dt * 10.0, 1.0)

    def _end_game(self) -> None:
        self._finish_swipe()
        self.game_over = True
        self.time_left = 0.0
        self.high_score = max(self.high_score, self.score)

    def render(self) -> None:
        if self._background is not None:
            self.screen.blit(self._background, (0, 0))
        else:
            self.screen.fill((22, 12, 8))

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
        for piece in self.fruits.pieces:
            piece.draw(self.screen)
        self.fruits.splatter.draw(self.screen)
        self._draw_floaters()

        if not self.game_over and not self.paused:
            if self.blade_active and len(self.blade_points) >= 2:
                self._draw_slash(self.blade_points)
            elif self.fingertip is not None:
                x, y = int(self.fingertip[0]), int(self.fingertip[1])
                pygame.draw.circle(self.screen, (90, 90, 98), (x, y), 5, 1)

        self._draw_hud()
        if self.game_over:
            self._draw_game_over()
        elif self.paused:
            self._draw_paused()
        pygame.display.flip()

    def _ninja_text(
        self,
        font: pygame.font.Font,
        text: str,
        *,
        top: tuple[int, int, int] = (255, 232, 96),
        bottom: tuple[int, int, int] = (255, 150, 24),
        outline: tuple[int, int, int] = (46, 18, 4),
        outline_width: int = 3,
        shadow: int = 4,
    ) -> pygame.Surface:
        """Fruit Ninja lettering: gold gradient fill, dark rim, drop shadow."""
        key = (id(font), text, top, bottom, outline, outline_width, shadow)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        mask = font.render(text, True, (255, 255, 255))
        w, h = mask.get_size()
        fill = pygame.Surface((w, h), pygame.SRCALPHA)
        if h > 0 and w > 0:
            arr = pygame.surfarray.pixels3d(fill)
            ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
            t = ys * ys * (3.0 - 2.0 * ys)
            top_a = np.asarray(top, dtype=np.float32)
            bot_a = np.asarray(bottom, dtype=np.float32)
            rows = top_a + (bot_a - top_a) * t[:, None]
            arr[:, :, :] = rows.astype(np.uint8)[None, :, :]
            del arr
            alpha = pygame.surfarray.pixels_alpha(fill)
            alpha[:, :] = 255
            del alpha
        fill.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        pad = outline_width + shadow + 1
        out = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
        rim = font.render(text, True, outline)
        shade = font.render(text, True, (0, 0, 0))
        shade.set_alpha(150)
        radius2 = outline_width * outline_width + 1
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx * dx + dy * dy <= radius2:
                    out.blit(shade, (pad + dx + shadow * 0.6, pad + dy + shadow))
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx * dx + dy * dy <= radius2:
                    out.blit(rim, (pad + dx, pad + dy))
        out.blit(fill, (pad, pad))
        self._text_cache[key] = out
        return out

    def _ninja_clock(
        self,
        font: pygame.font.Font,
        text: str,
        *,
        top: tuple[int, int, int] = (255, 232, 96),
        bottom: tuple[int, int, int] = (255, 150, 24),
        outline: tuple[int, int, int] = (46, 18, 4),
        outline_width: int = 3,
        shadow: int = 4,
    ) -> pygame.Surface:
        """Natural kerning, padded to a fixed width so the right edge stays put."""
        glyph = self._ninja_text(
            font,
            text,
            top=top,
            bottom=bottom,
            outline=outline,
            outline_width=outline_width,
            shadow=shadow,
        )
        block_w = max(self._clock_block_w, glyph.get_width())
        if glyph.get_width() == block_w:
            return glyph
        out = pygame.Surface((block_w, glyph.get_height()), pygame.SRCALPHA)
        out.blit(glyph, (block_w - glyph.get_width(), 0))
        return out

    def _draw_hud(self) -> None:
        if self._score_font is None or self._best_font is None:
            return
        margin_x, margin_y = 18, 10
        icon = self._yuzu_icon
        score = self._ninja_text(self._score_font, str(self.score))
        x = margin_x
        row_h = max(score.get_height(), icon.get_height() if icon is not None else 0)
        score_y = margin_y + (row_h - score.get_height()) // 2
        if icon is not None:
            self.screen.blit(icon, (x, margin_y + (row_h - icon.get_height()) // 2 + 2))
            x += icon.get_width() + 6
        self.screen.blit(score, (x - 4, score_y))

        best = self._ninja_text(
            self._best_font,
            f"BEST:  {self.high_score}",
            top=(255, 214, 92),
            bottom=(240, 138, 30),
            outline_width=2,
            shadow=3,
        )
        # Sit under the score digits; ninja_text surfaces include outline pad.
        self.screen.blit(best, (margin_x, score_y + score.get_height() - 26))

        if self.streak >= 2 and self._score_font is not None:
            streak = self._ninja_text(self._score_font, f"x{self.streak}")
            if self._streak_pulse > 1.02:
                streak = pygame.transform.rotozoom(streak, 0.0, self._streak_pulse)
            sx = self.width // 2 - streak.get_width() // 2
            sy = margin_y - 6
            self.screen.blit(streak, (sx, sy))
            name = combo_name(self.streak)
            if name and self._best_font is not None:
                shout = self._ninja_text(
                    self._best_font,
                    name,
                    top=(255, 224, 96),
                    bottom=(255, 140, 28),
                    outline_width=2,
                    shadow=3,
                )
                self.screen.blit(
                    shout,
                    (
                        self.width // 2 - shout.get_width() // 2,
                        sy + streak.get_height() - 28,
                    ),
                )

        urgent = self.time_left <= 10.0
        clock = self._ninja_clock(
            self._score_font,
            format_clock(self.time_left),
            top=(255, 110, 70) if urgent else (255, 232, 96),
            bottom=(215, 30, 20) if urgent else (255, 150, 24),
        )
        self.screen.blit(clock, (self.width - margin_x - clock.get_width(), margin_y))

    def _draw_game_over(self) -> None:
        if (
            self._title_font is None
            or self._overlay_font is None
            or self._hint_font is None
        ):
            return
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 168))
        self.screen.blit(dim, (0, 0))

        cx = self.width // 2
        y = int(self.height * 0.28)
        title = self._ninja_text(
            self._title_font,
            "TIME'S UP",
            top=(255, 120, 80),
            bottom=(220, 36, 24),
            outline_width=4,
            shadow=6,
        )
        self.screen.blit(title, (cx - title.get_width() // 2, y))
        y += title.get_height() + 10
        for text in (f"SCORE  {self.score}", f"BEST  {self.high_score}"):
            line = self._ninja_text(self._overlay_font, text)
            self.screen.blit(line, (cx - line.get_width() // 2, y))
            y += line.get_height() + 2
        hint = self._ninja_text(
            self._hint_font,
            "PRESS R TO RESTART",
            top=(240, 236, 228),
            bottom=(196, 190, 180),
            outline_width=2,
            shadow=2,
        )
        self.screen.blit(hint, (cx - hint.get_width() // 2, y + 14))

    def _draw_paused(self) -> None:
        if self._title_font is None or self._hint_font is None:
            return
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        self.screen.blit(dim, (0, 0))

        cx = self.width // 2
        title = self._ninja_text(
            self._title_font,
            "PAUSED",
            top=(255, 232, 96),
            bottom=(255, 150, 24),
            outline_width=4,
            shadow=6,
        )
        y = int(self.height * 0.38) - title.get_height() // 2
        self.screen.blit(title, (cx - title.get_width() // 2, y))
        hint = self._ninja_text(
            self._hint_font,
            "POINT TO RESUME",
            top=(240, 236, 228),
            bottom=(196, 190, 180),
            outline_width=2,
            shadow=2,
        )
        self.screen.blit(
            hint,
            (cx - hint.get_width() // 2, y + title.get_height() + 12),
        )

    def _draw_floaters(self) -> None:
        font = self._overlay_font if self._overlay_font is not None else self._score_font
        if font is None:
            return
        for floater in self._floaters:
            t = max(floater.life / floater.max_life, 0.0)
            if floater.kind == "combo":
                use = self._title_font or font
                top, bottom = (255, 236, 96), (255, 140, 24)
            else:
                use = font
                top, bottom = (255, 210, 80), (255, 90, 30)
            if use is None:
                continue
            glyph = self._ninja_text(
                use,
                floater.text,
                top=top,
                bottom=bottom,
                outline_width=4 if floater.kind == "combo" else 3,
                shadow=5 if floater.kind == "combo" else 3,
            )
            scale = 0.82 + 0.28 * t
            if scale != 1.0:
                glyph = pygame.transform.rotozoom(glyph, 0.0, scale)
            glyph.set_alpha(int(255 * min(t / 0.25, 1.0) * min(t * 2.2, 1.0)))
            self.screen.blit(
                glyph,
                (int(floater.x) - glyph.get_width() // 2, int(floater.y) - glyph.get_height() // 2),
            )

    def _draw_slash(self, points: list[tuple[float, float]]) -> None:
        """Fruit Ninja-style tapered blade instead of a fingertip circle."""
        n = len(points)
        if n < 2:
            return
        left: list[tuple[float, float]] = []
        right: list[tuple[float, float]] = []
        head = config.BLADE_HEAD_WIDTH
        tail = config.BLADE_TAIL_WIDTH
        for i, (x, y) in enumerate(points):
            t = i / (n - 1)
            width = tail + (head - tail) * (t * t)
            if i < n - 1:
                dx = points[i + 1][0] - x
                dy = points[i + 1][1] - y
            else:
                dx = x - points[i - 1][0]
                dy = y - points[i - 1][1]
            length = math.hypot(dx, dy) or 1.0
            px, py = -dy / length, dx / length
            left.append((x + px * width, y + py * width))
            right.append((x - px * width, y - py * width))
        poly = [(int(x), int(y)) for x, y in left + right[::-1]]
        if len(poly) >= 3:
            overlay = self._fx
            if overlay is None:
                overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 0))
            pygame.draw.polygon(overlay, (255, 255, 255, 220), poly)
            pygame.draw.polygon(overlay, (180, 230, 255, 90), poly, width=1)
            self.screen.blit(overlay, (0, 0))
        tip = points[-1]
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (int(tip[0]), int(tip[1])),
            max(int(config.BLADE_HEAD_WIDTH * 0.45), 3),
        )

    def tick(self) -> None:
        self.clock.tick(self.target_fps)

    def quit(self) -> None:
        pygame.quit()
