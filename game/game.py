"""Pygame game window and loop helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from game.background import build_dojo_wall
from game.fruit import FRUIT_STYLES, Fruit, cut_fruit_icon, warm_cut_cache
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


def combo_heat(count: int) -> float:
    """0 at the first named combo, 1 at the highest shoutout."""
    rank = 0
    for i, (threshold, _) in enumerate(_COMBO_NAMES):
        if count >= threshold:
            rank = i + 1
    if rank <= 0:
        return 0.0
    return (rank - 1) / max(len(_COMBO_NAMES) - 1, 1)


def combo_colors(heat: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Gold at low combos, white-hot into red at the top ranks."""
    t = min(max(heat, 0.0), 1.0)
    t = t * t * (3.0 - 2.0 * t)
    top = (
        int(255),
        int(232 + (252 - 232) * t),
        int(96 + (220 - 96) * t),
    )
    bottom = (
        int(255),
        int(150 + (28 - 150) * t),
        int(24 + (12 - 24) * t),
    )
    return top, bottom


@dataclass
class _Floater:
    text: str
    x: float
    y: float
    life: float
    max_life: float
    kind: str
    heat: float = 0.0


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
        self._fist_frames = 0
        self._title_start_in: float | None = None
        self._title_pulse = 0.0
        self._title_sliding = False
        self._title_slide = 0.0
        self._timer_hold = 0.0
        self.streak = 0
        self.multiplier = 1
        self._combo_timer = 0.0
        self._swipe_hits = 0
        self._prev_blade_active = False
        self._streak_pulse = 1.0
        self._floaters: list[_Floater] = []
        # Session best only — closing the game clears it.
        self.high_score = 0
        self.ended_by_bomb = False
        self.on_title = True
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
        self._title_layer: pygame.Surface | None = None
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
        self._title_layer = pygame.Surface((width, height), pygame.SRCALPHA)
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
        if self.on_title and self._title_start_in is None and not self._title_sliding:
            self._place_title_fruit()

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
                if not self.game_over and not self.on_title:
                    self._toggle_pause()
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_r,
                pygame.K_SPACE,
                pygame.K_RETURN,
            ):
                if self.game_over:
                    self.restart()

    def restart(self, *, from_title: bool = False) -> None:
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
        self.ended_by_bomb = False
        self.on_title = False
        self._title_start_in = None
        self._title_pulse = 0.0
        self._title_sliding = False
        self._title_slide = 0.0
        self._timer_hold = config.ROUND_TIMER_DELAY if from_title else 0.0
        self._fist_frames = 0
        self.blade_points = []
        self.blade_segments = []
        self.blade_active = False
        if from_title:
            self.fruits._time_to_spawn = config.TITLE_FIRST_SPAWN

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self._palm_frames = 0
        self._play_frames = 0
        self._fist_frames = 0
        if self.paused:
            self.blade_points = []
            self.blade_segments = []
            self.blade_active = False

    def go_to_title(self) -> None:
        """Leave a paused round and return to the start screen."""
        self.fruits.clear()
        self.score = 0
        self.time_left = float(config.ROUND_SECONDS)
        self.game_over = False
        self.paused = False
        self._palm_frames = 0
        self._play_frames = 0
        self._fist_frames = 0
        self.streak = 0
        self.multiplier = 1
        self._combo_timer = 0.0
        self._swipe_hits = 0
        self._prev_blade_active = False
        self._streak_pulse = 1.0
        self._floaters = []
        self.ended_by_bomb = False
        self.on_title = True
        self._title_start_in = None
        self._title_pulse = 0.0
        self._title_sliding = False
        self._title_slide = 0.0
        self._timer_hold = 0.0
        self.blade_points = []
        self.blade_segments = []
        self.blade_active = False
        self._place_title_fruit()

    def _place_title_fruit(self) -> None:
        """Put a whole yuzu under the title instructions so it's the thing to slash."""
        color, radius = FRUIT_STYLES["yuzu"]
        x, y = self._title_fruit_pos(radius)
        self.fruits.clear()
        self.fruits._time_to_spawn = 10.0
        self.fruits.fruits = [
            Fruit(
                x=x,
                y=y,
                velocity_x=0.0,
                velocity_y=0.0,
                radius=radius,
                sliced=False,
                active=True,
                fruit_type="yuzu",
                color=color,
                rotation=0.0,
                rotation_speed=0.0,
            )
        ]

    def _title_fruit_pos(self, radius: float) -> tuple[float, float]:
        layout = self._title_layout(radius)
        return self.width * 0.5, layout["fruit_y"]

    def _title_layout(self, radius: float | None = None) -> dict[str, float]:
        """Stack YUZU, the hint, then the fruit as one centered block."""
        if radius is None:
            radius = FRUIT_STYLES["yuzu"][1]
        title_h = 80.0
        hint_h = 40.0
        if self._title_font is not None:
            title_h = float(
                self._ninja_text(self._title_font, "YUZU", outline_width=4, shadow=6).get_height()
            )
        if self._hint_font is not None:
            hint_h = float(
                self._ninja_text(
                    self._hint_font,
                    "POINT A FINGER AND SLASH",
                    top=(240, 236, 228),
                    bottom=(196, 190, 180),
                    outline_width=2,
                    shadow=2,
                ).get_height()
            )
        visual = radius * 2.4
        # ninja_text surfaces include outline pad; overlap so the glyphs sit tight.
        title_to_hint = title_h - 20.0
        title_y = self.height * 0.30
        hint_y = title_y + title_to_hint
        # Sit the slash target lower, with room for the sprite above the bottom edge.
        fruit_y = min(self.height * 0.62, self.height - visual * 0.5 - 28.0)
        fruit_y = max(fruit_y, hint_y + hint_h * 0.35 + visual * 0.5)
        return {"title_y": title_y, "hint_y": hint_y, "fruit_y": fruit_y}

    def set_palm(self, palm: bool | None, fist: bool = False) -> None:
        """Open palm pauses; pointing resumes; a fist from pause returns to title."""
        if self.game_over or self.on_title or palm is None:
            return
        if palm:
            self._palm_frames += 1
            self._play_frames = 0
            self._fist_frames = 0
            if not self.paused and self._palm_frames >= config.PALM_PAUSE_FRAMES:
                self.paused = True
                self.blade_points = []
                self.blade_segments = []
                self.blade_active = False
            return
        if fist:
            self._palm_frames = 0
            self._play_frames = 0
            if not self.paused:
                self._fist_frames = 0
                return
            self._fist_frames += 1
            if self._fist_frames >= config.FIST_MENU_FRAMES:
                self.go_to_title()
            return
        self._play_frames += 1
        self._palm_frames = 0
        self._fist_frames = 0
        if self.paused and self._play_frames >= config.PALM_RESUME_FRAMES:
            self.paused = False

    def _tick_title(self, dt: float) -> None:
        if self._title_start_in is not None:
            self.fruits.update(dt, self.width, self.height, spawn=False)
            self._title_start_in -= dt
            if self._title_start_in <= 0.0:
                self._title_start_in = None
                self._title_sliding = True
                self._title_slide = 0.0
            return
        if self._title_sliding:
            self.fruits.update(dt, self.width, self.height, spawn=False)
            duration = max(config.TITLE_SLIDE_SECONDS, 1e-6)
            self._title_slide += dt / duration
            if self._title_slide >= 1.0:
                self.restart(from_title=True)
            return
        self._title_pulse += dt * 2.1
        if self.blade_active and len(self.blade_segments) >= 2:
            hits = self.fruits.slice_with_blade(self.blade_segments)
            if hits:
                self._title_start_in = config.TITLE_START_DELAY

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
        if self.on_title:
            self._tick_title(dt)
            return
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
            if self.fruits.hit_bomb:
                self._end_game(bomb=True)
                return
            if hits:
                self._register_hits(hits, self.fruits.last_slice_at)
        if self._prev_blade_active and not self.blade_active:
            self._finish_swipe()
        self._prev_blade_active = self.blade_active
        self._tick_combo(dt)
        self._tick_floaters(dt)
        if self._timer_hold > 0.0:
            self._timer_hold = max(0.0, self._timer_hold - dt)
            return
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
                self._streak_pulse = 1.32 + 0.16 * combo_heat(self.streak)
            name = combo_name(self.streak)
            if name and name != previous_name:
                self._add_floater(name, at[0], at[1], "combo", heat=combo_heat(self.streak))
            self.score += self.multiplier

    def _finish_swipe(self) -> None:
        n = self._swipe_hits
        self._swipe_hits = 0
        if n < config.SWIPE_COMBO_MIN:
            return
        self.score += n * self.multiplier
        x, y = self.fruits.last_slice_at or (self.width * 0.5, self.height * 0.4)
        label = combo_name(n) or f"COMBO  x{n}"
        self._add_floater(label, x, y, "combo", heat=combo_heat(n))

    def _tick_combo(self, dt: float) -> None:
        if self.streak <= 0:
            return
        self._combo_timer -= dt
        if self._combo_timer > 0.0:
            return
        self.streak = 0
        self.multiplier = 1
        self._streak_pulse = 1.0

    def _add_floater(
        self,
        text: str,
        x: float,
        y: float,
        kind: str,
        heat: float = 0.0,
    ) -> None:
        life = 0.95 if kind == "combo" else 0.7
        self._floaters.append(
            _Floater(text=text, x=x, y=y, life=life, max_life=life, kind=kind, heat=heat)
        )

    def _tick_floaters(self, dt: float) -> None:
        keep: list[_Floater] = []
        for floater in self._floaters:
            floater.life -= dt
            floater.y -= 90.0 * dt
            if floater.life > 0.0:
                keep.append(floater)
        self._floaters = keep
        self._streak_pulse += (1.0 - self._streak_pulse) * min(dt * 10.0, 1.0)

    def _end_game(self, *, bomb: bool = False) -> None:
        if bomb:
            self._swipe_hits = 0
        else:
            self._finish_swipe()
        self.game_over = True
        self.ended_by_bomb = bomb
        self.time_left = 0.0
        self.streak = 0
        self.multiplier = 1
        self._floaters = []
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

        if self.on_title:
            if self._title_sliding:
                self._draw_hud()
            self.screen.blit(self._compose_title_layer(), (0, self._title_slide_offset()))
            if not self._title_sliding:
                if self.blade_active and len(self.blade_points) >= 2:
                    self._draw_slash(self.blade_points)
                elif self.fingertip is not None:
                    x, y = int(self.fingertip[0]), int(self.fingertip[1])
                    pygame.draw.circle(self.screen, (90, 90, 98), (x, y), 5, 1)
            pygame.display.flip()
            return

        for fruit in self.fruits.active_fruits:
            fruit.draw(self.screen)
        for piece in self.fruits.pieces:
            piece.draw(self.screen)
        self.fruits.splatter.draw(self.screen)
        if not self.game_over:
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

        if not self.game_over and self.streak >= 2 and self._score_font is not None:
            heat = combo_heat(self.streak)
            top, bottom = combo_colors(heat)
            rank_scale = 1.0 + 0.42 * heat
            outline = 3 + int(round(2 * heat))
            streak = self._ninja_text(
                self._score_font,
                f"x{self.streak}",
                top=top,
                bottom=bottom,
                outline_width=outline,
                shadow=4 + int(round(2 * heat)),
            )
            scale = rank_scale
            if self._streak_pulse > 1.02:
                scale *= self._streak_pulse
            if scale != 1.0:
                streak = pygame.transform.rotozoom(streak, 0.0, scale)
            sx = self.width // 2 - streak.get_width() // 2
            sy = margin_y - 6 - int(8 * heat)
            self.screen.blit(streak, (sx, sy))
            name = combo_name(self.streak)
            if name and self._best_font is not None:
                name_font = self._overlay_font if heat >= 0.5 and self._overlay_font is not None else self._best_font
                shout = self._ninja_text(
                    name_font,
                    name,
                    top=top,
                    bottom=bottom,
                    outline_width=2 + int(round(2 * heat)),
                    shadow=3 + int(round(2 * heat)),
                )
                name_scale = rank_scale * (1.08 if heat >= 0.5 else 1.0)
                if self._streak_pulse > 1.02:
                    name_scale *= 1.0 + (self._streak_pulse - 1.0) * 0.6
                if name_scale != 1.0:
                    shout = pygame.transform.rotozoom(shout, 0.0, name_scale)
                overlap = int((26 + 6 * heat) * rank_scale)
                self.screen.blit(
                    shout,
                    (
                        self.width // 2 - shout.get_width() // 2,
                        sy + streak.get_height() - overlap,
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
            "GAME OVER" if self.ended_by_bomb else "TIME'S UP",
            top=(255, 90, 50) if self.ended_by_bomb else (255, 120, 80),
            bottom=(180, 12, 10) if self.ended_by_bomb else (220, 36, 24),
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
        y = int(self.height * 0.36) - title.get_height() // 2
        self.screen.blit(title, (cx - title.get_width() // 2, y))
        y += title.get_height() + max(int(self.height * 0.018), 8)
        icon_size = max(int(self.height * 0.055), 28)
        gap = max(int(icon_size * 0.28), 10)
        rows = (
            ("point", "POINT TO RESUME"),
            ("fist", "FIST FOR MENU"),
        )
        parts = [
            (self._gesture_icon(kind, icon_size), self._ninja_text(
                self._hint_font,
                text,
                top=(240, 236, 228),
                bottom=(196, 190, 180),
                outline_width=2,
                shadow=2,
            ))
            for kind, text in rows
        ]
        row_w = max(icon.get_width() + gap + label.get_width() for icon, label in parts)
        x0 = cx - row_w // 2
        line_gap = max(int(self.height * 0.05), 26)
        for icon, label in parts:
            row_h = max(icon.get_height(), label.get_height())
            self.screen.blit(icon, (x0, y + (row_h - icon.get_height()) // 2))
            self.screen.blit(
                label,
                (x0 + icon.get_width() + gap, y + (row_h - label.get_height()) // 2),
            )
            y += row_h + line_gap

    def _gesture_icon(self, kind: str, size: int) -> pygame.Surface:
        """Cream silhouette of a pointing hand or a fist, matching the HUD outline."""
        key = ("gesture_icon", kind, size)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        ss = 4
        inner = max(size, 8) * ss
        fill = (248, 244, 236)
        outline = (46, 18, 4)
        ow = max(ss * 2, 4)
        body = pygame.Surface((inner, inner), pygame.SRCALPHA)
        self._paint_gesture(body, kind, fill, outline, ow)
        shadow = pygame.Surface((inner, inner), pygame.SRCALPHA)
        self._paint_gesture(shadow, kind, (0, 0, 0), (0, 0, 0), ow)
        shadow.set_alpha(150)
        pad = ow + ss * 2
        out = pygame.Surface((inner + pad, inner + pad), pygame.SRCALPHA)
        out.blit(shadow, (int(ss * 1.2), ss * 2))
        out.blit(body, (0, 0))
        icon = pygame.transform.smoothscale(out, (size, size))
        self._text_cache[key] = icon
        return icon

    def _paint_gesture(
        self,
        surf: pygame.Surface,
        kind: str,
        fill: tuple[int, int, int],
        outline: tuple[int, int, int],
        ow: int,
    ) -> None:
        s = float(surf.get_width())

        def rr(u: float, v: float, w: float, h: float, r: float) -> None:
            x, y, ww, hh = u * s, v * s, w * s, h * s
            rad = max(int(r * s), 1)
            pygame.draw.rect(
                surf,
                outline,
                (int(x - ow), int(y - ow), int(ww + 2 * ow), int(hh + 2 * ow)),
                border_radius=rad + ow,
            )
            pygame.draw.rect(
                surf,
                fill,
                (int(x), int(y), int(ww), int(hh)),
                border_radius=rad,
            )

        def ell(u: float, v: float, w: float, h: float) -> None:
            x, y, ww, hh = u * s, v * s, w * s, h * s
            pygame.draw.ellipse(
                surf,
                outline,
                (int(x - ow), int(y - ow), int(ww + 2 * ow), int(hh + 2 * ow)),
            )
            pygame.draw.ellipse(surf, fill, (int(x), int(y), int(ww), int(hh)))

        def circ(u: float, v: float, r: float) -> None:
            pygame.draw.circle(
                surf,
                outline,
                (int(u * s), int(v * s)),
                int(r * s) + ow,
            )
            pygame.draw.circle(surf, fill, (int(u * s), int(v * s)), max(int(r * s), 1))

        if kind == "point":
            rr(0.72, 0.40, 0.12, 0.22, 0.06)  # pinky
            rr(0.60, 0.34, 0.14, 0.26, 0.07)  # ring
            rr(0.46, 0.28, 0.16, 0.32, 0.08)  # middle
            rr(0.22, 0.44, 0.54, 0.36, 0.16)  # palm
            rr(0.28, 0.06, 0.20, 0.52, 0.10)  # index
            ell(0.08, 0.50, 0.28, 0.22)  # thumb
            rr(0.34, 0.76, 0.32, 0.16, 0.07)  # wrist
            return
        # Fist: knuckles up, thumb wrapped on the left.
        rr(0.34, 0.70, 0.34, 0.18, 0.08)  # wrist
        rr(0.20, 0.26, 0.60, 0.50, 0.22)  # body
        circ(0.32, 0.28, 0.11)
        circ(0.46, 0.24, 0.12)
        circ(0.60, 0.24, 0.12)
        circ(0.74, 0.30, 0.10)
        ell(0.06, 0.42, 0.32, 0.26)

    def _title_slide_offset(self) -> int:
        """Ease-in drop so the title page hangs, then falls away."""
        t = min(max(self._title_slide, 0.0), 1.0)
        eased = t * t * t
        return int(eased * self.height)

    def _compose_title_layer(self) -> pygame.Surface:
        layer = self._title_layer
        if layer is None:
            layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self._title_layer = layer
        layer.fill((0, 0, 0, 0))
        self._draw_title(layer)
        for fruit in self.fruits.active_fruits:
            if not self._title_sliding:
                self._draw_title_focus(fruit, layer)
            fruit.draw(layer)
        for piece in self.fruits.pieces:
            piece.draw(layer)
        self.fruits.splatter.draw(layer)
        return layer

    def _draw_title_focus(self, fruit: Fruit, dest: pygame.Surface | None = None) -> None:
        """Expanding gold rings — a 'slash here' cue that reads at a glance."""
        cx, cy = int(fruit.x), int(fruit.y)
        overlay = self._fx
        if overlay is None:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))

        breathe = 0.5 + 0.5 * math.sin(self._title_pulse)
        halo_r = max(int(fruit.radius * (1.55 + 0.12 * breathe)), 8)
        halo_size = halo_r * 2 + 2
        halo = pygame.Surface((halo_size, halo_size), pygame.SRCALPHA)
        hx = hy = halo_size // 2
        xs = np.arange(halo_size, dtype=np.float32)[:, None]
        ys = np.arange(halo_size, dtype=np.float32)[None, :]
        dist = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2) / max(halo_r, 1)
        fall = np.clip(1.0 - dist, 0.0, 1.0)
        fall = fall * fall
        alpha = (fall * (90.0 + 70.0 * breathe)).astype(np.uint8)
        rgb = np.empty((halo_size, halo_size, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        rgb[..., 1] = 220
        rgb[..., 2] = 80
        pygame.surfarray.blit_array(halo, rgb)
        halo_a = pygame.surfarray.pixels_alpha(halo)
        halo_a[:, :] = alpha
        del halo_a
        overlay.blit(halo, halo.get_rect(center=(cx, cy)))

        for offset in (0.0, 0.5):
            u = (self._title_pulse * 0.28 + offset) % 1.0
            radius = int(fruit.radius * (1.12 + 1.05 * u))
            fade = (1.0 - u) ** 1.35
            ring_alpha = int(210 * fade)
            if ring_alpha < 8 or radius < 2:
                continue
            pygame.draw.circle(
                overlay,
                (255, 232, 96, ring_alpha),
                (cx, cy),
                radius,
                width=max(3, int(4 * fade + 1)),
            )
        target = dest if dest is not None else self.screen
        target.blit(overlay, (0, 0))

    def _draw_title(self, dest: pygame.Surface | None = None) -> None:
        if self._title_font is None or self._hint_font is None:
            return
        target = dest if dest is not None else self.screen
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 110))
        target.blit(dim, (0, 0))

        cx = self.width // 2
        title = self._ninja_text(
            self._title_font,
            "YUZU",
            outline_width=4,
            shadow=6,
        )
        hint = self._ninja_text(
            self._hint_font,
            "POINT A FINGER AND SLASH",
            top=(240, 236, 228),
            bottom=(196, 190, 180),
            outline_width=2,
            shadow=2,
        )
        layout = self._title_layout()
        target.blit(title, (cx - title.get_width() // 2, int(layout["title_y"])))
        target.blit(hint, (cx - hint.get_width() // 2, int(layout["hint_y"])))
        if self.high_score > 0 and self._overlay_font is not None:
            best = self._ninja_text(
                self._overlay_font,
                f"BEST  {self.high_score}",
            )
            fruit_bottom = layout["fruit_y"] + FRUIT_STYLES["yuzu"][1] * 1.2
            target.blit(
                best,
                (cx - best.get_width() // 2, int(fruit_bottom + 4)),
            )

    def _draw_floaters(self) -> None:
        font = self._overlay_font if self._overlay_font is not None else self._score_font
        if font is None:
            return
        for floater in self._floaters:
            t = max(floater.life / floater.max_life, 0.0)
            if floater.kind == "combo":
                use = self._title_font or font
                top, bottom = combo_colors(floater.heat)
                outline_w, shadow_w = 4 + int(round(2 * floater.heat)), 5 + int(round(2 * floater.heat))
            else:
                use = font
                top, bottom = (255, 210, 80), (255, 90, 30)
                outline_w, shadow_w = 3, 3
            if use is None:
                continue
            glyph = self._ninja_text(
                use,
                floater.text,
                top=top,
                bottom=bottom,
                outline_width=outline_w,
                shadow=shadow_w,
            )
            scale = (0.82 + 0.28 * t) * (1.0 + 0.38 * floater.heat)
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
