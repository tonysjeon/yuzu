"""Hand landmark detection via MediaPipe."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, TypedDict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

from src import config
from src.smoothing import AnchoredTip, OneEuroFilter

INDEX_TIP = 8
# Wrist plus the four finger MCP joints: large, blur-resistant structure.
_PALM_POINTS = (0, 5, 9, 13, 17)

# tip, pip, dip, mcp for each non-thumb finger
_FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "index": (8, 6, 7, 5),
    "middle": (12, 10, 11, 9),
    "ring": (16, 14, 15, 13),
    "pinky": (20, 18, 19, 17),
}

_DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "models"
    / "hand_landmarker.task"
)


class HandResult(TypedDict):
    detected: bool
    index_tip: tuple[float, float] | None
    raw_tip: tuple[float, float] | None
    confidence: float
    landmarks: list[tuple[float, float, float]]
    coasted: bool
    pointer_finger: str | None
    palm: bool


def _xy(landmarks: list[tuple[float, float, float]], index: int) -> np.ndarray:
    point = landmarks[index]
    return np.array([point[0], point[1]], dtype=np.float64)


def _palm_center(landmarks: list[tuple[float, float, float]]) -> tuple[float, float]:
    xs = [landmarks[i][0] for i in _PALM_POINTS]
    ys = [landmarks[i][1] for i in _PALM_POINTS]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _reach(
    landmarks: list[tuple[float, float, float]],
    tip_i: int,
) -> float:
    """Tip distance from the wrist, normalized by palm size.

    In a pointing pose the pointing finger reaches far past the curled ones,
    which stay clustered near the palm. Distance is far more robust to blur
    and foreshortening than joint angles.
    """
    wrist = _xy(landmarks, 0)
    palm = float(np.linalg.norm(_xy(landmarks, 9) - wrist))  # wrist -> middle MCP
    if palm < 1.0:
        return 0.0
    return float(np.linalg.norm(_xy(landmarks, tip_i) - wrist)) / palm


def is_open_palm(landmarks: list[tuple[float, float, float]]) -> bool:
    """True when every non-thumb finger is extended, not a pointing pose.

    A point has one finger reaching far past the rest. A pause palm has all
    four stretched a similar amount.
    """
    if len(landmarks) < 21:
        return False
    reaches = [_reach(landmarks, chain[0]) for chain in _FINGER_CHAINS.values()]
    if any(r < config.HAND_PALM_MIN_REACH for r in reaches):
        return False
    shortest = min(reaches)
    if shortest <= 0.0:
        return False
    return max(reaches) <= shortest * config.HAND_PALM_EVENNESS


class HandTracker:
    """Detect one hand and expose a stable pointing fingertip."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        max_hands: int = 1,
        min_detection_confidence: float = config.HAND_MIN_DETECTION,
        min_presence_confidence: float = config.HAND_MIN_PRESENCE,
        min_tracking_confidence: float = config.HAND_MIN_TRACKING,
        coast_ms: float = config.HAND_COAST_MS,
        max_coast_speed: float = config.HAND_MAX_COAST_SPEED,
        infer_max_width: int = config.HAND_INFER_MAX_WIDTH,
        pointer_switch_frames: int = config.HAND_POINTER_SWITCH_FRAMES,
    ) -> None:
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {path}. "
                "Download hand_landmarker.task into assets/models/."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )
        # Second detector for re-acquiring the hand from a crop around its last
        # position. Runs only while the main tracker has lost the hand.
        roi_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=config.HAND_ROI_MIN_PRESENCE,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._roi_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            roi_options
        )
        self._start = time.perf_counter()
        self._last_timestamp_ms = -1
        self._coast_s = coast_ms / 1000.0
        self._search_s = config.HAND_SEARCH_MS / 1000.0
        self._max_coast_speed = max_coast_speed
        self._infer_max_width = infer_max_width
        self._pointer_switch_frames = pointer_switch_frames

        self._filter = OneEuroFilter()
        self._anchor = AnchoredTip()
        self._last_tip: tuple[float, float] | None = None
        self._prev_tip: tuple[float, float] | None = None
        self._prev_seen = 0.0
        self._last_landmarks: list[tuple[float, float, float]] = []
        self._last_confidence = 0.0
        self._last_seen = 0.0
        self._pointer_finger = "index"
        self._switch_candidate: str | None = None
        self._switch_frames = 0

    def _choose_pointer_finger(
        self,
        landmarks: list[tuple[float, float, float]],
    ) -> str:
        """Pick the finger that clearly reaches farthest; default to index.

        MediaPipe occasionally swaps finger labels, so we trust geometry over
        the label: the pointing finger is the one whose tip is far from the
        wrist while the rest stay near the palm. If nothing is clearly
        dominant we keep whatever we had (index at rest) so the marker never
        hops onto a curled finger.
        """
        reach = {
            name: _reach(landmarks, chain[0])
            for name, chain in _FINGER_CHAINS.items()
        }
        current = self._pointer_finger
        best = max(reach, key=reach.get)
        others = [v for k, v in reach.items() if k != best]
        runner_up = max(others) if others else 0.0

        dominant = reach[best] >= runner_up * config.HAND_POINTER_DOMINANCE
        if not dominant:
            self._switch_candidate = None
            self._switch_frames = 0
            return current

        if best == current:
            self._switch_candidate = None
            self._switch_frames = 0
            return current

        # Snapping back to index is cheap; leaving it needs sustained evidence.
        needed = 1 if best == "index" else self._pointer_switch_frames
        if best == self._switch_candidate:
            self._switch_frames += 1
        else:
            self._switch_candidate = best
            self._switch_frames = 1

        if self._switch_frames >= needed:
            self._pointer_finger = best
            self._switch_candidate = None
            self._switch_frames = 0
        return self._pointer_finger

    def _coast_velocity(self) -> tuple[float, float]:
        if self._prev_tip is None or self._last_tip is None:
            return 0.0, 0.0
        dt = self._last_seen - self._prev_seen
        if dt <= 1e-4:
            return 0.0, 0.0
        vx = (self._last_tip[0] - self._prev_tip[0]) / dt
        vy = (self._last_tip[1] - self._prev_tip[1]) / dt
        speed = math.hypot(vx, vy)
        if speed > self._max_coast_speed:
            scale = self._max_coast_speed / speed
            vx *= scale
            vy *= scale
        return vx, vy

    def _reset_state(self) -> None:
        self._filter.reset()
        self._anchor.reset()
        self._last_tip = None
        self._prev_tip = None
        self._prev_seen = 0.0
        self._last_landmarks = []
        self._last_confidence = 0.0
        self._pointer_finger = "index"
        self._switch_candidate = None
        self._switch_frames = 0

    def _search_roi(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        """Crop window around the last known hand, grown to allow for motion."""
        if not self._last_landmarks:
            return None
        xs = [p[0] for p in self._last_landmarks]
        ys = [p[1] for p in self._last_landmarks]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        if self._last_tip is not None:
            # Bias toward where the coasting tip has moved.
            vx, vy = self._coast_velocity()
            dt = min(time.perf_counter() - self._last_seen, self._search_s)
            cx += vx * dt * 0.5
            cy += vy * dt * 0.5
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        half = max(size * config.HAND_SEARCH_SCALE, config.HAND_SEARCH_MIN_PX) / 2.0
        x0 = int(max(cx - half, 0))
        y0 = int(max(cy - half, 0))
        x1 = int(min(cx + half, width))
        y1 = int(min(cy + half, height))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return x0, y0, x1, y1

    def _detect_in_roi(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[list[tuple[float, float, float]], float] | None:
        height, width = frame_bgr.shape[:2]
        roi = self._search_roi(width, height)
        if roi is None:
            return None
        x0, y0, x1, y1 = roi
        crop = frame_bgr[y0:y1, x0:x1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._roi_landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        )
        if not result.hand_landmarks:
            return None
        cw, ch = x1 - x0, y1 - y0
        landmarks = [
            (x0 + lm.x * cw, y0 + lm.y * ch, lm.z) for lm in result.hand_landmarks[0]
        ]
        confidence = 0.0
        if result.handedness and result.handedness[0]:
            confidence = float(result.handedness[0][0].score)
        return landmarks, confidence

    def _accept(
        self,
        landmarks: list[tuple[float, float, float]],
        confidence: float,
        now: float,
    ) -> HandResult:
        previous_finger = self._pointer_finger
        finger = self._choose_pointer_finger(landmarks)
        if finger != previous_finger:
            # A relabel means the physical tip barely moved, but the raw
            # coordinate jumps; restart the filters so they don't smear.
            self._filter.reset()
            self._anchor.reset()
        tip_i = _FINGER_CHAINS[finger][0]
        raw = (landmarks[tip_i][0], landmarks[tip_i][1])
        dt = now - self._last_seen if self._last_tip is not None else 0.0
        stable = self._anchor.update(
            _palm_center(landmarks), raw, self._filter.speed, dt
        )
        tip = self._filter.update(stable[0], stable[1], now)

        self._prev_tip = self._last_tip
        self._prev_seen = self._last_seen
        self._last_tip = tip
        self._last_landmarks = landmarks
        self._last_confidence = confidence
        self._last_seen = now

        return {
            "detected": True,
            "index_tip": tip,
            "raw_tip": raw,
            "confidence": confidence,
            "landmarks": landmarks,
            "coasted": False,
            "pointer_finger": finger,
            "palm": is_open_palm(landmarks),
        }

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        """Run detection on a BGR frame. Tip coords are in pixel space."""
        now = time.perf_counter()
        height, width = frame_bgr.shape[:2]

        infer = frame_bgr
        if width > self._infer_max_width:
            scale = self._infer_max_width / width
            infer = cv2.resize(
                frame_bgr,
                (self._infer_max_width, int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((now - self._start) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            landmarks = [(lm.x * width, lm.y * height, lm.z) for lm in hand]
            confidence = 0.0
            if result.handedness and result.handedness[0]:
                confidence = float(result.handedness[0][0].score)
            return self._accept(landmarks, confidence, now)

        # Full-frame detection tends to fail when the hand overlaps the face
        # or hair. Look again in a crop around where the hand just was, where
        # it fills the image and the background is mostly excluded.
        if self._last_landmarks and (now - self._last_seen) <= self._search_s:
            found = self._detect_in_roi(frame_bgr)
            if found is not None:
                landmarks, confidence = found
                return self._accept(landmarks, confidence, now)

        if self._last_tip is not None and (now - self._last_seen) <= self._coast_s:
            vx, vy = self._coast_velocity()
            dt = now - self._last_seen
            # Decay the velocity so a drop right at a direction change doesn't
            # fling the cursor far past where the hand actually reversed.
            tau = config.HAND_COAST_DECAY_MS / 1000.0
            travel = tau * (1.0 - math.exp(-dt / tau))
            tip = (
                float(np.clip(self._last_tip[0] + vx * travel, 0, width - 1)),
                float(np.clip(self._last_tip[1] + vy * travel, 0, height - 1)),
            )
            self._filter.advance(tip[0], tip[1], now)
            return {
                "detected": True,
                "index_tip": tip,
                "raw_tip": None,
                "confidence": self._last_confidence,
                "landmarks": self._last_landmarks,
                "coasted": True,
                "pointer_finger": self._pointer_finger,
                "palm": False,
            }

        if (now - self._last_seen) > max(self._coast_s, self._search_s):
            self._reset_state()
        return {
            "detected": False,
            "index_tip": None,
            "raw_tip": None,
            "confidence": 0.0,
            "landmarks": [],
            "coasted": False,
            "pointer_finger": None,
            "palm": False,
        }

    def close(self) -> None:
        self._landmarker.close()
        self._roi_landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def draw_hand_overlay(frame_bgr: np.ndarray, hand: HandResult) -> None:
    """Draw landmarks, connections, fingertip, and confidence on the frame."""
    if not hand["detected"]:
        return

    height, width = frame_bgr.shape[:2]

    if hand["landmarks"] and not hand["coasted"]:
        normalized = [
            mp.tasks.components.containers.NormalizedLandmark(
                x=x / width, y=y / height, z=z
            )
            for x, y, z in hand["landmarks"]
        ]
        draw_landmarks(
            frame_bgr,
            normalized,
            HandLandmarksConnections.HAND_CONNECTIONS,
        )

    tip = hand["index_tip"]
    if tip is not None:
        cx, cy = int(tip[0]), int(tip[1])
        color = (0, 200, 255) if hand["coasted"] else (0, 255, 255)
        cv2.circle(frame_bgr, (cx, cy), 10, color, 2)
        cv2.circle(frame_bgr, (cx, cy), 4, color, -1)

    label = f"hand {hand['confidence']:.2f}"
    if hand.get("pointer_finger"):
        label += f" {hand['pointer_finger']}"
    if hand["coasted"]:
        label += " coast"
    cv2.putText(
        frame_bgr,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
