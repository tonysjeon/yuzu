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

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

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
    confidence: float
    landmarks: list[tuple[float, float, float]]
    coasted: bool
    pointer_finger: str | None


def _xy(landmarks: list[tuple[float, float, float]], index: int) -> np.ndarray:
    point = landmarks[index]
    return np.array([point[0], point[1]], dtype=np.float64)


def _extension_score(
    landmarks: list[tuple[float, float, float]],
    tip_i: int,
    pip_i: int,
    mcp_i: int,
) -> float:
    """How extended a finger is (higher = more pointing)."""
    wrist = _xy(landmarks, 0)
    tip = _xy(landmarks, tip_i)
    pip = _xy(landmarks, pip_i)
    mcp = _xy(landmarks, mcp_i)
    wrist_tip = float(np.linalg.norm(tip - wrist))
    wrist_pip = float(np.linalg.norm(pip - wrist))
    tip_mcp = float(np.linalg.norm(tip - mcp))
    pip_mcp = float(np.linalg.norm(pip - mcp))
    # Extended fingers stick farther past the PIP / MCP than curled ones.
    return (wrist_tip - wrist_pip) + 0.35 * (tip_mcp - pip_mcp)


def _is_finger_extended(
    landmarks: list[tuple[float, float, float]],
    tip_i: int,
    pip_i: int,
    dip_i: int,
    mcp_i: int,
    extend_ratio: float = config.HAND_EXTEND_RATIO,
    pip_cos_max: float = config.HAND_EXTEND_PIP_COS,
) -> bool:
    """True when a finger looks outstretched, not curled into the palm."""
    tip = _xy(landmarks, tip_i)
    pip = _xy(landmarks, pip_i)
    dip = _xy(landmarks, dip_i)
    mcp = _xy(landmarks, mcp_i)

    tip_mcp = float(np.linalg.norm(tip - mcp))
    pip_mcp = float(np.linalg.norm(pip - mcp))
    if pip_mcp < 1.0 or tip_mcp < pip_mcp * extend_ratio:
        return False

    # Angle at PIP: extended fingers fold open (tip goes away from MCP).
    a = mcp - pip
    b = tip - pip
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1.0 or nb < 1.0:
        return False
    cos_pip = float(np.dot(a, b) / (na * nb))
    if cos_pip > pip_cos_max:
        return False

    # Curled tips collapse near the DIP/MCP cluster.
    tip_dip = float(np.linalg.norm(tip - dip))
    dip_pip = float(np.linalg.norm(dip - pip))
    if tip_dip + dip_pip < pip_mcp * 0.55:
        return False

    return True


def _stabilize_finger_tip(
    landmarks: list[tuple[float, float, float]],
    tip_i: int,
    pip_i: int,
    dip_i: int,
    mcp_i: int,
) -> tuple[float, float]:
    """Blend the raw tip with a bone-axis estimate for one finger."""
    raw = _xy(landmarks, tip_i)
    mcp = _xy(landmarks, mcp_i)
    pip = _xy(landmarks, pip_i)
    dip = _xy(landmarks, dip_i)

    axis = dip - mcp
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-3:
        return float(raw[0]), float(raw[1])

    unit = axis / axis_len
    segment = float(np.linalg.norm(dip - pip))
    if segment < 1.0:
        segment = float(np.linalg.norm(pip - mcp)) * 0.5
    if segment < 1.0:
        segment = axis_len * 0.35

    constrained = dip + unit * segment
    upright = abs(float(unit[1]))
    blend = 0.30 + 0.50 * upright
    tip = (1.0 - blend) * raw + blend * constrained
    return float(tip[0]), float(tip[1])


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
        infer_max_width: int = config.HAND_INFER_MAX_WIDTH,
        tip_smoothing: float = config.HAND_TIP_SMOOTHING,
        max_tip_jump: float = config.HAND_MAX_TIP_JUMP,
        max_coast_speed: float = config.HAND_MAX_COAST_SPEED,
        pointer_lock_frames: int = config.HAND_POINTER_LOCK_FRAMES,
        pointer_switch_margin: float = config.HAND_POINTER_SWITCH_MARGIN,
        index_bias: float = config.HAND_INDEX_BIAS,
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
        self._start = time.perf_counter()
        self._last_timestamp_ms = -1
        self._coast_s = coast_ms / 1000.0
        self._infer_max_width = infer_max_width
        self._tip_smoothing = tip_smoothing
        self._max_tip_jump = max_tip_jump
        self._max_coast_speed = max_coast_speed
        self._pointer_lock_frames = pointer_lock_frames
        self._pointer_switch_margin = pointer_switch_margin
        self._index_bias = index_bias

        self._last_tip: tuple[float, float] | None = None
        self._smooth_tip: tuple[float, float] | None = None
        self._last_landmarks: list[tuple[float, float, float]] = []
        self._last_confidence = 0.0
        self._last_seen = 0.0
        self._velocity = (0.0, 0.0)
        self._pointer_finger = "index"
        self._pointer_candidate = "index"
        self._pointer_candidate_frames = 0
        self._last_pointer_finger: str | None = "index"

    def _choose_pointer_finger(
        self,
        landmarks: list[tuple[float, float, float]],
    ) -> str:
        scores: dict[str, float] = {}
        extended: dict[str, bool] = {}
        for name, (tip_i, pip_i, dip_i, mcp_i) in _FINGER_CHAINS.items():
            scores[name] = _extension_score(landmarks, tip_i, pip_i, mcp_i)
            extended[name] = _is_finger_extended(
                landmarks, tip_i, pip_i, dip_i, mcp_i
            )

        others_extended = [
            name
            for name in ("middle", "ring", "pinky")
            if extended[name]
        ]

        # Classic pointer: index out, others curled → always index.
        if extended["index"] and not others_extended:
            self._pointer_finger = "index"
            self._pointer_candidate = "index"
            self._pointer_candidate_frames = 0
            return "index"

        # Index still out and competing fingers are weak → stay on index.
        if extended["index"]:
            best_other = max(
                (scores[name] for name in ("middle", "ring", "pinky")),
                default=float("-inf"),
            )
            if scores["index"] + self._index_bias >= best_other:
                self._pointer_finger = "index"
                self._pointer_candidate = "index"
                self._pointer_candidate_frames = 0
                return "index"

        # Nothing clearly extended → keep using index tip (game default).
        candidates = [name for name, is_out in extended.items() if is_out]
        if not candidates:
            self._pointer_finger = "index"
            self._pointer_candidate = "index"
            self._pointer_candidate_frames = 0
            return "index"

        desired = max(candidates, key=lambda name: scores[name])
        # Prefer index among extended candidates when close.
        if "index" in candidates and scores["index"] + self._index_bias >= scores[desired]:
            desired = "index"

        locked = self._pointer_finger
        # Drop a lock immediately if that finger curled back in.
        if locked not in candidates:
            self._pointer_finger = desired
            self._pointer_candidate = desired
            self._pointer_candidate_frames = 0
            return desired

        if desired == locked:
            self._pointer_candidate = locked
            self._pointer_candidate_frames = 0
            return locked

        margin = abs(scores[desired]) * self._pointer_switch_margin + 15.0
        if scores[desired] < scores[locked] + margin:
            self._pointer_candidate = locked
            self._pointer_candidate_frames = 0
            return locked

        if desired == self._pointer_candidate:
            self._pointer_candidate_frames += 1
        else:
            self._pointer_candidate = desired
            self._pointer_candidate_frames = 1

        if self._pointer_candidate_frames >= self._pointer_lock_frames:
            self._pointer_finger = desired
            self._pointer_candidate_frames = 0

        return self._pointer_finger

    def _pointer_tip(
        self,
        landmarks: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        finger = self._choose_pointer_finger(landmarks)
        tip_i, pip_i, dip_i, mcp_i = _FINGER_CHAINS[finger]
        return _stabilize_finger_tip(landmarks, tip_i, pip_i, dip_i, mcp_i)

    def _refine_tip(
        self,
        tip: tuple[float, float],
        now: float,
    ) -> tuple[float, float]:
        if self._smooth_tip is None:
            self._smooth_tip = tip
            return tip

        dx = tip[0] - self._smooth_tip[0]
        dy = tip[1] - self._smooth_tip[1]
        dist = math.hypot(dx, dy)
        if dist > self._max_tip_jump and dist > 1e-6:
            scale = self._max_tip_jump / dist
            tip = (
                self._smooth_tip[0] + dx * scale,
                self._smooth_tip[1] + dy * scale,
            )

        alpha = self._tip_smoothing
        # Stronger smoothing on small jitters; stay snappier on real swipes.
        if dist < 25.0:
            alpha = min(alpha, 0.28)
        elif dist > 70.0:
            alpha = max(alpha, 0.55)

        smoothed = (
            alpha * tip[0] + (1.0 - alpha) * self._smooth_tip[0],
            alpha * tip[1] + (1.0 - alpha) * self._smooth_tip[1],
        )
        self._smooth_tip = smoothed

        if self._last_tip is not None and self._last_seen > 0:
            dt = now - self._last_seen
            if dt > 1e-4:
                vx = (smoothed[0] - self._last_tip[0]) / dt
                vy = (smoothed[1] - self._last_tip[1]) / dt
                speed = math.hypot(vx, vy)
                if speed > self._max_coast_speed:
                    scale = self._max_coast_speed / speed
                    vx *= scale
                    vy *= scale
                self._velocity = (vx, vy)

        return smoothed

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
            landmarks = [
                (lm.x * width, lm.y * height, lm.z) for lm in hand
            ]
            tip = self._pointer_tip(landmarks)
            tip = self._refine_tip(tip, now)

            confidence = 0.0
            if result.handedness:
                categories = result.handedness[0]
                if categories:
                    confidence = float(categories[0].score)

            self._last_tip = tip
            self._last_landmarks = landmarks
            self._last_confidence = confidence
            self._last_seen = now
            self._last_pointer_finger = self._pointer_finger

            return {
                "detected": True,
                "index_tip": tip,
                "confidence": confidence,
                "landmarks": landmarks,
                "coasted": False,
                "pointer_finger": self._pointer_finger,
            }

        if (
            self._last_tip is not None
            and (now - self._last_seen) <= self._coast_s
        ):
            dt = now - self._last_seen
            tip = (
                self._last_tip[0] + self._velocity[0] * dt,
                self._last_tip[1] + self._velocity[1] * dt,
            )
            tip = (
                float(np.clip(tip[0], 0, width - 1)),
                float(np.clip(tip[1], 0, height - 1)),
            )
            self._smooth_tip = tip
            return {
                "detected": True,
                "index_tip": tip,
                "confidence": self._last_confidence,
                "landmarks": self._last_landmarks,
                "coasted": True,
                "pointer_finger": self._last_pointer_finger,
            }

        self._last_tip = None
        self._smooth_tip = None
        self._last_landmarks = []
        self._velocity = (0.0, 0.0)
        self._pointer_finger = "index"
        self._pointer_candidate = "index"
        self._pointer_candidate_frames = 0
        self._last_pointer_finger = None
        return {
            "detected": False,
            "index_tip": None,
            "confidence": 0.0,
            "landmarks": [],
            "coasted": False,
            "pointer_finger": None,
        }

    def close(self) -> None:
        self._landmarker.close()

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
        normalized = []
        for x, y, z in hand["landmarks"]:
            normalized.append(
                mp.tasks.components.containers.NormalizedLandmark(
                    x=x / width,
                    y=y / height,
                    z=z,
                )
            )
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
