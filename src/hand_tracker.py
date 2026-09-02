"""Hand landmark detection via MediaPipe."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypedDict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

from src import config

INDEX_TIP = 8

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


class HandTracker:
    """Detect one hand and expose the index fingertip."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        max_hands: int = 1,
        min_detection_confidence: float = config.HAND_MIN_DETECTION,
        min_presence_confidence: float = config.HAND_MIN_PRESENCE,
        min_tracking_confidence: float = config.HAND_MIN_TRACKING,
        coast_ms: float = config.HAND_COAST_MS,
        infer_max_width: int = config.HAND_INFER_MAX_WIDTH,
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

        self._last_tip: tuple[float, float] | None = None
        self._last_landmarks: list[tuple[float, float, float]] = []
        self._last_confidence = 0.0
        self._last_seen = 0.0
        self._velocity = (0.0, 0.0)

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        """Run detection on a BGR frame. Tip coords are in pixel space."""
        now = time.perf_counter()
        height, width = frame_bgr.shape[:2]

        infer = frame_bgr
        scale = 1.0
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
            # Normalized coords map straight back to the full frame
            # because inference resize keeps aspect ratio.
            landmarks = [
                (lm.x * width, lm.y * height, lm.z) for lm in hand
            ]
            tip = (landmarks[INDEX_TIP][0], landmarks[INDEX_TIP][1])
            confidence = 0.0
            if result.handedness:
                categories = result.handedness[0]
                if categories:
                    confidence = float(categories[0].score)

            if self._last_tip is not None and self._last_seen > 0:
                dt = now - self._last_seen
                if dt > 1e-4:
                    self._velocity = (
                        (tip[0] - self._last_tip[0]) / dt,
                        (tip[1] - self._last_tip[1]) / dt,
                    )

            self._last_tip = tip
            self._last_landmarks = landmarks
            self._last_confidence = confidence
            self._last_seen = now

            return {
                "detected": True,
                "index_tip": tip,
                "confidence": confidence,
                "landmarks": landmarks,
                "coasted": False,
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
            return {
                "detected": True,
                "index_tip": tip,
                "confidence": self._last_confidence,
                "landmarks": self._last_landmarks,
                "coasted": True,
            }

        self._last_tip = None
        self._last_landmarks = []
        self._velocity = (0.0, 0.0)
        return {
            "detected": False,
            "index_tip": None,
            "confidence": 0.0,
            "landmarks": [],
            "coasted": False,
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
