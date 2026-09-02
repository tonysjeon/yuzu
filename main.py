"""Run the Yuzu webcam preview with hand tracking."""

from __future__ import annotations

import sys

import cv2
import numpy as np

from src.camera import Camera
from src.hand_tracker import HandTracker, draw_hand_overlay


def _window_size(window_name: str) -> tuple[int, int] | None:
    try:
        rect = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return None
    # rect = (x, y, width, height)
    width, height = int(rect[2]), int(rect[3])
    if width <= 1 or height <= 1:
        return None
    return width, height


def _fit_to_window(frame: np.ndarray, window_name: str) -> np.ndarray:
    """Scale the frame to fill the current window without letterboxing."""
    size = _window_size(window_name)
    if size is None:
        return frame
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def main() -> int:
    try:
        camera = Camera()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        tracker = HandTracker()
    except FileNotFoundError as exc:
        camera.release()
        print(exc, file=sys.stderr)
        return 1

    window_name = "yuzu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, camera.width, camera.height)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print(
                    "Failed to read from webcam. Check that nothing else is "
                    "using the camera and that Camera permission is enabled.",
                    file=sys.stderr,
                )
                return 1

            hand = tracker.process(frame)
            draw_hand_overlay(frame, hand)
            display = _fit_to_window(frame, window_name)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
