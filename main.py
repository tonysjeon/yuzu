"""Run the Yuzu webcam preview."""

from __future__ import annotations

import sys

import cv2

from src.camera import Camera


def main() -> int:
    try:
        camera = Camera()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    window_name = "yuzu"
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

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
