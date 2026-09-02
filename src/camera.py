"""Webcam capture helpers."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time

import cv2
import numpy as np

from src import config

_SKIP_NAME_PATTERN = re.compile(
    r"iphone|ipad|continuity|desk\s*view|apple\s*continuity",
    re.IGNORECASE,
)
_BUILTIN_NAME_PATTERN = re.compile(
    r"facetime|macbook|built-?in|isight",
    re.IGNORECASE,
)


def _camera_names_via_system_profiler() -> list[str]:
    """Return camera names in system order (usually matches OpenCV indexes)."""
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    cameras = payload.get("SPCameraDataType") or []
    return [str(cam.get("_name", "")).strip() for cam in cameras if cam.get("_name")]


def _builtin_camera_name_via_swift() -> str | None:
    """Return the localized name of the Mac built-in wide camera."""
    if platform.system() != "Darwin":
        return None

    script = r"""
import AVFoundation
import Foundation

let devices = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified
).devices

guard let builtin = devices.first else {
    fputs("NO_BUILTIN\n", stderr)
    exit(2)
}
print(builtin.localizedName)
"""
    try:
        result = subprocess.run(
            ["swift", "-"],
            input=script,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    name = result.stdout.strip().splitlines()
    return name[0].strip() if name else None


def _is_skipped_name(name: str) -> bool:
    return bool(_SKIP_NAME_PATTERN.search(name))


def _is_builtin_name(name: str) -> bool:
    return bool(_BUILTIN_NAME_PATTERN.search(name))


def resolve_webcam_candidates(names: list[str] | None = None) -> list[int]:
    """Ordered OpenCV indexes to try, FaceTime/built-in first, Continuity never."""
    camera_names = names if names is not None else _camera_names_via_system_profiler()
    builtin_name = _builtin_camera_name_via_swift()
    preferred: list[int] = []

    def add(index: int) -> None:
        if index not in preferred:
            preferred.append(index)

    # Match AVFoundation built-in name against system_profiler / OpenCV order.
    if builtin_name and camera_names:
        for index, name in enumerate(camera_names):
            if _is_skipped_name(name):
                continue
            if (
                name == builtin_name
                or builtin_name.lower() in name.lower()
                or name.lower() in builtin_name.lower()
            ):
                add(index)

    if camera_names:
        for index, name in enumerate(camera_names):
            if _is_skipped_name(name):
                continue
            if _is_builtin_name(name):
                add(index)
        for index, name in enumerate(camera_names):
            if not _is_skipped_name(name):
                add(index)
        return preferred

    # No names (permissions / timing). Continuity is usually 0 — try 1 first.
    for index in (1, 0, 2, 3, 4):
        add(index)
    return preferred


def _open_capture(device_index: int) -> cv2.VideoCapture | None:
    for backend in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        return cap
    return None


class Camera:
    """Open the built-in Mac webcam and return mirrored frames."""

    def __init__(
        self,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
    ) -> None:
        names = _camera_names_via_system_profiler()
        candidates = resolve_webcam_candidates(names)
        self.device_index = -1
        self.device_name = "unknown"
        self._cap: cv2.VideoCapture | None = None

        for index in candidates:
            if index < len(names) and _is_skipped_name(names[index]):
                continue

            cap = _open_capture(index)
            if cap is None:
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Drop stale buffered frames so the tip tracks the live hand.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._grab_first_frame(cap) is not None:
                # Final guard: never keep a Continuity / iPhone device by name.
                label = names[index] if index < len(names) else f"index {index}"
                if _is_skipped_name(label):
                    cap.release()
                    continue

                self._cap = cap
                self.device_index = index
                self.device_name = label
                break

            cap.release()

        if self._cap is None:
            raise RuntimeError(
                "Unable to open the Mac FaceTime webcam. Continuity Camera "
                "(iPhone) is ignored on purpose. Grant Camera access under "
                "System Settings → Privacy & Security → Camera, put the "
                "iPhone away/lock Continuity Camera if needed, then try again."
            )

        print(
            f"Using camera {self.device_index}: {self.device_name}",
            file=sys.stderr,
        )

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._width = actual_width if actual_width > 0 else width
        self._height = actual_height if actual_height > 0 else height

    @staticmethod
    def _grab_first_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.05)
        return None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def read(self) -> np.ndarray | None:
        """Return the next mirrored frame, or None if capture failed."""
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.flip(frame, 1)

    def release(self) -> None:
        """Release the webcam device."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None
