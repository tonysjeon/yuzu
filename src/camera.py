"""Webcam capture helpers."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time

import cv2
import numpy as np

from src import config

_SKIP_NAME_PATTERN = re.compile(
    r"iphone|ipad|continuity|desk\s*view",
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


def _builtin_index_via_names(names: list[str]) -> int | None:
    """Prefer FaceTime/MacBook cameras; skip Continuity / iPhone."""
    if not names:
        return None

    for index, name in enumerate(names):
        if _SKIP_NAME_PATTERN.search(name):
            continue
        if _BUILTIN_NAME_PATTERN.search(name):
            return index

    for index, name in enumerate(names):
        if not _SKIP_NAME_PATTERN.search(name):
            return index

    return None


def _builtin_index_via_swift() -> int | None:
    """Return index of the built-in wide camera in an AVFoundation device list."""
    if platform.system() != "Darwin":
        return None

    script = r"""
import AVFoundation
import Foundation

let types: [AVCaptureDevice.DeviceType] = [
    .builtInWideAngleCamera,
    .continuityCamera,
    .external,
    .deskViewCamera
]
let devices = AVCaptureDevice.DiscoverySession(
    deviceTypes: types,
    mediaType: .video,
    position: .unspecified
).devices

guard let builtin = devices.first(where: {
    $0.deviceType == .builtInWideAngleCamera
}) else {
    fputs("NO_BUILTIN\n", stderr)
    exit(2)
}

if let index = devices.firstIndex(where: { $0.uniqueID == builtin.uniqueID }) {
    print(index)
} else {
    fputs("NO_INDEX\n", stderr)
    exit(3)
}
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

    line = result.stdout.strip().splitlines()
    if not line:
        return None
    try:
        return int(line[0].strip())
    except ValueError:
        return None


def resolve_webcam_device_index(names: list[str] | None = None) -> int:
    """Pick the built-in Mac webcam; never prefer Continuity Camera."""
    camera_names = names if names is not None else _camera_names_via_system_profiler()
    by_name = _builtin_index_via_names(camera_names)
    if by_name is not None:
        return by_name

    by_swift = _builtin_index_via_swift()
    if by_swift is not None:
        return by_swift

    return 0


def _open_capture(device_index: int) -> cv2.VideoCapture | None:
    for backend in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        return cap
    return None


def _is_skipped_camera(index: int, names: list[str]) -> bool:
    if index >= len(names):
        return False
    return bool(_SKIP_NAME_PATTERN.search(names[index]))


class Camera:
    """Open the built-in Mac webcam and return mirrored frames."""

    def __init__(
        self,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
    ) -> None:
        names = _camera_names_via_system_profiler()
        preferred = resolve_webcam_device_index(names)
        self.device_index = preferred
        self._cap: cv2.VideoCapture | None = None

        candidates: list[int] = [preferred]
        for index in range(0, max(5, len(names))):
            if index not in candidates:
                candidates.append(index)

        for index in candidates:
            if _is_skipped_camera(index, names):
                continue

            cap = _open_capture(index)
            if cap is None:
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Drop stale buffered frames so the tip tracks the live hand.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._grab_first_frame(cap) is not None:
                self._cap = cap
                self.device_index = index
                break

            cap.release()

        if self._cap is None:
            raise RuntimeError(
                "Unable to open the Mac webcam. Continuity Camera (iPhone) "
                "is ignored on purpose. Grant Camera access under System "
                "Settings → Privacy & Security → Camera, then try again."
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
