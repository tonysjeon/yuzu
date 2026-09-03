"""Webcam capture helpers.

On macOS the built-in FaceTime camera is opened through AVFoundation's
built-in wide-angle device type only. Continuity Camera / iPhone is never
enumerated, so the phone is never asked to connect.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from src import config

_SKIP_NAME_PATTERN = re.compile(
    r"iphone|ipad|ios\s*camera|continuity|desk\s*view|"
    r"apple\s*continuity|sidecar|center\s*stage|camo|"
    r"continuity\s*camera",
    re.IGNORECASE,
)
_BUILTIN_NAME_PATTERN = re.compile(
    r"facetime|macbook|built-?in|isight",
    re.IGNORECASE,
)
_BUILTIN_TYPE_PATTERN = re.compile(r"builtin", re.IGNORECASE)
_SKIP_TYPE_PATTERN = re.compile(
    r"continuity|deskview|desk.?view|external",
    re.IGNORECASE,
)

_HELPER_SRC = Path(__file__).resolve().parent / "facetime_cam.swift"
_HELPER_BIN = Path(__file__).resolve().parent.parent / ".cache" / "facetime_cam"


class AVDevice(TypedDict):
    index: int
    name: str
    type: str
    unique_id: str
    builtin: bool


def _is_skipped_name(name: str) -> bool:
    return bool(_SKIP_NAME_PATTERN.search(name))


def _is_builtin_name(name: str) -> bool:
    return bool(_BUILTIN_NAME_PATTERN.search(name))


def _is_phone_or_continuity(device: AVDevice) -> bool:
    if device["builtin"]:
        return False
    if _SKIP_TYPE_PATTERN.search(device["type"]):
        return True
    if _is_skipped_name(device["name"]):
        return True
    unique = device["unique_id"].lower()
    if "continuity" in unique or "iphone" in unique:
        return True
    return False


def _is_facetime_device(device: AVDevice) -> bool:
    if _is_phone_or_continuity(device):
        return False
    if device["builtin"] or _BUILTIN_TYPE_PATTERN.search(device["type"]):
        return True
    return _is_builtin_name(device["name"])


def resolve_webcam_candidates(
    names: list[str] | None = None,
    av_devices: list[AVDevice] | None = None,
    system: str | None = None,
) -> list[int]:
    """Ordered OpenCV indexes to try.

    macOS: FaceTime only — never Continuity/iPhone (index 0 is usually the phone).
    Windows/Linux: built-in laptop webcam is almost always index 0.
    """
    devices = list(av_devices or [])
    camera_names = list(names or [])
    preferred: list[int] = []
    host = system if system is not None else platform.system()

    def add(index: int) -> None:
        if index not in preferred:
            preferred.append(index)

    if devices:
        for device in devices:
            if _is_facetime_device(device):
                add(device["index"])
        return preferred

    if camera_names:
        for index, name in enumerate(camera_names):
            if _is_skipped_name(name):
                continue
            if _is_builtin_name(name):
                add(index)
        return preferred

    if host == "Darwin":
        # Continuity is usually 0 — never try it when we cannot read names.
        for index in (1, 2, 3, 4):
            add(index)
        return preferred

    for index in (0, 1, 2, 3, 4):
        add(index)
    return preferred


def _ensure_facetime_helper() -> Path:
    if not _HELPER_SRC.is_file():
        raise RuntimeError(f"FaceTime capture helper missing at {_HELPER_SRC}")
    _HELPER_BIN.parent.mkdir(parents=True, exist_ok=True)
    src_mtime = _HELPER_SRC.stat().st_mtime
    if _HELPER_BIN.is_file() and _HELPER_BIN.stat().st_mtime >= src_mtime:
        return _HELPER_BIN
    result = subprocess.run(
        [
            "swiftc",
            "-O",
            "-o",
            str(_HELPER_BIN),
            str(_HELPER_SRC),
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
            "-framework",
            "Foundation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not _HELPER_BIN.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Could not build the built-in FaceTime capture helper. "
            f"{detail or 'swiftc failed.'}"
        )
    return _HELPER_BIN


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        piece = stream.read(size - len(chunks))
        if not piece:
            return b""
        chunks.extend(piece)
    return bytes(chunks)


class _FaceTimeCapture:
    """macOS built-in wide-angle camera; never probes Continuity / iPhone."""

    def __init__(self, width: int, height: int) -> None:
        helper = _ensure_facetime_helper()
        self._proc = subprocess.Popen(
            [str(helper), str(width), str(height)],
            stdout=subprocess.PIPE,
            stderr=None,
        )
        stdout = self._proc.stdout
        if stdout is None:
            self.release()
            raise RuntimeError("Built-in camera helper has no stdout.")
        header_line = stdout.readline()
        if not header_line:
            code = self._proc.poll()
            self.release()
            raise RuntimeError(
                "Unable to open the Mac FaceTime webcam. Continuity Camera "
                "(iPhone) is never used. Grant Camera access under "
                "System Settings → Privacy & Security → Camera, then try again."
                + (f" Helper exit {code}." if code else "")
            )
        try:
            meta = json.loads(header_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.release()
            raise RuntimeError(f"Built-in camera helper sent a bad header: {exc}") from exc
        self.name = str(meta.get("name") or "MacBook Camera")
        self.width = int(meta.get("width") or width)
        self.height = int(meta.get("height") or height)
        self._stdout = stdout

    def read(self) -> np.ndarray | None:
        header = _read_exact(self._stdout, 4)
        if len(header) < 4:
            return None
        size = int.from_bytes(header, "big")
        if size <= 0 or size > 32_000_000:
            return None
        payload = _read_exact(self._stdout, size)
        if len(payload) < size:
            return None
        bgra = np.frombuffer(payload, dtype=np.uint8)
        try:
            bgra = bgra.reshape((self.height, self.width, 4))
        except ValueError:
            return None
        return cv2.flip(np.ascontiguousarray(bgra[:, :, :3]), 1)

    def release(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        self._proc = None


def _opencv_backends() -> tuple[int, ...]:
    host = platform.system()
    if host == "Windows":
        backends: list[int] = []
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)
        return tuple(backends)
    if host == "Darwin":
        return (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY)
    if hasattr(cv2, "CAP_V4L2"):
        return (cv2.CAP_V4L2, cv2.CAP_ANY)
    return (cv2.CAP_ANY,)


def _open_capture(device_index: int) -> cv2.VideoCapture | None:
    for backend in _opencv_backends():
        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        return cap
    return None


class Camera:
    """Open the built-in webcam and return mirrored frames."""

    def __init__(
        self,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
    ) -> None:
        self.device_index = -1
        self.device_name = "unknown"
        self._cap: cv2.VideoCapture | None = None
        self._facetime: _FaceTimeCapture | None = None

        if platform.system() == "Darwin":
            self._facetime = _FaceTimeCapture(width, height)
            self.device_name = self._facetime.name
            if _is_skipped_name(self.device_name):
                self._facetime.release()
                self._facetime = None
                raise RuntimeError(
                    "Refusing to use a Continuity / iPhone camera. "
                    "The built-in FaceTime camera was not available."
                )
            self._width = self._facetime.width
            self._height = self._facetime.height
            print(
                f"Using built-in FaceTime camera: {self.device_name}",
                file=sys.stderr,
            )
            return

        self._open_opencv(width, height)

    def _open_opencv(self, width: int, height: int) -> None:
        candidates = resolve_webcam_candidates(names=[], av_devices=[])
        for index in candidates:
            cap = _open_capture(index)
            if cap is None:
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self._grab_first_frame(cap) is None:
                cap.release()
                continue
            self._cap = cap
            self.device_index = index
            self.device_name = f"index {index}"
            break

        if self._cap is None:
            if platform.system() == "Windows":
                raise RuntimeError(
                    "Unable to open the laptop webcam. Allow Camera access "
                    "under Settings → Privacy & security → Camera for this "
                    "app (Terminal, Python, or Cursor), then try again."
                )
            raise RuntimeError(
                "Unable to open a webcam. Continuity Camera (iPhone) is never used."
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
        if self._facetime is not None:
            return self._facetime.read()
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.flip(frame, 1)

    def release(self) -> None:
        """Release the webcam device."""
        if self._facetime is not None:
            self._facetime.release()
            self._facetime = None
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None
