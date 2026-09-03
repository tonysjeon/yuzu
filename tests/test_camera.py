"""Tests for webcam device selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2

from src.camera import _is_skipped_name, _opencv_backends, resolve_webcam_candidates


class WebcamCandidateTests(unittest.TestCase):
    def test_mac_nameless_fallback_skips_index_zero(self) -> None:
        self.assertEqual(
            resolve_webcam_candidates(names=[], av_devices=[], system="Darwin"),
            [1, 2, 3, 4],
        )

    def test_windows_tries_built_in_webcam_first(self) -> None:
        self.assertEqual(
            resolve_webcam_candidates(names=[], av_devices=[], system="Windows"),
            [0, 1, 2, 3, 4],
        )

    def test_skips_iphone_camera_name(self) -> None:
        self.assertTrue(_is_skipped_name("Tony's iPhone Camera"))
        self.assertTrue(_is_skipped_name("Tony’s iPhone Camera"))
        self.assertFalse(_is_skipped_name("MacBook Air Camera"))

    def test_windows_prefers_msmf_then_dshow(self) -> None:
        with patch("src.camera.platform.system", return_value="Windows"):
            backends = _opencv_backends()
        self.assertEqual(backends[0], cv2.CAP_MSMF)
        self.assertIn(cv2.CAP_DSHOW, backends)


if __name__ == "__main__":
    unittest.main()
