"""Tests for the image quality assessment gate."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality import assess_frame


def _make_frame(brightness=128, blur=0):
    """Create a synthetic BGR frame with controlled brightness and blur."""
    # Start with a sharp checkerboard pattern (high Laplacian variance)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(100):
        for x in range(100):
            val = brightness if (x // 10 + y // 10) % 2 == 0 else min(255, brightness + 60)
            frame[y, x] = val
    if blur > 1:
        ksize = int(blur) | 1  # ensure odd kernel size
        frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return frame


def test_good_frame():
    frame = _make_frame(brightness=128, blur=5)
    result = assess_frame(frame)
    assert result["ok"], f"Expected ok, got {result}"


def test_dark_frame_rejected():
    frame = np.full((100, 100, 3), 5, dtype=np.uint8)  # flat dark frame
    result = assess_frame(frame)
    assert not result["ok"]
    assert result["reason"] == "poor_frame_quality"


def test_bright_frame_rejected():
    frame = np.full((100, 100, 3), 250, dtype=np.uint8)  # flat bright frame
    result = assess_frame(frame)
    assert not result["ok"]


def test_blurry_frame_rejected():
    frame = _make_frame(brightness=128, blur=51)  # large blur kernel kills all texture
    result = assess_frame(frame)
    assert not result["ok"]
    assert result["reason"] == "poor_frame_quality"
