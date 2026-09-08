"""Tests for video analysis and utility helpers."""
import sys
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_test_video(path, frames=10, fps=24, width=320, height=240):
    """Create a small synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        val = 80 + (i * 15) % 100
        for y in range(0, height, 20):
            for x in range(0, width, 20):
                c = val if (x // 20 + y // 20) % 2 == 0 else min(255, val + 40)
                frame[y:y+20, x:x+20] = c
        writer.write(frame)
    writer.release()
    return path


def test_annotate_frame_importable():
    """Verify annotate_frame can be imported and is callable."""
    from utils import annotate_frame
    assert callable(annotate_frame)


def test_make_test_video():
    """Verify we can create and read a synthetic video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        _make_test_video(path, frames=5)
        cap = cv2.VideoCapture(path)
        assert cap.isOpened()
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 5
    finally:
        os.unlink(path)


def test_make_test_video_frame_dimensions():
    """Verify the synthetic video has correct dimensions."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        _make_test_video(path, frames=3, width=640, height=480)
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert w == 640
        assert h == 480
    finally:
        os.unlink(path)
