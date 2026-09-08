"""Tests for detection helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import source_value


def test_source_value_camera_index():
    assert source_value("0") == 0
    assert source_value("1") == 1


def test_source_value_file_path():
    assert source_value("video.mp4") == "video.mp4"
    assert source_value("/path/to/video.avi") == "/path/to/video.avi"
