"""Shared utility functions for source handling."""
import cv2


def source_value(value):
    """Convert source string to int (camera index) or keep as file path."""
    try:
        return int(value)
    except ValueError:
        return value
