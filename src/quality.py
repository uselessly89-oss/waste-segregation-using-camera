"""Image quality assessment gate.

Rejects frames that are too dark, too bright, or too blurry for reliable
detection. This prevents the model from making low-confidence guesses on
unusable input.
"""
import cv2


def assess_frame(frame, min_brightness=25, max_brightness=235, min_blur_variance=30):
    """Evaluate frame quality for waste detection.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image from OpenCV.
    min_brightness : float
        Minimum acceptable mean brightness (0-255).
    max_brightness : float
        Maximum acceptable mean brightness (0-255).
    min_blur_variance : float
        Minimum Laplacian variance (sharpness measure).

    Returns
    -------
    dict
        Keys: ok (bool), brightness (float), sharpness (float), reason (str).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_ok = min_brightness <= brightness <= max_brightness
    sharp_ok = sharpness >= min_blur_variance

    return {
        "ok": brightness_ok and sharp_ok,
        "brightness": brightness,
        "sharpness": sharpness,
        "reason": "ok" if brightness_ok and sharp_ok else "poor_frame_quality",
    }
