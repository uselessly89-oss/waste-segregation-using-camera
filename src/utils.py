"""Shared utility functions for source handling and annotation."""
import cv2
import numpy as np


def source_value(value):
    """Convert source string to int (camera index) or keep as file path."""
    try:
        return int(value)
    except ValueError:
        return value


def annotate_frame(model, frame, conf, iou, cluster_mode=True):
    """Run YOLO on a single BGR frame, draw boxes, return (annotated_rgb, counts).

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded YOLO model.
    frame : numpy.ndarray
        BGR image.
    conf : float
        Confidence threshold.
    iou : float
        IoU threshold.
    cluster_mode : bool
        If True, always report counts (unused here but kept for API compat).

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Annotated RGB image and class counts dict.
    """
    results = model.predict(frame, conf=conf, iou=iou, imgsz=640, verbose=False)
    r = results[0]
    counts = {}
    if r.boxes is not None:
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            cls_id = int(r.boxes.cls[i].item())
            score = float(r.boxes.conf[i].item())
            name = str(model.names[cls_id])
            counts[name] = counts.get(name, 0) + 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{name} {score:.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), counts
