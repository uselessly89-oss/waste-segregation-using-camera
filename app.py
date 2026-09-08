"""Waste Segregation Camera App.

Easy camera-based waste detection using a laptop webcam, phone camera
(same Wi-Fi), or uploaded photos/videos. Uses YOLO for real-time
segmentation and detection with cluster-aware reporting.

Usage:
    pip install -r requirements-app.txt
    python app.py

Then open the displayed URL. On a phone, use http://<LAPTOP-IP>:7860.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.getenv("WASTE_MODEL", "models/best.pt")
model = None


def get_model():
    """Lazy-load the YOLO model on first inference call."""
    global model
    if model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. "
                "Train a model first or set WASTE_MODEL env var."
            )
        model = YOLO(MODEL_PATH)
    return model


def annotate(image, conf=0.45, iou=0.50, cluster_mode=True):
    """Run inference on a single frame and return annotated image + decision text.

    Parameters
    ----------
    image : numpy.ndarray or PIL Image
        Input frame from webcam / upload.
    conf : float
        Confidence threshold for detections.
    iou : float
        IoU threshold for NMS.
    cluster_mode : bool
        When True, reports the full cluster breakdown and routes uncertain
        items to REJECT instead of guessing.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Annotated RGB image and human-readable decision string.
    """
    if image is None:
        return None, "No image received."

    m = get_model()
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    results = m.predict(
        frame, conf=float(conf), iou=float(iou), imgsz=640, verbose=False
    )
    r = results[0]

    counts = {}
    total = 0
    if r.boxes is not None:
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            cls_id = int(r.boxes.cls[i].item())
            score = float(r.boxes.conf[i].item())
            name = str(m.names[cls_id])
            counts[name] = counts.get(name, 0) + 1
            total += 1

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} {score:.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    if cluster_mode and total == 0:
        message = "NO CONFIDENT OBJECTS — send this item/cluster to REJECT."
    elif cluster_mode:
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        message = (
            f"CLUSTER: {total} visible object(s) | {parts} | "
            f"Unknown/hidden material -> REJECT"
        )
    else:
        message = f"Detected {total} object(s)."

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), message


def run():
    """Launch the Gradio interface."""
    iface = gr.Interface(
        fn=annotate,
        inputs=[
            gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Waste camera / photo",
            ),
            gr.Slider(
                0.20, 0.90, value=0.45, step=0.05, label="Confidence threshold"
            ),
            gr.Slider(
                0.20, 0.90, value=0.50, step=0.05, label="IoU threshold"
            ),
            gr.Checkbox(value=True, label="Cluster mode"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Segregation view"),
            gr.Textbox(label="Decision"),
        ],
        title="\u267b\ufe0f Waste Segregation Camera",
        description=(
            "Use a laptop webcam or phone browser camera. Visible items are "
            "segmented/detected; uncertain or hidden material is routed to "
            "REJECT rather than guessed."
        ),
    )
    return iface


if __name__ == "__main__":
    run().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
