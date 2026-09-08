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
from src.utils import annotate_frame as _annotate_frame

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
    annotated, counts = _annotate_frame(m, frame, float(conf), float(iou), cluster_mode)
    total = sum(counts.values())

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

    return annotated, message


def analyze_video(video_path, conf=0.45, iou=0.50, frame_skip=2):
    """Process a video frame-by-frame, return annotated video + summary.

    Parameters
    ----------
    video_path : str
        Path to uploaded video file.
    conf, iou : float
        Detection thresholds.
    frame_skip : int
        Analyze every Nth frame (1 = every frame, 2 = every other, etc.).

    Returns
    -------
    tuple[str, str]
        Path to annotated output video, summary text.
    """
    if video_path is None:
        return None, "No video uploaded."

    m = get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Cannot open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = str(Path(video_path).parent / "annotated_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Accumulate class counts across all processed frames
    global_counts = {}
    processed = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip == 0:
            annotated_rgb, counts = _annotate_frame(
                m, frame.copy(), conf, iou, cluster_mode=True,
            )
            for cls, cnt in counts.items():
                global_counts[cls] = global_counts.get(cls, 0) + cnt
            processed += 1
            # Write annotated frame (convert RGB back to BGR for cv2)
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        else:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # Build summary
    total_det = sum(global_counts.values())
    lines = [f"Video: {total_frames} frames total, {processed} analyzed (every {frame_skip} frames)"]
    if total_det > 0:
        lines.append(f"Total detections across video: {total_det}")
        for cls, cnt in sorted(global_counts.items()):
            lines.append(f"  {cls}: {cnt}")
    else:
        lines.append("No objects detected in any analyzed frame.")
    lines.append(f"\nAnnotated video saved to: {out_path}")

    return out_path, "\n".join(lines)


CUSTOM_CSS = """
/* ---- Global ---- */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* ---- Header / Title ---- */
#main-title {
    text-align: center;
    font-size: 2rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}
#main-desc {
    text-align: center;
    color: #555;
    font-size: 0.95rem;
    margin-top: 0.25rem !important;
}

/* ---- Sidebar panel (controls) ---- */
#controls {
    background: #f0fdf4;
    border: 1px solid #c8e6c9;
    border-radius: 12px;
    padding: 1.2rem;
}
#controls label {
    font-weight: 600;
    color: #2e7d32;
}

/* ---- Output panel ---- */
#output-panel {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1rem;
}

/* ---- Decision textbox ---- */
#decision-box textarea {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1b5e20;
    background: #e8f5e9;
    border-radius: 8px;
}

/* ---- Footer ---- */
#footer {
    text-align: center;
    color: #999;
    font-size: 0.8rem;
    margin-top: 1.5rem;
}
"""


def run():
    """Launch the Gradio Blocks interface with custom theme."""
    with gr.Blocks(css=CUSTOM_CSS, title="Waste Segregation Camera") as demo:
        # -- Header --
        gr.Markdown(
            "# \u267b\ufe0f Waste Segregation Camera",
            elem_id="main-title",
        )
        gr.Markdown(
            "Point a laptop webcam, phone camera, or upload a photo. "
            "Visible items are detected; uncertain material goes to REJECT.",
            elem_id="main-desc",
        )

        with gr.Tabs():
            # ===== Tab 1: Camera / Photo =====
            with gr.TabItem("\ud83d\udcf7  Camera / Photo"):
                with gr.Row():
                    with gr.Column(scale=1, elem_id="controls"):
                        gr.Markdown("### \u2699\ufe0f Settings")
                        camera_input = gr.Image(
                            sources=["webcam", "upload"],
                            type="numpy",
                            label="Waste camera / photo",
                        )
                        conf_slider = gr.Slider(
                            0.20, 0.90, value=0.45, step=0.05,
                            label="Confidence threshold",
                        )
                        iou_slider = gr.Slider(
                            0.20, 0.90, value=0.50, step=0.05,
                            label="IoU threshold",
                        )
                        cluster_cb = gr.Checkbox(value=True, label="Cluster mode")
                        run_btn = gr.Button(
                            "\u25b6  Detect", variant="primary", size="lg",
                        )

                    with gr.Column(scale=2, elem_id="output-panel"):
                        gr.Markdown("### \ud83d\udcca Results")
                        output_image = gr.Image(type="numpy", label="Segregation view")
                        decision_box = gr.Textbox(
                            label="Decision",
                            interactive=False,
                            elem_id="decision-box",
                        )

            # ===== Tab 2: Video Analysis =====
            with gr.TabItem("\ud83c\udfac  Video Analysis"):
                gr.Markdown(
                    "Upload a video file to analyze it frame-by-frame. "
                    "The app runs YOLO on sampled frames, draws detections, "
                    "and returns an annotated video + full detection summary."
                )
                with gr.Row():
                    with gr.Column(scale=1, elem_id="controls"):
                        gr.Markdown("### \u2699\ufe0f Settings")
                        video_input = gr.Video(label="Upload video")
                        v_conf = gr.Slider(
                            0.20, 0.90, value=0.45, step=0.05,
                            label="Confidence threshold",
                        )
                        v_iou = gr.Slider(
                            0.20, 0.90, value=0.50, step=0.05,
                            label="IoU threshold",
                        )
                        frame_skip_slider = gr.Slider(
                            1, 10, value=2, step=1,
                            label="Analyze every Nth frame",
                        )
                        video_btn = gr.Button(
                            "\u25b6  Analyze Video", variant="primary", size="lg",
                        )

                    with gr.Column(scale=2, elem_id="output-panel"):
                        gr.Markdown("### \ud83d\udcca Results")
                        video_output = gr.Video(label="Annotated video")
                        video_summary = gr.Textbox(
                            label="Detection summary",
                            interactive=False,
                            lines=8,
                            elem_id="decision-box",
                        )

        # -- Footer --
        gr.Markdown(
            "---\n\u267b\ufe0f *Waste Segregation Using Camera* \u2014 "
            "YOLO-powered real-time waste detection",
            elem_id="footer",
        )

        # Wire up buttons
        run_btn.click(
            fn=annotate,
            inputs=[camera_input, conf_slider, iou_slider, cluster_cb],
            outputs=[output_image, decision_box],
        )
        video_btn.click(
            fn=analyze_video,
            inputs=[video_input, v_conf, v_iou, frame_skip_slider],
            outputs=[video_output, video_summary],
        )

    return demo


if __name__ == "__main__":
    run().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
