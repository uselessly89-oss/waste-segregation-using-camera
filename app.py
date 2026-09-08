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

import csv
import tempfile

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
    """Run inference on a single frame and return annotated image + decision + CSV.

    Returns
    -------
    tuple[numpy.ndarray, str, str]
        Annotated RGB image, decision text, and path to detections CSV.
    """
    if image is None:
        return None, "No image received.", None

    m = get_model()
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    annotated, counts, detections = _annotate_frame(m, frame, float(conf), float(iou), cluster_mode)
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

    # Write detections to CSV
    csv_path = str(Path(tempfile.mktemp(suffix="_detections.csv")))
    csv_fields = ["frame", "timestamp_s", "class", "confidence",
                   "x1", "y1", "x2", "y2", "cx", "cy"]
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer_csv.writeheader()
        for det in detections:
            det["frame"] = 0
            det["timestamp_s"] = 0.0
            writer_csv.writerow(det)

    return annotated, message, csv_path


def _build_heatmap(detections, width, height, class_counts, class_filter=None):
    """Build a detection heatmap overlay from a list of detection dicts.

    Parameters
    ----------
    detections : list[dict]
        Each dict must have 'cx', 'cy', 'class' keys.
    width, height : int
        Frame dimensions.
    class_counts : dict
        Per-class detection counts for the legend.
    class_filter : str or None
        If set, only include detections matching this class name.

    Returns
    -------
    numpy.ndarray
        RGB heatmap image.
    """
    # Filter by class if requested
    filtered = detections
    if class_filter:
        filtered = [d for d in detections if d["class"] == class_filter]

    # Accumulate detection points on a blank canvas
    canvas = np.zeros((height, width), dtype=np.float32)
    for det in filtered:
        cx = int(det["cx"])
        cy = int(det["cy"])
        if 0 <= cx < width and 0 <= cy < height:
            canvas[cy, cx] += 1.0

    # Smooth with a large Gaussian to create density blobs
    ksize = max(31, min(width, height) // 8) | 1
    heatmap = cv2.GaussianBlur(canvas, (ksize, ksize), 0)

    # Normalize to 0-255
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = (heatmap / hmax * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    # Apply colormap (JET: blue=low, red=high)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Create base frame (dark gray) and blend
    base = np.full((height, width, 3), 40, dtype=np.uint8)
    mask = heatmap > 0
    blended = base.copy()
    blended[mask] = cv2.addWeighted(base, 0.4, heatmap_rgb, 0.6, 0)[mask]

    # Build title
    det_count = len(filtered)
    if class_filter:
        title = f"{class_filter} heatmap ({det_count} detections)"
    else:
        title = f"All classes heatmap ({det_count} detections)"

    # Draw legend
    y_offset = 25
    cv2.putText(
        blended, title,
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    y_offset += 30
    legend_colors = [
        (0, 255, 0), (255, 255, 0), (255, 128, 0),
        (0, 200, 255), (255, 0, 255), (128, 255, 0),
    ]
    # For per-class heatmaps, show just that class; for all, show legend
    if class_filter:
        cv2.circle(blended, (20, y_offset - 5), 6, (255, 255, 255), -1)
        cv2.putText(
            blended, f"{class_filter}: {det_count}", (35, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    else:
        for i, (cls, cnt) in enumerate(sorted(class_counts.items())):
            color = legend_colors[i % len(legend_colors)]
            cv2.circle(blended, (20, y_offset - 5), 6, color, -1)
            cv2.putText(
                blended, f"{cls}: {cnt}", (35, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            y_offset += 22

    return blended


def _build_timeline(detections, total_duration_s, class_counts):
    """Build a per-second detection timeline chart.

    Returns an RGB numpy image showing a bar chart of detections per second,
    with stacked colors for each waste class.

    Parameters
    ----------
    detections : list[dict]
        Detection dicts with 'timestamp_s' and 'class' keys.
    total_duration_s : float
        Video duration in seconds.
    class_counts : dict
        Per-class counts (used for legend colors).

    Returns
    -------
    numpy.ndarray
        RGB chart image.
    """
    import math

    # Bin detections by second
    n_bins = max(1, int(math.ceil(total_duration_s)))
    bins = np.zeros((n_bins, len(class_counts)), dtype=np.int32)
    cls_names = sorted(class_counts.keys())
    cls_idx = {name: i for i, name in enumerate(cls_names)}

    for det in detections:
        sec = min(int(det["timestamp_s"]), n_bins - 1)
        name = det["class"]
        if name in cls_idx:
            bins[sec, cls_idx[name]] += 1

    # Chart dimensions
    chart_w, chart_h = 800, 300
    margin_left, margin_bottom, margin_top, margin_right = 70, 40, 40, 20
    plot_w = chart_w - margin_left - margin_right
    plot_h = chart_h - margin_top - margin_bottom

    # Canvas
    img = np.full((chart_h, chart_w, 3), 255, dtype=np.uint8)

    # Find max value for scaling
    row_totals = bins.sum(axis=1)
    y_max = max(int(row_totals.max()), 1)
    # Round up to nice number
    y_max = int(math.ceil(y_max / max(1, int(y_max ** 0.5))) * max(1, int(y_max ** 0.5)))
    if y_max < 5:
        y_max = 5

    # Class colors (BGR for OpenCV)
    cls_colors = [
        (0, 200, 0),    # green
        (0, 220, 255),  # orange
        (0, 128, 255),  # blue-ish
        (255, 128, 0),  # cyan-ish
        (128, 0, 255),  # magenta
        (0, 255, 200),  # teal
    ]

    # Draw axes
    ax_left = margin_left
    ax_right = chart_w - margin_right
    ax_top = margin_top
    ax_bottom = chart_h - margin_bottom
    cv2.line(img, (ax_left, ax_top), (ax_left, ax_bottom), (80, 80, 80), 2)
    cv2.line(img, (ax_left, ax_bottom), (ax_right, ax_bottom), (80, 80, 80), 2)

    # Y-axis grid lines and labels
    n_grid = min(y_max, 5)
    for i in range(n_grid + 1):
        y_val = int(y_max * i / n_grid)
        y_px = ax_bottom - int(plot_h * i / n_grid)
        cv2.line(img, (ax_left, y_px), (ax_right, y_px), (200, 200, 200), 1)
        cv2.putText(img, str(y_val), (5, y_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # Draw stacked bars
    bar_w = max(1, plot_w // n_bins - 1)
    for sec in range(n_bins):
        x_left = ax_left + int(plot_w * sec / n_bins)
        cumulative = 0
        for ci, cls_name in enumerate(cls_names):
            val = bins[sec, ci]
            if val == 0:
                continue
            h = int(plot_h * val / y_max)
            y_top = ax_bottom - cumulative - h
            y_bot = ax_bottom - cumulative
            color = cls_colors[ci % len(cls_colors)]
            cv2.rectangle(img, (x_left, y_top), (x_left + bar_w, y_bot), color, -1)
            cumulative += h

        # X-axis label (every N seconds)
        if n_bins <= 30 or sec % max(1, n_bins // 10) == 0:
            cv2.putText(
                img, f"{sec}s", (x_left, ax_bottom + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1,
            )

    # Title
    cv2.putText(
        img, "Detections per second", (margin_left, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1,
    )

    # Legend
    legend_x = chart_w - 150
    legend_y = 25
    for ci, cls_name in enumerate(cls_names):
        color = cls_colors[ci % len(cls_colors)]
        cv2.rectangle(img, (legend_x, legend_y - 8), (legend_x + 10, legend_y + 2), color, -1)
        cv2.putText(
            img, cls_name, (legend_x + 15, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1,
        )
        legend_y += 15

    return img


def analyze_video(video_path, conf=0.45, iou=0.50, frame_skip=2, progress=gr.Progress(track_tqdm=True)):
    """Process a video frame-by-frame with live progress.

    Returns
    -------
    tuple[str, str, str]
        Path to annotated video, summary text, path to CSV detections file.
    """
    if video_path is None:
        return None, "No video uploaded.", None

    m = get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Cannot open video: {video_path}", None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_analyze = max(1, (total_frames + frame_skip - 1) // frame_skip)

    out_path = str(Path(video_path).parent / "annotated_output.mp4")
    csv_path = str(Path(video_path).parent / "detections.csv")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    global_counts = {}
    all_detections = []  # list of per-detection dicts
    processed = 0
    frame_idx = 0

    progress(0, desc=f"Starting analysis of {total_frames} frames...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip == 0:
            annotated_rgb, counts, detections = _annotate_frame(
                m, frame.copy(), conf, iou, cluster_mode=True,
            )
            for cls, cnt in counts.items():
                global_counts[cls] = global_counts.get(cls, 0) + cnt
            # Tag each detection with frame metadata
            timestamp = frame_idx / fps
            for det in detections:
                det["frame"] = frame_idx
                det["timestamp_s"] = round(timestamp, 3)
            all_detections.extend(detections)
            processed += 1
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        else:
            writer.write(frame)

        frame_idx += 1
        progress(
            processed / frames_to_analyze,
            desc=f"Frame {frame_idx}/{total_frames} | Detected {sum(global_counts.values())} objects",
        )

    cap.release()
    writer.release()

    # Write CSV
    csv_fields = ["frame", "timestamp_s", "class", "confidence",
                   "x1", "y1", "x2", "y2", "cx", "cy"]
    if all_detections:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer_csv.writeheader()
            writer_csv.writerows(all_detections)
    else:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=csv_fields)
            writer_csv.writeheader()

    # --- Generate heatmaps ---
    parent = Path(video_path).parent
    heatmap_path = str(parent / "detection_heatmap.png")
    heatmap_img = _build_heatmap(all_detections, width, height, global_counts)
    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

    # Per-class heatmaps
    per_class_heatmaps = []  # list of (class_name, np.ndarray)
    for cls_name in sorted(global_counts.keys()):
        cls_img = _build_heatmap(
            all_detections, width, height, global_counts, class_filter=cls_name,
        )
        per_class_heatmaps.append((cls_name, cls_img))
        # Also save to disk
        cls_path = str(parent / f"heatmap_{cls_name}.png")
        cv2.imwrite(cls_path, cv2.cvtColor(cls_img, cv2.COLOR_RGB2BGR))

    # --- Generate timeline chart ---
    total_duration = total_frames / fps
    timeline_img = _build_timeline(all_detections, total_duration, global_counts)
    timeline_path = str(parent / "detection_timeline.png")
    cv2.imwrite(timeline_path, cv2.cvtColor(timeline_img, cv2.COLOR_RGB2BGR))

    # Build summary
    total_det = sum(global_counts.values())
    lines = [
        f"\u2705 Analysis complete!",
        f"Video: {total_frames} frames total, {processed} analyzed (every {frame_skip} frames)",
        f"Duration: {total_frames / fps:.1f}s at {fps:.0f} FPS",
        f"Resolution: {width}x{height}",
    ]
    if total_det > 0:
        lines.append(f"\nTotal detections across video: {total_det}")
        for cls, cnt in sorted(global_counts.items()):
            lines.append(f"  {cls}: {cnt}")
    else:
        lines.append("\nNo objects detected in any analyzed frame.")
    lines.append(f"\nAnnotated video saved to: {out_path}")
    lines.append(f"Detections CSV saved to: {csv_path}")
    lines.append(f"Overall heatmap: {heatmap_path}")
    lines.append(f"Timeline chart: {timeline_path}")
    for cls_name, _ in per_class_heatmaps:
        lines.append(f"  {cls_name} heatmap: heatmap_{cls_name}.png")

    return out_path, "\n".join(lines), csv_path, heatmap_img, per_class_heatmaps, timeline_img


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

/* ---- Progress bar ---- */
.progress-bar-container {
    width: 100%;
    background: #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
    height: 24px;
    margin: 0.5rem 0;
}
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #2e7d32, #66bb6a);
    border-radius: 8px;
    transition: width 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
}
#video-status textarea {
    font-family: monospace;
    font-size: 0.85rem;
    background: #f5f5f5;
    border-radius: 8px;
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
                        photo_csv_download = gr.File(
                            label="\ud83d\udcc5 Download detections CSV",
                            visible=True,
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
                        video_status = gr.Textbox(
                            label="Processing status",
                            interactive=False,
                            lines=2,
                            elem_id="video-status",
                        )
                        video_summary = gr.Textbox(
                            label="Detection summary",
                            interactive=False,
                            lines=8,
                            elem_id="decision-box",
                        )
                        csv_download = gr.File(
                            label="\ud83d\udcc5 Download detections CSV",
                            visible=True,
                        )
                        heatmap_output = gr.Image(
                            type="numpy",
                            label="Overall detection heatmap",
                        )
                        gr.Markdown("### \ud83d\udd38 Per-class heatmaps")
                        per_class_gallery = gr.Gallery(
                            label="Heatmaps per waste type",
                            columns=4,
                            height="auto",
                        )
                        gr.Markdown("### \ud83d\udcc8 Detection timeline")
                        timeline_output = gr.Image(
                            type="numpy",
                            label="Detections per second",
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
            outputs=[output_image, decision_box, photo_csv_download],
        )
        def _run_video_analysis(video_path, conf, iou, frame_skip):
            """Wrapper that returns all video analysis outputs for the UI."""
            out_path, summary, csv_path, heatmap_img, per_class, timeline_img = analyze_video(
                video_path, conf, iou, frame_skip,
            )
            lines = summary.split("\n")
            status = lines[0] if lines else ""
            gallery_items = [(img, name) for name, img in per_class]
            return out_path, status, summary, csv_path, heatmap_img, gallery_items, timeline_img

        video_btn.click(
            fn=_run_video_analysis,
            inputs=[video_input, v_conf, v_iou, frame_skip_slider],
            outputs=[video_output, video_status, video_summary, csv_download, heatmap_output, per_class_gallery, timeline_output],
        )

    return demo


if __name__ == "__main__":
    run().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
