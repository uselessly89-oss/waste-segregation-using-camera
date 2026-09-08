# ♻️ Waste Segregation Using Camera

AI-based waste detection and segregation using a camera. The simplest way to
demo waste classification — point a laptop webcam, open the app on a phone,
or upload a photo.

This is the **camera-only** project. For the full industrial pipeline with
conveyor tracking, temporal consensus, and actuator control, see the
[`waste`](https://github.com/uselessly89-oss/waste) repository.

## What it can recognize

Initial model classes:
- `plastic`
- `paper`
- `metal`
- `organic`

The architecture can later add `glass`, `textile`, `e_waste`, and `other`.

## Quick Start — Laptop or Phone Camera

Install the app dependencies:

```bash
pip install -r requirements-app.txt
```

Start:

```bash
python app.py
```

### Laptop

Open the displayed browser address and select **Webcam**.

### Phone

Keep the laptop and phone on the same Wi-Fi network. Start the app with:

```bash
python app.py
```

Then open `http://<LAPTOP-IP>:7860` on the phone and use the phone camera.

See `docs/CAMERA_MODE.md` for the full procedure.

## Command-line Detection

```bash
python src/detect.py --model models/best.pt --source 0 --show
```

- `--source 0` — default webcam
- `--source video.mp4` — a recorded video
- `--conf 0.40` — lower confidence threshold

## Dataset

Put YOLO segmentation data under:

```text
dataset/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
└── labels/test
```

Use real examples containing lighting changes, shadows, motion blur, dirty/wet
objects, crushed/deformed objects, partial occlusion, different orientations,
and multiple simultaneous objects.

Audit it:

```bash
python tools/dataset_audit.py
```

## Training

```bash
pip install -r requirements.txt
python tools/dataset_audit.py
python src/train.py --data configs/data.yaml --model yolo11n-seg.pt --epochs 100 --imgsz 640
```

Copy the trained weights for the app:

```bash
cp runs/waste/waste_detector/weights/best.pt models/best.pt
```

## Evaluation

```bash
python tools/evaluate.py --model models/best.pt --data configs/data.yaml --split test
```

## Cluster Mode

The app reports every confidently visible object. It does **not** guess about
hidden items. Low-confidence or visually inseparable material is routed to
REJECT. For mixed piles, the correct design is segmentation plus auxiliary
sensing and a second pass after items are separated.

## Project Structure

```text
├── app.py                  # Gradio web app (webcam + phone + upload)
├── models/                 # Trained weights (best.pt)
├── configs/
│   ├── data.yaml           # Dataset class definitions
│   └── runtime.yaml        # Detection and quality thresholds
├── src/
│   ├── detect.py           # Real-time camera detection (CLI)
│   ├── train.py            # YOLO training script
│   └── quality.py          # Image quality gate
├── tools/
│   ├── dataset_audit.py    # Check dataset integrity
│   └── evaluate.py         # Model evaluation
├── dataset/                # YOLO-format training data
├── tests/                  # Unit tests
├── docs/
│   ├── CAMERA_MODE.md      # Camera usage guide
│   └── IMPLEMENTATION.md   # Architecture details
└── .github/workflows/ci.yml
```

## Safety

This camera app is for perception and demonstration. Physical actuator control
requires additional hardware validation. Always validate emergency stops,
electrical isolation, and mechanical interlocks before connecting real hardware.

## Project Status

The repository contains the camera app, inference pipeline, training code,
quality gate, dataset audit, evaluation, tests, and CI. A trained `best.pt`
cannot be included until a real labeled dataset is supplied.
