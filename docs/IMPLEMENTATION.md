# Waste Segregation Using Camera — Implementation

## Architecture

```
Camera / Upload
      |
  Quality Gate  (reject dark / blurry frames)
      |
  YOLO Detection + Segmentation
      |
  Cluster Report  (visible objects + REJECT for uncertain)
```

This is a simplified camera-only pipeline. The `waste` repository contains
the full industrial pipeline with ByteTrack tracking, temporal consensus,
conveyor timing, and actuator control.

## What is Implemented

1. **Capture** — OpenCV reads from webcam, video file, or Gradio camera input.
2. **Quality gate** — Rejects frames that are too dark, too bright, or blurry.
3. **Detection** — Ultralytics YOLO produces class, confidence, and bounding boxes.
4. **Cluster mode** — Reports all visible objects; uncertain/hidden material → REJECT.
5. **Training** — Standard YOLO training pipeline with data augmentation.
6. **Evaluation** — mAP metrics, confusion matrix, per-class analysis.
7. **Gradio app** — Web UI for laptop webcam and phone camera access.

## Classes

- `plastic`
- `paper`
- `metal`
- `organic`

Future: `glass`, `textile`, `e_waste`, `other` (requires additional labeled data).

## Training

```bash
# Audit dataset
python tools/dataset_audit.py

# Train
python src/train.py --data configs/data.yaml --model yolo11n-seg.pt --epochs 100

# Copy weights to models/
cp runs/waste/waste_detector/weights/best.pt models/best.pt

# Evaluate
python tools/evaluate.py --model models/best.pt --data configs/data.yaml
```

## Running

```bash
# Gradio web app (easiest)
python app.py

# Command-line detection
python src/detect.py --model models/best.pt --source 0 --show
```

## Limitations

RGB images cannot identify hidden contents, chemical composition, or material
properties when appearance is ambiguous. The correct behavior is REJECT / manual
review, or use calibrated auxiliary sensing (NIR, depth, weight).
