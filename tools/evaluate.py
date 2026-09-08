"""Evaluate a trained YOLO waste model on the configured test/val split.

Reports mAP, per-class metrics, and saves confusion matrix plots.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Evaluate waste model")
    p.add_argument("--model", required=True, help="Path to best.pt")
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, split=args.split, plots=True
    )

    print("Evaluation complete.")
    print("mAP50:", getattr(metrics.box, "map50", None))
    print("mAP50-95:", getattr(metrics.box, "map", None))
    if hasattr(metrics, "seg"):
        print("Seg mAP50:", getattr(metrics.seg, "map50", None))
        print("Seg mAP50-95:", getattr(metrics.seg, "map", None))
    print("See the Ultralytics run directory for confusion matrix and plots.")


if __name__ == "__main__":
    main()
