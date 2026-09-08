"""Real-time waste detection from camera or video source.

Simpler than the full conveyor pipeline — just reads frames, runs YOLO,
and displays results. Good for quick demonstrations and testing.
"""
import argparse
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from quality import assess_frame
from utils import source_value


def main():
    p = argparse.ArgumentParser(description="Real-time waste detection (camera mode)")
    p.add_argument("--model", required=True, help="Path to trained YOLO model (.pt)")
    p.add_argument("--source", default="0", help="Camera index (0) or video path")
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--show", action="store_true", help="Display live window")
    p.add_argument("--conf", type=float, default=None, help="Override confidence")
    p.add_argument("--iou", type=float, default=None, help="Override IoU threshold")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = YOLO(args.model)
    names = model.names

    conf = args.conf if args.conf is not None else cfg["detection"]["conf_threshold"]
    iou = args.iou if args.iou is not None else cfg["detection"]["iou_threshold"]

    cap = cv2.VideoCapture(source_value(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera/video: {args.source}")

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            quality = assess_frame(frame, **cfg["quality"])
            if not quality["ok"]:
                if args.show:
                    cv2.putText(
                        frame, "REJECT: poor frame quality", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    cv2.imshow("Waste Segregation", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            results = model.predict(
                frame, conf=conf, iou=iou,
                imgsz=cfg["detection"]["imgsz"], verbose=False,
            )
            r = results[0]

            if r.boxes is not None:
                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    cls_id = int(r.boxes.cls[i].item())
                    score = float(r.boxes.conf[i].item())
                    name = names[cls_id]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{name} {score:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )

            if args.show:
                cv2.imshow("Waste Segregation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
