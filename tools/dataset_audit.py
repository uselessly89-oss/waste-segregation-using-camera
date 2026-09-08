"""Audit a YOLO-format waste dataset.

Checks for missing labels, counts images per split, and reports per-class
distribution. Run this before training to catch data issues early.
"""
from pathlib import Path
from collections import Counter

import yaml


def main():
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    root = Path(cfg["path"])
    names = cfg["names"]

    for split in ("train", "val", "test"):
        image_dir = root / cfg[split]
        label_dir = Path(str(image_dir).replace("/images/", "/labels/"))
        counts = Counter()
        images = 0
        missing = 0

        for image in image_dir.rglob("*") if image_dir.exists() else []:
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            images += 1
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                missing += 1
                continue
            for line in label.read_text().splitlines():
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1

        print(f"{split}: images={images}, missing_labels={missing}")
        for cls_id, count in sorted(counts.items()):
            print(f"  {cls_id}: {names.get(cls_id, cls_id)} -> {count}")


if __name__ == "__main__":
    main()
