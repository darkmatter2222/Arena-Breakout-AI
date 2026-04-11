"""
Scrub images with overlapping bounding boxes from the dataset.

Scans dataset/images/{train,val} and their labels. Any image whose label
file contains two or more boxes that overlap at all (IoU > 0) gets moved
to dataset/archive/{images,labels}/{split}/. Prints a summary when done.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
ARCHIVE = DATASET / "archive"


def parse_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def has_overlap(boxes: list) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) > 0:
                return True
    return False


def main():
    archived = 0
    scanned = 0

    for split in ("train", "val"):
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split
        if not img_dir.exists():
            continue

        arch_img = ARCHIVE / "images" / split
        arch_lbl = ARCHIVE / "labels" / split
        arch_img.mkdir(parents=True, exist_ok=True)
        arch_lbl.mkdir(parents=True, exist_ok=True)

        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            scanned += 1
            lbl = lbl_dir / (img.stem + ".txt")
            boxes = parse_boxes(lbl)
            if len(boxes) >= 2 and has_overlap(boxes):
                shutil.move(str(img), str(arch_img / img.name))
                if lbl.exists():
                    shutil.move(str(lbl), str(arch_lbl / lbl.name))
                archived += 1

    print(f"Scanned {scanned} images, archived {archived} with overlapping boxes.")
    print(f"Remaining: {scanned - archived}")
    print(f"Archive location: {ARCHIVE}")


if __name__ == "__main__":
    main()
