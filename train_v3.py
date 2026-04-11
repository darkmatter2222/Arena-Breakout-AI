"""
Train V3 model: remove only images with overlapping bounding boxes.

1. Scan all labels (train + val) for overlapping boxes (IoU > threshold).
2. Copy clean images + labels to dataset_v3/.
3. Fresh 85/15 train/val split.
4. Train yolo11n for 200 epochs with patience 40.
"""

import random
import shutil
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ORIG_DATASET = ROOT / "dataset"
V3_DATASET = ROOT / "dataset_v3"
V3_DATA_YAML = V3_DATASET / "data.yaml"

IOU_THRESHOLD = 0.0  # any overlap at all → evict
VAL_SPLIT = 0.15
SEED = 42

# Training
MODEL_NAME = "yolo11n.pt"
EPOCHS = 200
IMGSZ = 640
BATCH_SIZE = 16
PATIENCE = 40
WORKERS = 8


def parse_yolo_label(label_path: Path) -> list[tuple[float, float, float, float]]:
    """Read YOLO label file, return list of (cx, cy, w, h) normalized."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            _, cx, cy, w, h = parts[:5]
            boxes.append((float(cx), float(cy), float(w), float(h)))
    return boxes


def box_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two YOLO-format boxes (cx, cy, w, h) normalized."""
    ax1 = a[0] - a[2] / 2
    ay1 = a[1] - a[3] / 2
    ax2 = a[0] + a[2] / 2
    ay2 = a[1] + a[3] / 2

    bx1 = b[0] - b[2] / 2
    by1 = b[1] - b[3] / 2
    bx2 = b[0] + b[2] / 2
    by2 = b[1] + b[3] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def has_overlapping_boxes(boxes: list[tuple]) -> bool:
    """Return True if any pair of boxes has IoU > threshold."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) > IOU_THRESHOLD:
                return True
    return False


def collect_all_images() -> list[tuple[Path, Path]]:
    """Gather all (image, label) pairs from train + val."""
    pairs = []
    for split in ["train", "val"]:
        img_dir = ORIG_DATASET / "images" / split
        lbl_dir = ORIG_DATASET / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lbl = lbl_dir / (img.stem + ".txt")
                pairs.append((img, lbl))
    return pairs


def cleanse_and_copy():
    """Filter out images with overlapping boxes, copy clean ones to dataset_v3."""
    pairs = collect_all_images()
    print(f"[cleanse] Total source images: {len(pairs)}")

    clean = []
    evicted = 0
    overlap_count = 0

    for img_path, lbl_path in pairs:
        boxes = parse_yolo_label(lbl_path)
        if len(boxes) >= 2 and has_overlapping_boxes(boxes):
            overlap_count += 1
            evicted += 1
            continue
        clean.append((img_path, lbl_path))

    print(f"[cleanse] Overlapping boxes: {overlap_count} images evicted")
    print(f"[cleanse] Clean images: {len(clean)}")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(clean)
    n_val = max(1, int(len(clean) * VAL_SPLIT))
    val_set = clean[:n_val]
    train_set = clean[n_val:]

    print(f"[split] Train: {len(train_set)}, Val: {len(val_set)}")

    # Create directory structure
    for split in ["train", "val"]:
        (V3_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (V3_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for img_path, lbl_path in split_data:
            dst_img = V3_DATASET / "images" / split_name / img_path.name
            dst_lbl = V3_DATASET / "labels" / split_name / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                # Empty label = negative sample
                dst_lbl.write_text("")

    # Write data.yaml
    data_yaml = {
        "path": str(V3_DATASET).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "target"},
    }
    V3_DATA_YAML.write_text(yaml.dump(data_yaml, default_flow_style=False))
    print(f"[cleanse] Wrote {V3_DATA_YAML}")


def train():
    """Train YOLO11n on the cleansed dataset, resuming from last checkpoint if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[train] Device: {device}")

    last_pt = ROOT / "runs" / "yolo11n_target_v3" / "weights" / "last.pt"
    if last_pt.exists():
        print(f"[train] Resuming from {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        print(f"[train] Epochs: {EPOCHS}, Patience: {PATIENCE}, Batch: {BATCH_SIZE}")
        model = YOLO(MODEL_NAME)
        model.train(
            data=str(V3_DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            device=device,
            workers=WORKERS,
            seed=SEED,
            project=str(ROOT / "runs"),
            name="yolo11n_target_v3",
            exist_ok=True,
            save_period=25,
            val=True,
            plots=True,
            verbose=True,
        )
    print("\n[train] V3 training complete!")
    print(f"[train] Weights: {ROOT / 'runs' / 'yolo11n_target_v3' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    # Only cleanse if dataset_v3 doesn't exist yet
    if not V3_DATASET.exists() or not V3_DATA_YAML.exists():
        if V3_DATASET.exists():
            shutil.rmtree(V3_DATASET)
        cleanse_and_copy()
    else:
        print(f"[cleanse] Reusing existing {V3_DATASET}")

    train()
