"""
AV2 Training Pipeline — Data Quality Cleansing + YOLO11n Training

Phase 1 – Analyse every image+label in the V1 dataset and flag problems:
    • Overlapping bounding boxes (IoU > 0.0 between any two boxes in the same image)
    • Missing / corrupt images
    • Empty labels (no detections — may dilute positive training signal)
    • Malformed label lines (wrong field count, out-of-range values)
    • Extremely tiny boxes (area < 0.5 % of image — likely noise)
    • Extremely large boxes (area > 60 % of image — likely bad annotation)
    • V1 model low-confidence disagreement (model sees no target but label has one, or vice-versa)

Phase 2 – Copy only clean images+labels into dataset_v2/
Phase 3 – Split train/val and train yolo11n → runs/yolo11n_target_v2/

Does NOT delete any original data.
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# V1 dataset
V1_IMAGES_TRAIN = ROOT / "dataset" / "images" / "train"
V1_LABELS_TRAIN = ROOT / "dataset" / "labels" / "train"
V1_IMAGES_VAL   = ROOT / "dataset" / "images" / "val"
V1_LABELS_VAL   = ROOT / "dataset" / "labels" / "val"
V1_MODEL        = ROOT / "runs" / "yolo11n_target" / "weights" / "best.pt"

# V2 output
V2_ROOT      = ROOT / "dataset_v2"
V2_IMAGES    = V2_ROOT / "images"
V2_LABELS    = V2_ROOT / "labels"
V2_YAML      = V2_ROOT / "data.yaml"
V2_REPORT    = ROOT / "v2_cleanse_report.json"

# ── Quality thresholds ───────────────────────────────────────────────────
IOU_OVERLAP_THRESH   = 0.0    # any overlap at all = flagged
MIN_BOX_AREA_FRAC    = 0.005  # < 0.5 % of image area
MAX_BOX_AREA_FRAC    = 0.60   # > 60 % of image area
CONF_DISAGREE_THRESH = 0.40   # V1 model conf below this = "doesn't see it"
BATCH_INFER_SIZE     = 32

# ── Training hypers (same as V1 with tweaks) ─────────────────────────────
MODEL_NAME = "yolo11n.pt"
EPOCHS     = 200
IMGSZ      = 640
BATCH_SIZE = 32
PATIENCE   = 40
VAL_SPLIT  = 0.15
SEED       = 42
WORKERS    = 8


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Data quality analysis
# ═════════════════════════════════════════════════════════════════════════

def iou(box_a, box_b):
    """Compute IoU between two YOLO boxes [cx, cy, w, h] (normalised)."""
    ax1 = box_a[0] - box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay2 = box_a[1] + box_a[3] / 2

    bx1 = box_b[0] - box_b[2] / 2
    by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2
    by2 = box_b[1] + box_b[3] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    if union < 1e-9:
        return 0.0
    return inter / union


def parse_label(label_path: Path):
    """Parse a YOLO label file. Returns list of (cls, cx, cy, w, h) or error string."""
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []
    boxes = []
    for i, line in enumerate(text.split("\n")):
        parts = line.strip().split()
        if len(parts) != 5:
            return f"malformed_line_{i}_fields_{len(parts)}"
        try:
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            return f"malformed_line_{i}_parse"
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return f"out_of_range_line_{i}"
        boxes.append((cls, cx, cy, w, h))
    return boxes


def check_image_readable(img_path: Path) -> bool:
    """Verify the image can be decoded."""
    img = cv2.imread(str(img_path))
    return img is not None and img.size > 0


def analyse_pair(img_path: Path, lbl_path: Path):
    """Analyse one image+label pair. Returns (is_clean, [reasons])."""
    reasons = []

    # -- image readability --
    if not img_path.exists():
        return False, ["image_missing"]
    if not check_image_readable(img_path):
        return False, ["image_corrupt"]

    # -- label parsing --
    boxes = parse_label(lbl_path)
    if isinstance(boxes, str):
        return False, [boxes]

    # -- empty label (negative sample) — keep a limited proportion later --
    if len(boxes) == 0:
        return True, ["empty_label"]

    # -- per-box checks --
    for i, (cls, cx, cy, w, h) in enumerate(boxes):
        area = w * h
        if area < MIN_BOX_AREA_FRAC:
            reasons.append(f"tiny_box_{i}_area_{area:.4f}")
        if area > MAX_BOX_AREA_FRAC:
            reasons.append(f"huge_box_{i}_area_{area:.4f}")

    # -- overlapping boxes --
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box_i = boxes[i][1:]  # (cx, cy, w, h)
            box_j = boxes[j][1:]
            overlap = iou(box_i, box_j)
            if overlap > IOU_OVERLAP_THRESH:
                reasons.append(f"overlap_boxes_{i}_{j}_iou_{overlap:.3f}")

    if reasons:
        return False, reasons
    return True, []


def run_model_disagreement(model, image_paths, label_map):
    """Check if V1 model disagrees with labels (batch inference)."""
    disagreements = {}
    total = len(image_paths)

    for start in range(0, total, BATCH_INFER_SIZE):
        batch_paths = image_paths[start:start + BATCH_INFER_SIZE]
        batch_imgs = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is not None:
                batch_imgs.append(img)
            else:
                batch_imgs.append(np.zeros((416, 600, 3), dtype=np.uint8))

        results = model(batch_imgs, imgsz=640, conf=CONF_DISAGREE_THRESH, verbose=False)

        for img_path, result in zip(batch_paths, results):
            stem = img_path.stem
            lbl_boxes = label_map.get(stem, [])
            has_label = len(lbl_boxes) > 0
            model_dets = len(result.boxes) if result.boxes is not None else 0
            has_model = model_dets > 0

            if has_label and not has_model:
                disagreements[stem] = "label_yes_model_no"
            elif has_model and not has_label:
                disagreements[stem] = "label_no_model_yes"

        done = min(start + BATCH_INFER_SIZE, total)
        if done % 200 < BATCH_INFER_SIZE or done == total:
            print(f"  [model check] {done}/{total}")

    return disagreements


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Build clean V2 dataset
# ═════════════════════════════════════════════════════════════════════════

def build_v2_dataset(clean_stems, all_image_dir, all_label_dir, negative_limit_frac=0.15):
    """Copy clean images+labels into dataset_v2, capping negatives."""
    train_img = V2_IMAGES / "train"
    train_lbl = V2_LABELS / "train"
    val_img   = V2_IMAGES / "val"
    val_lbl   = V2_LABELS / "val"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Separate positives and negatives
    positives = []
    negatives = []
    for stem, img_ext in clean_stems:
        lbl = all_label_dir / f"{stem}.txt"
        boxes = parse_label(lbl)
        if isinstance(boxes, list) and len(boxes) > 0:
            positives.append((stem, img_ext))
        else:
            negatives.append((stem, img_ext))

    # Cap negatives to a fraction of positives
    max_negatives = max(1, int(len(positives) * negative_limit_frac))
    random.seed(SEED)
    if len(negatives) > max_negatives:
        random.shuffle(negatives)
        negatives = negatives[:max_negatives]

    all_clean = positives + negatives
    random.seed(SEED)
    random.shuffle(all_clean)

    # Split
    n_val = max(1, int(len(all_clean) * VAL_SPLIT))
    val_set = all_clean[:n_val]
    train_set = all_clean[n_val:]

    for subset, img_dir, lbl_dir in [(train_set, train_img, train_lbl),
                                      (val_set, val_img, val_lbl)]:
        for stem, ext in subset:
            src_img = all_image_dir / f"{stem}{ext}"
            src_lbl = all_label_dir / f"{stem}.txt"
            shutil.copy2(str(src_img), str(img_dir / f"{stem}{ext}"))
            if src_lbl.exists():
                shutil.copy2(str(src_lbl), str(lbl_dir / f"{stem}.txt"))
            else:
                # Write empty label for negatives
                (lbl_dir / f"{stem}.txt").write_text("")

    print(f"[V2 dataset] {len(train_set)} train, {len(val_set)} val")
    print(f"  positives: {len(positives)}, negatives kept: {len(negatives)}")
    return len(train_set), len(val_set)


def write_v2_yaml():
    """Write data.yaml for the V2 dataset."""
    yaml_content = f"""# Vision Dataset V2 - Cleansed
path: {V2_ROOT.as_posix()}
train: images/train
val: images/val
test:

names:
  0: target
"""
    V2_YAML.write_text(yaml_content)
    print(f"[V2 yaml] written to {V2_YAML}")


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Train AV2
# ═════════════════════════════════════════════════════════════════════════

def train_v2():
    print("=" * 60)
    print("  AV2 — YOLO11n Training on Cleansed Data")
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  Patience: {PATIENCE}")
    print("=" * 60)

    model = YOLO(MODEL_NAME)
    results = model.train(
        data=str(V2_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=0,
        workers=WORKERS,
        seed=SEED,
        # augmentation – same as V1
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        # output
        project=str(ROOT / "runs"),
        name="yolo11n_target_v2",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=25,
        val=True,
        verbose=True,
    )
    return results


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       AV2 Pipeline — Cleanse + Train                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Gather all images from BOTH train and val of V1 ───────────────
    # We'll re-split cleanly for V2
    # First, collect everything into a temporary merged view
    all_images = {}  # stem -> (img_path, lbl_dir)
    for img_dir, lbl_dir in [(V1_IMAGES_TRAIN, V1_LABELS_TRAIN),
                              (V1_IMAGES_VAL, V1_LABELS_VAL)]:
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                all_images[f.stem] = (f, lbl_dir)

    total = len(all_images)
    print(f"[scan] Found {total} images across V1 train+val\n")

    if total == 0:
        print("ERROR: No images found.")
        sys.exit(1)

    # ── Phase 1a: Static quality checks ───────────────────────────────
    print("─── Phase 1a: Static quality analysis ───────────────────")
    clean_stems = []       # (stem, ext)
    evicted = {}           # stem -> [reasons]
    empty_stems = []       # stems with no annotations
    label_map = {}         # stem -> list of box tuples (for model check)

    for i, (stem, (img_path, lbl_dir)) in enumerate(all_images.items()):
        lbl_path = lbl_dir / f"{stem}.txt"
        is_clean, reasons = analyse_pair(img_path, lbl_path)

        # Store parsed boxes for model disagreement check
        boxes = parse_label(lbl_path)
        if isinstance(boxes, list):
            label_map[stem] = boxes

        if not is_clean:
            evicted[stem] = reasons
        elif "empty_label" in reasons:
            empty_stems.append((stem, img_path.suffix))
            clean_stems.append((stem, img_path.suffix))
        else:
            clean_stems.append((stem, img_path.suffix))

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] clean={len(clean_stems)} evicted={len(evicted)}")

    print(f"\n  Static check done: {len(clean_stems)} clean, {len(evicted)} evicted")
    print(f"  Empty labels (negatives): {len(empty_stems)}")

    # Summarise eviction reasons
    reason_counts = {}
    for reasons in evicted.values():
        for r in reasons:
            tag = r.split("_")[0] if "_" in r else r
            reason_counts[tag] = reason_counts.get(tag, 0) + 1
    if reason_counts:
        print("  Eviction reasons:")
        for tag, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {tag}: {count}")

    # ── Phase 1b: V1 model disagreement ───────────────────────────────
    print("\n─── Phase 1b: V1 model disagreement check ──────────────")
    if V1_MODEL.exists():
        v1_model = YOLO(str(V1_MODEL))
        # Only check currently-clean stems
        clean_img_paths = []
        clean_stem_list = []
        for stem, ext in clean_stems:
            img_path = all_images[stem][0]
            clean_img_paths.append(img_path)
            clean_stem_list.append(stem)

        disagreements = run_model_disagreement(v1_model, clean_img_paths, label_map)

        # Evict "label says yes but model sees nothing" — likely bad annotation
        model_evict_count = 0
        for stem, reason in disagreements.items():
            if reason == "label_yes_model_no":
                evicted[stem] = evicted.get(stem, []) + ["model_disagree_" + reason]
                clean_stems = [(s, e) for s, e in clean_stems if s != stem]
                model_evict_count += 1

        print(f"  Model disagreements found: {len(disagreements)}")
        print(f"    label_yes_model_no (evicted): {model_evict_count}")
        label_no_model_yes = sum(1 for v in disagreements.values() if v == "label_no_model_yes")
        print(f"    label_no_model_yes (kept — could be missed annotations): {label_no_model_yes}")
    else:
        print("  WARNING: V1 model not found, skipping model disagreement check")

    print(f"\n  Final: {len(clean_stems)} clean, {len(evicted)} total evicted")

    # ── Save report ───────────────────────────────────────────────────
    report = {
        "total_images": total,
        "clean_images": len(clean_stems),
        "evicted_images": len(evicted),
        "eviction_reasons_summary": reason_counts,
        "evicted_files": {k: v for k, v in sorted(evicted.items())},
    }
    V2_REPORT.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {V2_REPORT}\n")

    # ── Phase 2: Build V2 dataset ─────────────────────────────────────
    print("─── Phase 2: Building V2 dataset ────────────────────────")
    if V2_ROOT.exists():
        print(f"  Removing old V2 dataset at {V2_ROOT}")
        shutil.rmtree(V2_ROOT)

    # We need a single source dir for copying — create temp merged dirs
    merged_img = ROOT / "_v2_merged_img"
    merged_lbl = ROOT / "_v2_merged_lbl"
    merged_img.mkdir(exist_ok=True)
    merged_lbl.mkdir(exist_ok=True)

    # Copy all clean images into merged dir
    for stem, ext in clean_stems:
        src_img = all_images[stem][0]
        lbl_dir = all_images[stem][1]
        src_lbl = lbl_dir / f"{stem}.txt"
        shutil.copy2(str(src_img), str(merged_img / f"{stem}{ext}"))
        if src_lbl.exists():
            shutil.copy2(str(src_lbl), str(merged_lbl / f"{stem}.txt"))

    n_train, n_val = build_v2_dataset(clean_stems, merged_img, merged_lbl)
    write_v2_yaml()

    # Clean up merged temp
    shutil.rmtree(merged_img)
    shutil.rmtree(merged_lbl)

    # ── Phase 3: Train ────────────────────────────────────────────────
    print("\n─── Phase 3: Training AV2 ──────────────────────────────")
    train_v2()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       AV2 Training Complete!                           ║")
    print(f"║  Model: runs/yolo11n_target_v2/weights/best.pt        ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
