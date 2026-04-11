"""
YOLO11n training script for single-class detection.
RTX 5090 (32 GB VRAM) · CUDA 12.8 · Ultralytics 8.x
"""

import os
import random
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# ── cuDNN stability ──────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ── paths ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
DATA_YAML   = DATASET_DIR / "data.yaml"
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_DIR  = DATASET_DIR / "labels"

# ── training hyper-parameters ────────────────────────────────────────────
MODEL_NAME  = "yolo11n.pt"        # nano – fastest, good for real-time
EPOCHS      = 150
IMGSZ       = 640                 # standard YOLO input size
BATCH_SIZE  = 16                  # 16 avoids cuDNN OOM with this dataset
PATIENCE    = 30                  # early-stop if no improvement for N epochs
VAL_SPLIT   = 0.15               # 15% held out for validation
SEED        = 42
WORKERS     = 8


def split_dataset():
    """Fresh random train/val split every run — merges everything back first."""
    train_img = IMAGES_DIR / "train"
    val_img   = IMAGES_DIR / "val"
    train_lbl = LABELS_DIR / "train"
    val_lbl   = LABELS_DIR / "val"

    train_img.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    # ── 1. Merge val back into train ─────────────────────────────────
    moved_back = 0
    for img_path in list(val_img.glob("*.jpg")) + list(val_img.glob("*.png")):
        dest = train_img / img_path.name
        if not dest.exists():
            shutil.move(str(img_path), str(dest))
            moved_back += 1
        else:
            img_path.unlink()  # duplicate
        lbl = val_lbl / f"{img_path.stem}.txt"
        lbl_dest = train_lbl / lbl.name
        if lbl.exists():
            if not lbl_dest.exists():
                shutil.move(str(lbl), str(lbl_dest))
            else:
                lbl.unlink()
    if moved_back:
        print(f"[split] merged {moved_back} val images back into train")

    # ── 2. Fresh random split ────────────────────────────────────────
    images = sorted(train_img.glob("*.jpg")) + sorted(train_img.glob("*.png"))
    if not images:
        print("[split] ERROR: no images found in", train_img)
        sys.exit(1)

    random.seed(SEED)  # reproducible split for cross-run comparison
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_SPLIT))

    for img_path in images[:n_val]:
        lbl_path = train_lbl / f"{img_path.stem}.txt"
        shutil.move(str(img_path), str(val_img / img_path.name))
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(val_lbl / lbl_path.name))

    print(f"[split] {len(images)} total → train={len(images) - n_val}, val={n_val}")


def update_data_yaml():
    """Ensure data.yaml points val to images/val."""
    text = DATA_YAML.read_text()
    if "images/val" not in text:
        text = text.replace("val: images/train", "val: images/val")
        DATA_YAML.write_text(text)
        print("[yaml] updated data.yaml → val: images/val")


def train():
    """Run YOLO11n training, resuming from last checkpoint if available."""
    print("=" * 60)
    print(f"  YOLO11n Training")
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  ImgSz: {IMGSZ}")
    print("=" * 60)

    run_dir = ROOT / "runs" / "yolo11n_target"
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"

    # Detect if a previous run exists and whether it's resumable
    resume = False
    if last_pt.exists():
        # Check if training was still in progress (not completed/early-stopped)
        results_csv = run_dir / "results.csv"
        completed = False
        if results_csv.exists():
            lines = [l for l in results_csv.read_text().splitlines() if l.strip()]
            last_epoch = len(lines) - 1  # minus header
            if last_epoch <= 0:
                completed = False
            else:
                # Completed if hit all epochs OR early-stopped (best.pt newer than last.pt)
                completed = (last_epoch >= EPOCHS) or (
                    best_pt.exists() and best_pt.stat().st_mtime > last_pt.stat().st_mtime
                )
        if completed:
            # Archive old run, start fresh
            archive_name = f"yolo11n_target_{int(best_pt.stat().st_mtime)}" if best_pt.exists() else f"yolo11n_target_old"
            archive_dir = ROOT / "runs" / archive_name
            if not archive_dir.exists():
                print(f"  Previous training complete — archiving to runs/{archive_dir.name}/")
                shutil.move(str(run_dir), str(archive_dir))
            else:
                print(f"  Previous training complete — clearing run dir")
                shutil.rmtree(run_dir)
        else:
            resume = True
            print(f"  Incomplete training found — resuming from {last_pt}")

    if resume:
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
    else:
        print(f"  Starting fresh training on {DATA_YAML}")
        model = YOLO(MODEL_NAME)
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            device=0,
            workers=WORKERS,
            seed=SEED,
            # augmentation
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
            name="yolo11n_target",
            exist_ok=True,
            plots=True,
            save=True,
            save_period=25,          # checkpoint every 25 epochs
            val=True,
            verbose=True,
        )
    return results


def evaluate():
    """Run validation and print performance analytics."""
    # Find the best weights from the latest run
    run_dir = ROOT / "runs" / "yolo11n_target"
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print("[eval] best.pt not found – skipping evaluation")
        return

    print("\n" + "=" * 60)
    print("  Performance Analytics")
    print("=" * 60)

    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=0,
        plots=True,
        save_json=True,       # COCO-format per-image predictions for error analysis
        save_txt=True,        # per-image prediction .txt files
    )

    # ── summary metrics ──────────────────────────────────────────────
    box = metrics.box
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'mAP@0.5':<25} {box.map50:>10.4f}")
    print(f"{'mAP@0.5:0.95':<25} {box.map:>10.4f}")
    print(f"{'Precision':<25} {box.mp:>10.4f}")
    print(f"{'Recall':<25} {box.mr:>10.4f}")

    # Per-class (just one class here)
    if hasattr(box, 'maps') and len(box.maps) > 0:
        print(f"\nPer-class mAP@0.5:0.95:")
        for i, m in enumerate(box.maps):
            print(f"  class {i} (target): {m:.4f}")

    # ── F1-optimal confidence threshold ──────────────────────────────
    if hasattr(metrics.box, 'f1') and hasattr(metrics.box, 'conf'):
        f1_arr = metrics.box.f1
        if len(f1_arr) > 0:
            import numpy as np
            best_idx = np.argmax(f1_arr.mean(0)) if f1_arr.ndim > 1 else np.argmax(f1_arr)
            print(f"\nF1-optimal conf threshold: {metrics.box.conf[best_idx]:.3f}")
            print(f"  F1 at optimal          : {(f1_arr.mean(0)[best_idx] if f1_arr.ndim > 1 else f1_arr[best_idx]):.4f}")

    # ── inference speed ──────────────────────────────────────────────
    speed = metrics.speed
    print(f"\nInference Speed (per image):")
    print(f"  Preprocess : {speed.get('preprocess', 0):.1f} ms")
    print(f"  Inference  : {speed.get('inference', 0):.1f} ms")
    print(f"  Postprocess: {speed.get('postprocess', 0):.1f} ms")
    total = sum(speed.get(k, 0) for k in ('preprocess', 'inference', 'postprocess'))
    if total > 0:
        print(f"  Total      : {total:.1f} ms  ({1000/total:.0f} FPS)")

    # ── generated artifacts ──────────────────────────────────────────
    print(f"\nArtifacts saved to: {run_dir}")
    plot_files = sorted(run_dir.glob("*.png")) + sorted(run_dir.glob("*.jpg"))
    for p in plot_files:
        print(f"  {p.name}")
    # Check for JSON predictions
    json_files = sorted(run_dir.glob("**/*.json"))
    for j in json_files:
        print(f"  {j.relative_to(run_dir)} ({j.stat().st_size / 1024:.0f} KB)")

    print(f"\nBest weights: {best_pt}")
    print(f"Model size  : {best_pt.stat().st_size / 1024**2:.1f} MB")
    print(f"\nKey files for optimization review:")
    print(f"  results.csv          — per-epoch train/val metrics")
    print(f"  predictions.json     — per-image COCO predictions")
    print(f"  confusion_matrix.png — TP/FP/FN breakdown")
    print(f"  F1_curve.png         — F1 vs confidence")
    print(f"  PR_curve.png         — precision-recall curve")


if __name__ == "__main__":
    split_dataset()
    update_data_yaml()
    train()
    evaluate()
