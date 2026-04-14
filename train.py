"""
YOLO11n training script for single-class detection.
RTX 5090 (32 GB VRAM) · CUDA 12.8 · Ultralytics 8.x

v4 hyperparams: reverted loss weights to defaults (v3 rebalancing regressed
mAP@0.5:0.95), kept cutmix/perspective/lr0 from v3, added stratified
negative downsampling in training split (53% → 35% negatives).
"""

import random
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# ── cuDNN tuning ─────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True   # constant input size → safe & faster
torch.backends.cudnn.deterministic = True


# ── paths ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
DATA_YAML   = DATASET_DIR / "data.yaml"
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_DIR  = DATASET_DIR / "labels"
SPLITS_DIR  = DATASET_DIR / "splits"
MODELS_DIR  = ROOT / "models"

# ── training hyper-parameters ────────────────────────────────────────────
MODEL_NAME  = "yolo11n.pt"        # nano – fastest, good for real-time
EPOCHS      = 300                 # doubled — previous run never early-stopped
IMGSZ       = 800                 # up from 640 — exploit 32 GB VRAM for tighter boxes
BATCH_SIZE  = 32                  # up from 16 — only used 2.3/32 GB before
PATIENCE    = 50                  # wider window so plateaus don't kill a run
VAL_SPLIT       = 0.15            # 15% held out for validation
TARGET_NEG_RATE = 0.35            # cap training negatives at 35% (was 53%)
SEED            = 42
WORKERS         = 8


def split_dataset():
    """Write splits/train.txt and splits/val.txt with stratified splits.

    Images and labels stay flat in dataset/images/ and dataset/labels/.
    Training negatives are downsampled to TARGET_NEG_RATE to boost
    positive density per epoch.  Validation keeps full distribution.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        print("[split] ERROR: no images found in", IMAGES_DIR)
        sys.exit(1)

    # Classify images as positive (has annotations) or negative (empty label)
    positives, negatives = [], []
    for img in images:
        lbl = LABELS_DIR / img.with_suffix(".txt").name
        if lbl.exists() and lbl.stat().st_size > 0:
            positives.append(img)
        else:
            negatives.append(img)

    print(f"[split] {len(images)} total: {len(positives)} pos, {len(negatives)} neg"
          f" ({100 * len(negatives) / len(images):.0f}% neg)")

    random.seed(SEED)
    random.shuffle(positives)
    random.shuffle(negatives)

    # Stratified split — both groups get same val fraction
    n_val_pos = max(1, int(len(positives) * VAL_SPLIT))
    n_val_neg = max(1, int(len(negatives) * VAL_SPLIT))

    val_set   = positives[:n_val_pos] + negatives[:n_val_neg]
    train_pos = positives[n_val_pos:]
    train_neg = negatives[n_val_neg:]

    # Downsample training negatives to TARGET_NEG_RATE
    max_neg = int(len(train_pos) * TARGET_NEG_RATE / (1 - TARGET_NEG_RATE))
    if len(train_neg) > max_neg:
        print(f"[split] Downsampling train negatives: {len(train_neg)} → {max_neg}"
              f" (target {TARGET_NEG_RATE * 100:.0f}%)")
        train_neg = train_neg[:max_neg]

    train_set = train_pos + train_neg
    random.shuffle(train_set)
    random.shuffle(val_set)

    train_txt = SPLITS_DIR / "train.txt"
    val_txt   = SPLITS_DIR / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_set) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val_set) + "\n")

    neg_pct = 100 * len(train_neg) / len(train_set) if train_set else 0
    print(f"[split] train={len(train_set)} ({len(train_neg)} neg, {neg_pct:.0f}%),"
          f" val={len(val_set)}")
    print(f"[split] wrote {train_txt.relative_to(ROOT)} and {val_txt.relative_to(ROOT)}")


def update_data_yaml():
    """Write data.yaml pointing to text-file splits."""
    DATA_YAML.write_text(
        f"# Vision Dataset - YOLO11n Training\n"
        f"path: {DATASET_DIR.as_posix()}\n"
        f"train: splits/train.txt\n"
        f"val: splits/val.txt\n"
        f"\n"
        f"names:\n"
        f"  0: target\n"
    )
    print("[yaml] wrote data.yaml → text-file splits")


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
            # ── loss weights ── defaults (v3 rebalancing regressed mAP@0.5:0.95)
            box=7.5,
            dfl=1.5,
            cls=0.5,
            # ── learning rate ──
            lr0=0.015,                # slightly higher w/ cosine annealing
            warmup_epochs=5.0,        # longer warmup for stability
            cos_lr=True,              # cosine annealing — smoother decay
            # ── augmentation (detect-valid only) ──
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,              # slight rotation invariance
            translate=0.1,
            scale=0.7,                # aggressive multi-scale
            perspective=0.0005,       # slight perspective distortion
            fliplr=0.5,
            flipud=0.0,
            mosaic=1.0,
            mixup=0.15,               # label-blending regularization
            cutmix=0.1,              # partial-region occlusion robustness
            close_mosaic=15,          # fine-tune w/o mosaic for last 15 ep
            # ── output ──
            project=str(ROOT / "runs"),
            name="yolo11n_target",
            exist_ok=True,
            plots=True,
            save=True,
            save_period=25,           # checkpoint every 25 epochs
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
        project=str(run_dir),
        name="val",
        exist_ok=True,
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

    # ── copy best weights to models/ ─────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_pt), str(MODELS_DIR / "best.pt"))
    print(f"Copied best.pt → {MODELS_DIR / 'best.pt'}")

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
