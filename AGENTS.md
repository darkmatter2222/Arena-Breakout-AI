# Vision Pipeline — Agent Context

## What This Project Does

Single-class YOLO11n object detection pipeline. One class: **target** (class_id=0).
Full lifecycle: screen capture → bbox annotation → model training → real-time inference.

## Tech Stack

- **Model**: YOLO11n (Ultralytics nano, ~5 MB)
- **Framework**: Ultralytics 8.4.x, PyTorch 2.x, CUDA 12.8
- **Python**: 3.13+ with venv (`venv/`)
- **GPU**: RTX 5090 (32 GB VRAM) — batch 32 at 800 imgsz
- **OS**: Windows (ctypes/SendInput in detect.py, mss for DXGI capture)

## Dataset Layout

Flat structure — no train/val subdirs. Splits are text files.

```
dataset/
├── images/          # All .jpg images (flat)
├── labels/          # YOLO .txt labels (flat, matching image stems)
├── splits/          # train.txt / val.txt (one relative path per line)
├── archive/         # Excluded data (images/ + labels/)
└── data.yaml        # Points to splits/*.txt
```

YOLO label format: `<class_id> <xc> <yc> <w> <h>` (normalized 0–1).
Labels auto-resolve: Ultralytics replaces `images/` → `labels/` and `.jpg` → `.txt`.

## Key Conventions

- **Single class only**: class_id=0, name="target". Do not add multi-class logic.
- **Flat dataset**: images and labels live directly in `dataset/images/` and `dataset/labels/`. No subdirectories.
- **Text-file splits**: `train.py` writes `dataset/splits/train.txt` and `val.txt`. Paths inside are relative to `dataset/` (e.g., `images/filename.jpg`).
- **Models directory**: production weights live in `models/best.pt`. `train.py` auto-copies after training.
- **No file-moving splits**: unlike some YOLO pipelines, files are never physically moved between train/val folders.
- **Archive, don't delete**: preprocessing scripts move problem data to `dataset/archive/`, never delete.
- **Seed 42**: all random operations use seed=42 for reproducibility.
- **600×416 captures**: screen captures are center-cropped to 600×416, then resized to 800×800 by YOLO during training/inference.

## Script Responsibilities

| Script | Purpose | Reads From | Writes To |
|---|---|---|---|
| `capture.py` | Screen capture | Monitor | `captures/`, `logs/` |
| `annotate.py` | Bbox annotation GUI | `captures/`, `models/best.pt` | `dataset/images/`, `dataset/labels/` |
| `train.py` | Training + evaluation | `dataset/` | `runs/yolo11n_target/`, `models/best.pt`, `dataset/splits/` |
| `detect.py` | Live inference | `models/best.pt` | `harvest/` (optional) |
| `eda.py` | Label EDA | `dataset/labels/`, `dataset/images/` | stdout |
| `scrub_overlaps.py` | Archive overlapping boxes | `dataset/images/`, `dataset/labels/` | `dataset/archive/` |
| `strip_tiny_boxes.py` | Remove tiny boxes | `dataset/labels/` | `dataset/labels/` (in-place) |

## Training Details

- `train.py` auto-resumes incomplete runs from `runs/yolo11n_target/weights/last.pt`
- Completed runs are archived to `runs/yolo11n_target_<timestamp>/` before starting fresh
- Evaluation saves COCO JSON predictions and per-image .txt files inside `runs/yolo11n_target/val/`
- F1-optimal confidence threshold is printed after evaluation

### Current Hyperparameters (v4 — 2026-04-12)

| Param | v1 | v2 | v3 | v4 | Rationale (v4) |
|---|---|---|---|---|---|
| imgsz | 640 | 800 | 800 | 800 | — |
| batch | 16 | 32 | 32 | 32 | — |
| epochs | 150 | 300 | 300 | 300 | — |
| patience | 30 | 50 | 50 | 50 | — |
| cos_lr | off | on | on | on | — |
| lr0 | 0.01 | 0.01 | 0.015 | 0.015 | Kept from v3 |
| warmup_epochs | 3 | 3 | 5 | 5 | Kept from v3 |
| box | 7.5 | 7.5 | 8.5 | 7.5 | **Reverted** — v3 regressed mAP@0.5:0.95 |
| dfl | 1.5 | 1.5 | 2.0 | 1.5 | **Reverted** — v3 regressed mAP@0.5:0.95 |
| cls | 0.5 | 0.5 | 0.3 | 0.5 | **Reverted** — v3 regressed mAP@0.5:0.95 |
| scale | 0.5 | 0.7 | 0.7 | 0.7 | — |
| degrees | 0 | 5 | 5 | 5 | — |
| mixup | 0.1 | 0.15 | 0.15 | 0.15 | — |
| cutmix | 0 | 0 | 0.1 | 0.1 | Detect-valid occlusion augmentation |
| perspective | 0 | 0 | 0.0005 | 0.0005 | Mild perspective distortion |
| close_mosaic | 10 | 15 | 15 | 15 | — |
| neg_rate | ~53% | ~53% | ~53% | **35%** | Train-only downsample, val untouched |
| train_imgs | ~8,624 | ~8,624 | ~8,624 | ~6,236 | Fewer negatives → denser positive signal |

**v4 strategy**: Revert v3 loss weights (which regressed mAP@0.5:0.95 from 0.564→0.557).
Instead, attack data composition: downsample training negatives from 53% to 35% so the
model sees more positive examples per epoch. Val split keeps full distribution for fair eval.
Kept v3's beneficial augmentations (cutmix, perspective, lr0, warmup).

### Training History

| Run | Dataset | Epochs | mAP@0.5 | mAP@0.5:0.95 | P | R | Notes |
|---|---|---|---|---|---|---|---|
| v1 | 9,128 imgs | 150/150 | 0.891 | 0.550 | 0.902 | 0.834 | batch=16, imgsz=640, plateaued ~ep70 |
| v2 | 10,145 imgs | 300/300 | 0.916 | 0.564 | 0.904 | 0.875 | +11% data, copy_paste/erasing silently ignored |
| v3 | 10,145 imgs | 300/300 | 0.923 | 0.557 | 0.915 | 0.867 | Loss rebalancing hurt mAP@0.5:0.95; better calibration (F1@0.305) |
| v4 | 6,236 train | 300 (running) | — | — | — | — | Reverted losses, neg downsample 53%→35% |

## Things to Watch Out For

- `detect.py` requires Administrator (UAC elevation for SendInput)
- `annotate.py` inference cache (`dataset/.inference_cache.json`) auto-invalidates when model path changes
- When adding images, always run `annotate.py` to create labels — then optionally `scrub_overlaps.py` and `strip_tiny_boxes.py` before training
- Dataset now has 10,145 images and 10,145 labels (perfect 1:1 after new annotations)
- `detect.py` uses PyAudio streaming engine for tick sounds (replaces winsound)
