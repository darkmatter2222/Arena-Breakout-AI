# Vision Pipeline — Agent Context

## What This Project Does

Single-class YOLO11n object detection pipeline. One class: **target** (class_id=0).
Full lifecycle: screen capture → bbox annotation → model training → real-time inference.

## Tech Stack

- **Model**: YOLO11n (Ultralytics nano, ~5 MB)
- **Framework**: Ultralytics 8.4.x, PyTorch 2.x, CUDA 12.8
- **Python**: 3.13+ with venv (`venv/`)
- **GPU**: RTX 5090 (32 GB VRAM) — batch 16 at 640 imgsz
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
- **600×416 captures**: screen captures are center-cropped to 600×416, then resized to 640×640 by YOLO during training/inference.

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

## Things to Watch Out For

- `detect.py` requires Administrator (UAC elevation for SendInput)
- `annotate.py` inference cache (`dataset/.inference_cache.json`) auto-invalidates when model path changes
- When adding images, always run `annotate.py` to create labels — then optionally `scrub_overlaps.py` and `strip_tiny_boxes.py` before training
- The 2-label-file discrepancy (9130 labels vs 9128 images) is from orphaned labels after archiving — harmless
