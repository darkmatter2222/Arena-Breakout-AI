<div align="center">

# Arena Breakout AI

### End-to-end YOLO11 computer vision for gameplay capture, AI-assisted annotation, training, dataset QA, and real-time Windows inference

[![GitHub stars](https://img.shields.io/github/stars/darkmatter2222/Arena-Breakout-AI?style=for-the-badge&logo=github)](https://github.com/darkmatter2222/Arena-Breakout-AI/stargazers)
[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-111F68?style=for-the-badge)](https://docs.ultralytics.com/models/yolo11/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?style=for-the-badge&logo=windows)](https://github.com/darkmatter2222/Arena-Breakout-AI)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

**If this project helps your computer-vision experiments, please star the repository. Stars make the project easier for other developers to find.**

</div>

---

## What is Arena Breakout AI?

Arena Breakout AI is an open-source, Windows-first computer-vision pipeline built around **Ultralytics YOLO11**. It covers the complete development loop for a single-class real-time detector:

```text
Gameplay capture → bounding-box annotation → dataset analysis and cleanup
                 → YOLO11 training → evaluation → real-time inference
```

The repository is a practical reference for developers experimenting with:

- Arena Breakout and Arena Breakout: Infinite computer vision
- Real-time YOLO11 object detection
- AI-assisted image annotation
- Local vision-language-model quality scoring
- High-FPS screen capture and inference on Windows
- Dataset balancing, negative samples, and bounding-box quality
- Capture-safe overlays and model-assisted data harvesting

> [!IMPORTANT]
> This is an independent research and educational project. It is not affiliated with, endorsed by, or sponsored by MoreFun Studios, Tencent, Arena Breakout, or Arena Breakout: Infinite. Automated input may violate a game's rules or terms of service. Use the software only where you have permission and accept responsibility for any account or system consequences.

## Why this repository is different

Most object-detection examples stop after training a model. This project includes the complete working loop:

| Stage | Included capability |
|---|---|
| Capture | Center-screen capture with global hotkeys and asynchronous image writers |
| Annotate | Desktop bounding-box GUI with auto-save, undo, filtering, and keyboard-first controls |
| Assist | Cached YOLO predictions, one-click acceptance, and a 4×4 bulk-review workflow |
| VLM QA | Optional OpenAI-compatible local vision model for head-presence and box-quality scoring |
| Clean | EDA, malformed-label checks, overlap detection, and tiny-box removal |
| Train | Reproducible splits, negative downsampling, resume support, evaluation, and model promotion |
| Detect | Real-time inference, target locking, adaptive smoothing, capture-safe status overlay, and data harvesting |

## Architecture

```text
                         ┌──────────────────────┐
                         │      capture.py      │
                         │  600×416 screenshots │
                         └──────────┬───────────┘
                                    │
                                    ▼
                              captures/*.jpg
                                    │
                                    ▼
             ┌────────────────────────────────────────┐
             │              annotate.py               │
             │ manual boxes · YOLO suggestions · VLM  │
             │ filters · bulk review · persistent QA  │
             └───────────────────┬────────────────────┘
                                 │
                                 ▼
                    dataset/{images,labels}/
                                 │
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                 ▼
           eda.py       scrub_overlaps.py   strip_tiny_boxes.py
               └─────────────────┬─────────────────┘
                                 │
                                 ▼
                            train.py
                    splits · train · evaluate
                                 │
                                 ▼
                     runs/yolo11n_target/
                                 │
                                 ▼
                         models/best.pt
                                 │
                                 ▼
                            detect.py
                 live inference · overlay · harvest
```

## Reported training results

These are local experiment results recorded in the repository. They are not independent benchmarks, and the dataset and production weights are not included in Git.

| Run | Dataset | Epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 9,128 images | 150 | 0.891 | 0.550 | 0.902 | 0.834 |
| v2 | 10,145 images | 300 | 0.916 | **0.564** | 0.904 | **0.875** |
| v3 | 10,145 images | 300 | **0.923** | 0.557 | **0.915** | 0.867 |

The experiments were run with YOLO11n on an NVIDIA RTX 5090. See [`AGENTS.md`](AGENTS.md) for the tracked hyperparameter history and implementation notes.

## Requirements

- Windows 10 or Windows 11
- Python 3.13 or newer
- NVIDIA GPU and CUDA recommended
- Administrator elevation for the global-input capture and real-time detection scripts
- Numpad keys for the default hotkey layout

## Quick start

### 1. Clone and create an environment

```powershell
git clone https://github.com/darkmatter2222/Arena-Breakout-AI.git
cd Arena-Breakout-AI

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build appropriate for your NVIDIA driver and CUDA environment if the default package does not provide GPU support.

### 2. Capture training images

```powershell
python capture.py
```

| Input | Action |
|---|---|
| Numpad 5 | Arm or disarm capture |
| Hold left mouse button | Capture at the primary collection rate |
| Hold right mouse button | Capture at the secondary collection rate |
| Numpad 6 | Toggle automatic capture |
| Escape | Exit |

Images are written to `captures/`. JPEG encoding and disk I/O run through a writer queue so they do not block the capture loop.

### 3. Annotate the dataset

```powershell
python annotate.py
```

| Input | Action |
|---|---|
| Left-drag | Draw a bounding box |
| Right-click a box | Delete that box |
| A / D or arrow keys | Previous or next image |
| Space | Save an empty label as a negative sample |
| Z | Undo the most recent box |
| C | Clear all boxes |
| Enter | Accept all YOLO prediction boxes |
| Shift | Accept the prediction under the cursor |
| B | Open 4×4 bulk annotation mode |
| Ctrl + drag | Add multiple boxes without auto-advance |

The tool automatically:

- Moves completed images into `dataset/images/`
- Writes YOLO labels into `dataset/labels/`
- Persists annotation progress
- Caches model predictions from `models/best.pt`
- Supports confidence and quality filters
- Offers optional local VLM scoring for pre-annotation and annotation QA

### 4. Inspect and clean labels

```powershell
python eda.py
python scrub_overlaps.py
python strip_tiny_boxes.py
```

| Script | Purpose |
|---|---|
| `eda.py` | Reports counts, class distribution, box geometry, malformed labels, out-of-bounds boxes, tiny boxes, and overlaps |
| `scrub_overlaps.py` | Moves images with overlapping boxes into `dataset/archive/` instead of deleting them |
| `strip_tiny_boxes.py` | Removes boxes with a side smaller than the configured pixel threshold |

Review the output before applying in-place cleanup to valuable data.

### 5. Train and evaluate YOLO11n

```powershell
python train.py
```

The training script:

1. Finds positive and negative samples.
2. Creates reproducible train and validation split files with seed 42.
3. Keeps the full validation distribution.
4. Downsamples training negatives to the configured target rate.
5. Resumes an incomplete run when possible.
6. Archives a completed run before starting another.
7. Runs validation and saves plots, JSON predictions, and per-image predictions.
8. Copies the selected `best.pt` into `models/best.pt`.

| Setting | Current value |
|---|---:|
| Base model | `yolo11n.pt` |
| Input size | 800 |
| Batch size | 32 |
| Epochs | 300 |
| Patience | 50 |
| Validation split | 15% |
| Training negative target | 35% |
| Seed | 42 |

### 6. Run real-time inference

Place a trained model at:

```text
models/best.pt
```

Then run:

```powershell
python detect.py
```

| Input | Action |
|---|---|
| Numpad 2 | Toggle detection tick audio |
| Numpad 3 | Toggle adaptive detection-frame harvesting |
| Numpad 5 | Arm or disarm cursor movement |
| Numpad 6 | Toggle frame capture |
| Hold right mouse button | Engage movement toward the selected detection while armed |

The runtime includes confidence filtering, nearest-target selection, lock grace, exponential movement easing, box-size damping, bounded steps, optional frame harvesting, and a Windows overlay excluded from supported screen-capture paths.

## Optional local VLM annotation QA

`annotate.py` can send images to an OpenAI-compatible vision endpoint to:

1. Score whether a target head is likely present.
2. Score the quality of model-proposed or manually drawn boxes.

Configure your own endpoint with environment variables:

```powershell
$env:ARENA_AI_LLM_BASE_URL = "http://127.0.0.1:8001"
$env:ARENA_AI_LLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
python annotate.py
```

Do not commit private hostnames, LAN addresses, tokens, or credentials. The VLM feature is optional; manual annotation and local YOLO prediction overlays work without it.

## Repository layout

```text
.
├── capture.py              # Screen capture and asynchronous image writing
├── annotate.py             # Annotation GUI, YOLO assistance, bulk mode, VLM QA
├── train.py                # Split generation, YOLO training, evaluation, promotion
├── detect.py               # Live inference, overlay, cursor movement, harvesting
├── eda.py                  # Dataset statistics and label-quality analysis
├── scrub_overlaps.py       # Archive samples containing overlapping boxes
├── strip_tiny_boxes.py     # Remove boxes below a configured pixel size
├── requirements.txt
├── AGENTS.md               # Detailed architecture and experiment context
├── LICENSE                 # Apache License 2.0
├── captures/               # Ignored raw captures
├── dataset/                # Ignored training images, labels, splits, and caches
├── models/                 # Ignored model weights
├── runs/                   # Ignored training artifacts
├── harvest/                # Ignored runtime samples
└── logs/                   # Ignored runtime logs
```

## Training data and model weights

The training dataset and model weights are intentionally excluded from this repository because of their size and distribution considerations.

- **Training data:** available upon request at the maintainer's discretion.
- **Hugging Face:** a future dataset or model release is being considered, but no publication date is committed.
- **Model weights:** train locally with `train.py`, then place the resulting model at `models/best.pt`.

Open a GitHub issue describing the research or development use case when requesting access to the training data.

## Security and privacy

The repository ignores datasets, captures, model weights, logs, local environments, and `.env` files. Before opening a pull request, verify that changes do not include:

- API keys, passwords, tokens, cookies, or private certificates
- Public IP addresses, private hostnames, or machine-specific LAN endpoints
- Personal file-system paths or usernames
- Gameplay captures or datasets you do not have permission to redistribute
- Proprietary model weights or third-party assets with incompatible licenses

See [`SECURITY.md`](SECURITY.md) for reporting guidance.

## Roadmap

- Publish a reproducible dataset card and sample subset
- Release selected weights through Hugging Face
- Replace source-level constants with command-line or file-based configuration
- Add automated tests for label conversion, IoU, and movement math
- Add benchmarks for end-to-end capture, inference, and post-processing latency
- Add ONNX and TensorRT export paths
- Add configurable hotkeys, capture region, and model thresholds
- Publish a demonstration and build walkthrough on YouTube

## Contributing

Issues and pull requests are welcome for annotation workflow improvements, dataset-quality tooling, YOLO experiments, Windows capture reliability, performance profiling, documentation, and reproducibility.

Keep changes focused, explain the validation performed, and never include private data or proprietary game assets.

## Support the project

Found this project through Arena Breakout AI, YOLO11, game computer vision, object detection, or local AI research?

1. **Star the repository** so other developers can discover it.
2. Open an issue with reproducible feedback or benchmark results.
3. Share a link to the project rather than redistributing stale copies.
4. Contribute improvements through a pull request.

## License

Source code is licensed under the [Apache License 2.0](LICENSE).

Arena Breakout, Arena Breakout: Infinite, MoreFun Studios, Tencent, and all related marks and assets belong to their respective owners. The repository license does not grant rights to third-party game assets, screenshots, datasets, or trademarks.
