"""
Full EDA of YOLO label data for dataset/.
Outputs statistics, distributions, and flags problem labels.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import statistics

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"

def parse_labels():
    """Parse all label files in the flat dataset. Returns list of dicts."""
    label_dir = DATASET_DIR / "labels"
    image_dir = DATASET_DIR / "images"
    results = []
    for lbl_file in sorted(label_dir.glob("*.txt")):
        stem = lbl_file.stem
        # Check if corresponding image exists
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = image_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        
        boxes = []
        with open(lbl_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    boxes.append({"error": f"bad format line {line_num}: {line}"})
                    continue
                cls_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append({"cls": cls_id, "xc": xc, "yc": yc, "w": w, "h": h})
        
        results.append({
            "file": lbl_file.name,
            "stem": stem,
            "has_image": img_path is not None,
            "boxes": boxes,
        })
    return results


def iou(b1, b2):
    """Compute IoU between two YOLO normalized boxes (xc,yc,w,h)."""
    x1_min = b1["xc"] - b1["w"] / 2
    x1_max = b1["xc"] + b1["w"] / 2
    y1_min = b1["yc"] - b1["h"] / 2
    y1_max = b1["yc"] + b1["h"] / 2

    x2_min = b2["xc"] - b2["w"] / 2
    x2_max = b2["xc"] + b2["w"] / 2
    y2_min = b2["yc"] - b2["h"] / 2
    y2_max = b2["yc"] + b2["h"] / 2

    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)

    if ix_max <= ix_min or iy_max <= iy_min:
        return 0.0

    inter = (ix_max - ix_min) * (iy_max - iy_min)
    area1 = b1["w"] * b1["h"]
    area2 = b2["w"] * b2["h"]
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def analyze():
    data = parse_labels()

    print(f"\n{'='*70}")
    print(f"  DATASET  ({len(data)} label files)")
    print(f"{'='*70}")

    # Basic counts
    total_boxes = 0
    empty_files = 0
    missing_images = 0
    format_errors = 0
    boxes_per_image = []
    all_widths = []
    all_heights = []
    all_areas = []
    all_xc = []
    all_yc = []
    all_aspect_ratios = []
    class_counts = defaultdict(int)

    # Problem detection
    oob_boxes = []          # out-of-bounds
    tiny_boxes = []         # very small
    huge_boxes = []         # very large
    overlapping_pairs = []  # IoU > 0 between boxes in same image
    high_overlap_files = [] # files with IoU > 0.3

    for entry in data:
        if not entry["has_image"]:
            missing_images += 1

        valid_boxes = [b for b in entry["boxes"] if "error" not in b]
        error_boxes = [b for b in entry["boxes"] if "error" in b]
        format_errors += len(error_boxes)

        if len(valid_boxes) == 0:
            empty_files += 1

        total_boxes += len(valid_boxes)
        boxes_per_image.append(len(valid_boxes))

        for b in valid_boxes:
            class_counts[b["cls"]] += 1
            all_widths.append(b["w"])
            all_heights.append(b["h"])
            all_areas.append(b["w"] * b["h"])
            all_xc.append(b["xc"])
            all_yc.append(b["yc"])
            if b["h"] > 0:
                all_aspect_ratios.append(b["w"] / b["h"])

            # Out-of-bounds check
            x_min = b["xc"] - b["w"] / 2
            x_max = b["xc"] + b["w"] / 2
            y_min = b["yc"] - b["h"] / 2
            y_max = b["yc"] + b["h"] / 2
            if x_min < -0.001 or y_min < -0.001 or x_max > 1.001 or y_max > 1.001:
                oob_boxes.append((entry["file"], b))

            # Tiny boxes (area < 0.0005 = ~0.7% of image side)
            area = b["w"] * b["h"]
            if area < 0.0005:
                tiny_boxes.append((entry["file"], b, area))

            # Huge boxes (area > 0.5 = more than 50% of image)
            if area > 0.5:
                huge_boxes.append((entry["file"], b, area))

        # Overlap detection within the same image
        for i in range(len(valid_boxes)):
            for j in range(i + 1, len(valid_boxes)):
                iou_val = iou(valid_boxes[i], valid_boxes[j])
                if iou_val > 0:
                    overlapping_pairs.append((entry["file"], iou_val, valid_boxes[i], valid_boxes[j]))
                if iou_val > 0.3:
                    high_overlap_files.append((entry["file"], iou_val))

    # ── Print stats ──
    print(f"\n  Total label files  : {len(data)}")
    print(f"  Total boxes        : {total_boxes}")
    print(f"  Empty label files  : {empty_files} (no boxes = negative samples)")
    print(f"  Missing images     : {missing_images}")
    print(f"  Format errors      : {format_errors}")
    print(f"  Classes found      : {dict(class_counts)}")

    print(f"\n  ── Boxes per image ──")
    if boxes_per_image:
        print(f"    Mean   : {statistics.mean(boxes_per_image):.2f}")
        print(f"    Median : {statistics.median(boxes_per_image):.1f}")
        print(f"    Min    : {min(boxes_per_image)}")
        print(f"    Max    : {max(boxes_per_image)}")
        # Distribution
        bpi_dist = defaultdict(int)
        for n in boxes_per_image:
            bpi_dist[n] += 1
        print(f"    Distribution:")
        for k in sorted(bpi_dist.keys()):
            pct = bpi_dist[k] / len(boxes_per_image) * 100
            print(f"      {k} boxes: {bpi_dist[k]} images ({pct:.1f}%)")

    print(f"\n  ── Box size distribution (normalized) ──")
    if all_widths:
        print(f"    Width  — min: {min(all_widths):.6f}  max: {max(all_widths):.6f}  "
              f"mean: {statistics.mean(all_widths):.6f}  median: {statistics.median(all_widths):.6f}")
        print(f"    Height — min: {min(all_heights):.6f}  max: {max(all_heights):.6f}  "
              f"mean: {statistics.mean(all_heights):.6f}  median: {statistics.median(all_heights):.6f}")
        print(f"    Area   — min: {min(all_areas):.8f}  max: {max(all_areas):.6f}  "
              f"mean: {statistics.mean(all_areas):.6f}  median: {statistics.median(all_areas):.6f}")

    print(f"\n  ── Aspect ratio (w/h) ──")
    if all_aspect_ratios:
        print(f"    Min    : {min(all_aspect_ratios):.4f}")
        print(f"    Max    : {max(all_aspect_ratios):.4f}")
        print(f"    Mean   : {statistics.mean(all_aspect_ratios):.4f}")
        print(f"    Median : {statistics.median(all_aspect_ratios):.4f}")

    print(f"\n  ── Center position (xc, yc) ──")
    if all_xc:
        print(f"    xc — min: {min(all_xc):.4f}  max: {max(all_xc):.4f}  mean: {statistics.mean(all_xc):.4f}")
        print(f"    yc — min: {min(all_yc):.4f}  max: {max(all_yc):.4f}  mean: {statistics.mean(all_yc):.4f}")

    # Size buckets (in terms of normalized area)
    print(f"\n  ── Size buckets ──")
    if all_areas:
        buckets = {
            "Tiny   (area < 0.001)": 0,
            "Small  (0.001–0.01)":   0,
            "Medium (0.01–0.05)":    0,
            "Large  (0.05–0.2)":     0,
            "Huge   (> 0.2)":        0,
        }
        for a in all_areas:
            if a < 0.001:
                buckets["Tiny   (area < 0.001)"] += 1
            elif a < 0.01:
                buckets["Small  (0.001–0.01)"] += 1
            elif a < 0.05:
                buckets["Medium (0.01–0.05)"] += 1
            elif a < 0.2:
                buckets["Large  (0.05–0.2)"] += 1
            else:
                buckets["Huge   (> 0.2)"] += 1
        for label, count in buckets.items():
            pct = count / len(all_areas) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")

    # Pixel-equivalent sizes assuming 640x640 input
    print(f"\n  ── Pixel-equivalent sizes at 640x640 imgsz ──")
    if all_widths:
        px_widths = [w * 640 for w in all_widths]
        px_heights = [h * 640 for h in all_heights]
        px_areas = [w * h * 640 * 640 for w, h in zip(all_widths, all_heights)]
        print(f"    Width  (px) — min: {min(px_widths):.1f}  max: {max(px_widths):.1f}  mean: {statistics.mean(px_widths):.1f}")
        print(f"    Height (px) — min: {min(px_heights):.1f}  max: {max(px_heights):.1f}  mean: {statistics.mean(px_heights):.1f}")
        print(f"    Area   (px²) — min: {min(px_areas):.1f}  max: {max(px_areas):.1f}")

        # Count boxes smaller than various pixel thresholds at 640
        for thresh in [4, 8, 12, 16, 20]:
            count = sum(1 for w, h in zip(px_widths, px_heights) if w < thresh or h < thresh)
            pct = count / len(px_widths) * 100
            print(f"    Boxes with any side < {thresh}px: {count} ({pct:.1f}%)")

    # ── Problems ──
    print(f"\n  ── PROBLEMS ──")
    print(f"    Out-of-bounds boxes          : {len(oob_boxes)}")
    if oob_boxes[:5]:
        for f, b in oob_boxes[:5]:
            print(f"      {f}: xc={b['xc']:.4f} yc={b['yc']:.4f} w={b['w']:.4f} h={b['h']:.4f}")
        if len(oob_boxes) > 5:
            print(f"      ... and {len(oob_boxes)-5} more")

    print(f"    Tiny boxes (area < 0.0005)   : {len(tiny_boxes)}")
    if tiny_boxes[:5]:
        for f, b, a in tiny_boxes[:5]:
            print(f"      {f}: w={b['w']:.6f} h={b['h']:.6f} area={a:.8f}")
        if len(tiny_boxes) > 5:
            print(f"      ... and {len(tiny_boxes)-5} more")

    print(f"    Huge boxes (area > 0.5)      : {len(huge_boxes)}")
    if huge_boxes[:5]:
        for f, b, a in huge_boxes[:5]:
            print(f"      {f}: w={b['w']:.6f} h={b['h']:.6f} area={a:.6f}")

    print(f"    Overlapping box pairs (IoU>0): {len(overlapping_pairs)}")
    print(f"    High overlap pairs (IoU>0.3) : {len(high_overlap_files)}")
    if high_overlap_files:
        # Deduplicate to unique filenames
        unique_files = sorted(set(f for f, _ in high_overlap_files))
        print(f"    Unique files with IoU>0.3   : {len(unique_files)}")
        for f, iou_val in sorted(high_overlap_files, key=lambda x: -x[1])[:10]:
            print(f"      {f}: IoU={iou_val:.4f}")
        if len(high_overlap_files) > 10:
            print(f"      ... and {len(high_overlap_files)-10} more pairs")

    # IoU distribution
    if overlapping_pairs:
        iou_vals = [x[1] for x in overlapping_pairs]
        print(f"\n    ── Overlap IoU distribution ──")
        for thresh_low, thresh_high, label in [
            (0.0, 0.1, "0.0–0.1 (touching)"),
            (0.1, 0.3, "0.1–0.3 (partial)"),
            (0.3, 0.5, "0.3–0.5 (significant)"),
            (0.5, 0.7, "0.5–0.7 (heavy)"),
            (0.7, 1.01, "0.7–1.0 (near-duplicate)"),
        ]:
            count = sum(1 for v in iou_vals if thresh_low < v <= thresh_high)
            print(f"      {label}: {count}")


if __name__ == "__main__":
    analyze()
