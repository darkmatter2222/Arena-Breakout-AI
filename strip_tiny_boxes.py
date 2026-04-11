"""
Strip tiny bounding boxes from YOLO label files.

Any box whose width or height is < MIN_PX pixels (at IMGSZ resolution)
is removed from the label file. The image is kept; if all boxes are
removed the label file becomes empty (valid negative sample).

Prints a summary when done.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"

IMGSZ = 640
MIN_PX = 16  # minimum side length in pixels at IMGSZ


def process_split(split: str) -> tuple[int, int, int]:
    """Returns (files_scanned, files_modified, boxes_removed)."""
    lbl_dir = DATASET / "labels" / split
    if not lbl_dir.exists():
        return 0, 0, 0

    scanned = 0
    modified = 0
    removed = 0

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        scanned += 1
        lines = lbl_file.read_text().strip().splitlines()
        kept = []
        dropped = 0

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                kept.append(line)
                continue
            w_norm, h_norm = float(parts[3]), float(parts[4])
            w_px = w_norm * IMGSZ
            h_px = h_norm * IMGSZ
            if w_px < MIN_PX or h_px < MIN_PX:
                dropped += 1
            else:
                kept.append(line)

        if dropped > 0:
            lbl_file.write_text("\n".join(kept) + ("\n" if kept else ""))
            modified += 1
            removed += dropped

    return scanned, modified, removed


def main():
    total_scanned = 0
    total_modified = 0
    total_removed = 0

    for split in ("train", "val"):
        s, m, r = process_split(split)
        total_scanned += s
        total_modified += m
        total_removed += r
        print(f"  {split}: scanned {s} files, modified {m}, removed {r} tiny boxes")

    print(f"\nTotal: scanned {total_scanned} files, modified {total_modified}, "
          f"removed {total_removed} boxes (any side < {MIN_PX}px at {IMGSZ})")


if __name__ == "__main__":
    main()
