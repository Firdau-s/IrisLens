"""Export the trained YOLO26 checkpoint to NCNN for GUI inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "cataract-yolo26n" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained YOLO26 weights to NCNN format.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained best.pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Export input size")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.weights}")

    exported = YOLO(str(args.weights)).export(format="ncnn", imgsz=args.imgsz)
    print(f"Model NCNN tersimpan di: {exported}")


if __name__ == "__main__":
    main()
