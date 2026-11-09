"""Train a YOLO26 detector on the local cataract dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR.parent / "cataract detection.v3i.yolo26"
DATA_YAML = DATASET_DIR / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO26 detector for immature, mature, and normal cataract classes."
    )
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO26 model/weights (default: yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size; use -1 for auto batch sizing")
    parser.add_argument("--device", default=None, help="Device, e.g. 0, cpu, or mps; default lets Ultralytics choose")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--name", default="cataract-yolo26n", help="Training run name")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent checkpoint for this run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config tidak ditemukan: {DATA_YAML}")
    for split in ("train", "valid"):
        images_dir = DATASET_DIR / split / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Folder gambar {split} tidak ditemukan: {images_dir}")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_DIR / "runs"),
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
