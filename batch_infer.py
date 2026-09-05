"""Batch inference runner: process all images in the dataset and save annotated evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import logging

import config
from detect import Detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch infer over dataset images and save evidence")
    parser.add_argument("--model", default=str(config.MODEL_DIR / "atm_theft_training" / "weights" / "best.pt"))
    parser.add_argument("--confidence", type=float, default=config.DEFAULT_CONFIDENCE)
    parser.add_argument("--dir", default=str(config.DATASET_ROOT / "train" / "images"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dataset_dir = Path(args.dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset images directory not found: {dataset_dir}")

    detector = Detector(model_path=args.model, confidence_threshold=args.confidence)

    images = sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not images:
        print("No images found to process.")
        return

    print(f"Processing {len(images)} images from {dataset_dir} using model {args.model}")
    processed = 0
    alerts = 0
    for img in images:
        frame = None
        try:
            import cv2

            frame = cv2.imread(str(img))
        except Exception:
            continue

        if frame is None:
            continue

        result = detector.infer_frame(frame, source_name=str(img), save_output=True)
        processed += 1
        if result["threat_level"] != "LOW":
            alerts += 1
            print(f"ALERT {img.name}: {result['threat_level']} score={result['threat_score']} reasons={result['threat_reasons']}")
        else:
            print(f"OK {img.name}: LOW")

    print(f"Done. Processed {processed} images, alerts generated: {alerts}")


if __name__ == "__main__":
    main()
