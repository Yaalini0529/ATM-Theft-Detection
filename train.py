"""Training entry point for fine-tuning YOLOv8 on ATM theft detection data."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import config

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency for environments without GPU/torch support
    YOLO = None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser."""
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for ATM theft detection")
    parser.add_argument("--data", default=str(config.DATASET_ROOT / "data.yaml"), help="Path to the YAML dataset file")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Base YOLO model to use")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--save-period", type=int, default=1, help="Save weights every N epochs")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the dataset and exit without training")
    return parser

def normalize_label(name: str) -> str:
    """Map known annotation names to YOLO-compatible class names."""
    normalized = name.strip().lower()
    if "helmet" in normalized:
        return "helmet"
    if "mask" in normalized:
        return "face mask"
    return normalized


def convert_xml_annotations(
    annotation_paths: list[Path],
    output_dir: Path,
    image_dir: Path | None = None,
    class_map: dict[str, int] | None = None,
) -> list[Path]:
    """Convert VOC-style XML labels into YOLO txt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for annotation_path in annotation_paths:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        filename = (root.findtext("filename") or "").strip()
        if not filename:
            continue

        image_path = image_dir / filename if image_dir is not None else None
        if image_path is not None and not image_path.exists():
            continue

        width = int((root.find("size").findtext("width") or "0").strip()) if root.find("size") is not None else 0
        height = int((root.find("size").findtext("height") or "0").strip()) if root.find("size") is not None else 0
        if width <= 0 or height <= 0:
            continue

        lines: list[str] = []
        for obj in root.findall("object"):
            name = normalize_label(obj.findtext("name") or "")
            if not name:
                continue

            if class_map is None:
                # default mapping if none provided
                class_map = {"helmet": 0, "face mask": 1}

            if name not in class_map:
                continue

            class_id = class_map[name]

            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            xmin = float((bbox.findtext("xmin") or "0").strip())
            ymin = float((bbox.findtext("ymin") or "0").strip())
            xmax = float((bbox.findtext("xmax") or "0").strip())
            ymax = float((bbox.findtext("ymax") or "0").strip())
            if xmax <= xmin or ymax <= ymin:
                continue

            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_width = max(0.0, min(1.0, box_width))
            box_height = max(0.0, min(1.0, box_height))
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        label_path = output_dir / f"{Path(filename).stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        written_paths.append(label_path)

    return written_paths


def prepare_dataset(dataset_root: Path = config.DATASET_ROOT, annotations_dir: Path | None = None) -> Path:
    """Create YOLO-ready train/val/test folders from the XML dataset."""
    print(f"Preparing dataset under {dataset_root} from {annotations_dir or config.PROJECT_ROOT / 'annotations'}...")
    annotations_dir = annotations_dir or config.PROJECT_ROOT / "annotations"
    image_root = dataset_root / "images"
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    test_dir = dataset_root / "test"

    print("Creating dataset folders...")
    for folder in (train_dir / "images", train_dir / "labels", valid_dir / "images", valid_dir / "labels", test_dir / "images", test_dir / "labels"):
        folder.mkdir(parents=True, exist_ok=True)

    annotation_paths = sorted(annotations_dir.glob("*.xml"))
    image_paths = [path for path in image_root.glob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    print(f"Found {len(annotation_paths)} annotation files and {len(image_paths)} image files")

    annotated_images: list[tuple[Path, Path]] = []
    for annotation_path in annotation_paths:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        filename = (root.findtext("filename") or "").strip()
        if not filename:
            continue

        image_path = image_root / filename
        if image_path.exists() and image_path in image_paths:
            annotated_images.append((image_path, annotation_path))

    if not annotated_images:
        raise FileNotFoundError(f"No valid XML annotations were found in {annotations_dir}")

    print(f"Found {len(annotated_images)} annotated images; splitting into train/valid sets...")
    split_index = max(1, int(len(annotated_images) * 0.9))
    for index, (image_path, annotation_path) in enumerate(annotated_images):
        destination_dir = valid_dir if index >= split_index else train_dir
        destination_image = destination_dir / "images" / image_path.name
        shutil.copy2(image_path, destination_image)

        output_dir = destination_dir / "labels"
        print(f"Processing image {index + 1}/{len(annotated_images)} -> {destination_dir.name}")
        convert_xml_annotations([annotation_path], output_dir=output_dir, image_dir=image_root)

    for folder in (test_dir / "images", test_dir / "labels"):
        folder.mkdir(parents=True, exist_ok=True)

    data_yaml_path = dataset_root / "data.yaml"
    data_yaml_path.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "nc: 1",
                "names: ['helmet']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Dataset preparation complete. YAML written to {data_yaml_path}")
    return data_yaml_path


def main() -> None:
    """Run training if a valid dataset configuration is present."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    args = build_arg_parser().parse_args()
    config.ensure_directories()

    print(f"Using dataset config: {args.data}")
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset config not found at {data_path}; preparing dataset...")
        data_path = prepare_dataset(config.DATASET_ROOT)
    else:
        print(f"Using existing dataset config at {data_path}")

    if args.prepare_only:
        print(f"Dataset prepared successfully at {data_path}")
        return

    if YOLO is None:
        raise RuntimeError("Ultralytics/YOLO is not available in this environment; install the training dependencies first.")

    print(f"Starting training with model '{args.model}' for {args.epochs} epochs")
    print(f"Image size: {args.imgsz}, batch size: {args.batch}")
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        save_period=args.save_period,
        project=str(config.MODEL_DIR),
        name="atm_theft_training",
        exist_ok=True,
    )
    print("Training run completed.")


if __name__ == "__main__":
    main()
