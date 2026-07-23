"""Main application entry point for the ATM theft detection project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import config
from database import DatabaseManager
from detect import Detector


def configure_logging() -> None:
    """Set up a basic logger for the application."""
    config.ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "atm_theft.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run the ATM theft detection system")
    parser.add_argument("--source", default=str(config.DEFAULT_SOURCE), help="Image path, video path, or camera index")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="YOLOv8 model to use")
    parser.add_argument("--confidence", type=float, default=config.DEFAULT_CONFIDENCE, help="Detection confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display the annotated feed")
    parser.add_argument("--save-output", action="store_true", help="Save annotated output images")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for video processing")
    return parser


def main() -> None:
    """Entry point for the app."""
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    logger = logging.getLogger("atm_detector.app")
    logger.info("Starting ATM theft detection system")
    DatabaseManager()
    detector = Detector(model_path=args.model, confidence_threshold=args.confidence)
    detector.run(source=args.source, show_window=args.show, save_output=args.save_output, max_frames=args.max_frames)
    logger.info("ATM theft detection system completed")


if __name__ == "__main__":
    main()
