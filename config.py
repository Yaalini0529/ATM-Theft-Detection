"""Configuration settings for the ATM theft detection project."""

import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
RUNTIME_ROOT = Path(os.getenv("ATM_RUNTIME_DIR", str(Path(tempfile.gettempdir()) / "atm_theft_detection")))

EVIDENCE_DIR = RUNTIME_ROOT / "evidence"
ALERTS_DIR = RUNTIME_ROOT / "alerts"
DATABASE_DIR = RUNTIME_ROOT / "database"
LOG_DIR = RUNTIME_ROOT / "logs"
DATABASE_PATH = DATABASE_DIR / "atm_theft.db"
BASE_MODEL = PROJECT_ROOT / "yolov8n.pt"
TRAINED_MODEL = MODEL_DIR / "atm_theft_training" / "weights" / "best.pt"
DEFAULT_MODEL = str(TRAINED_MODEL if TRAINED_MODEL.exists() else BASE_MODEL)
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
DEFAULT_SOURCE = str(PROJECT_ROOT / "dataset" / "images" / "zidane.jpg")
DEFAULT_CONFIDENCE = 0.45
FACE_OCCLUSION_CONFIRMATION_FRAMES = 2
FACE_OCCLUSION_OVERLAP_THRESHOLD = 0.18
FACE_OCCLUSION_CONFIDENCE_THRESHOLD = 0.55
FACE_OCCLUSION_FACE_REGION_RATIO = 0.38
TARGET_OBJECTS = (
    "person",
    "helmet",
    "face mask",
    "mask",
    "motorcycle",
    "backpack",
    "mobile phone",
    "cell phone",
    "phone",
)


def ensure_directories() -> dict[str, Path]:
    """Create the folders needed by the application."""
    directories = {
        "project": PROJECT_ROOT,
        "dataset": DATASET_ROOT,
        "models": MODEL_DIR,
        "evidence": EVIDENCE_DIR,
        "alerts": ALERTS_DIR,
        "database": DATABASE_DIR,
        "logs": LOG_DIR,
    }
    for path in directories.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # The app can still run inference if optional runtime storage is unavailable.
            continue
    return directories
