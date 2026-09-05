"""Configuration settings for the ATM theft detection project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
ALERTS_DIR = PROJECT_ROOT / "alerts"
DATABASE_DIR = PROJECT_ROOT / "database"
LOG_DIR = PROJECT_ROOT / "logs"
DATABASE_PATH = DATABASE_DIR / "atm_theft.db"
DEFAULT_MODEL = "yolov8n.pt"
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
        path.mkdir(parents=True, exist_ok=True)
    return directories
