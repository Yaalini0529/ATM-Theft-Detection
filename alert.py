"""Alert handling and notification module."""

from __future__ import annotations

import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from config import ALERTS_DIR, EVIDENCE_DIR


class AlertManager:
    """Create alert images, save alert text, and trigger alarms."""

    def __init__(self, alerts_dir: Optional[Path] = None, evidence_dir: Optional[Path] = None) -> None:
        self.alerts_dir = Path(alerts_dir or ALERTS_DIR)
        self.evidence_dir = Path(evidence_dir or EVIDENCE_DIR)
        self.logger = logging.getLogger("atm_detector.alert")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def create_alert(self, frame, threat_level: str, reason: str) -> tuple[str, str]:
        """Save a screenshot and a text summary for an alert."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.evidence_dir / f"alert_{timestamp}_{threat_level.lower()}.jpg"
        text_path = self.alerts_dir / f"alert_{timestamp}.txt"

        success = cv2.imwrite(str(image_path), frame)
        if not success:
            self.logger.error("Failed to save alert image to %s", image_path)

        text_path.write_text(
            f"timestamp={datetime.now().isoformat()}\n"
            f"threat_level={threat_level}\n"
            f"reason={reason}\n",
            encoding="utf-8",
        )
        self.logger.info("Saved alert image to %s", image_path)
        self.play_alarm(threat_level)
        return str(image_path), str(text_path)

    def play_alarm(self, threat_level: str) -> None:
        """Emit an audible alert when the threat reaches high or critical."""
        if threat_level not in {"HIGH", "CRITICAL"}:
            return
        if platform.system() == "Windows":
            import winsound

            winsound.MessageBeep()
        else:
            self.logger.info("Alarm sound skipped because the platform does not support the built-in sound hook.")
