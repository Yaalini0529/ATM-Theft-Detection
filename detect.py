"""Detection/inference entry point for the ATM theft detection system."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config
from alert import AlertManager
from behavior import BehaviorAnalyzer
from database import DatabaseManager
from threat import ThreatAnalyzer


@dataclass
class DetectionResult:
    """Represents a single object detection result."""

    name: str
    confidence: float
    box: tuple[float, float, float, float]


class Detector:
    """Run YOLOv8 inference, analyze behavior, and emit alerts."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = config.DEFAULT_CONFIDENCE,
        db_manager: Optional[DatabaseManager] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self.logger = logging.getLogger("atm_detector.detect")
        self.confidence_threshold = confidence_threshold
        self.db_manager = db_manager or DatabaseManager()
        self.alert_manager = alert_manager or AlertManager()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.threat_analyzer = ThreatAnalyzer()
        self.model_path = model_path or config.DEFAULT_MODEL
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path: str):
        """Load a YOLO model from disk or download the pretrained weights automatically."""
        try:
            return YOLO(model_path)
        except Exception as exc:  # pragma: no cover - runtime fallback
            self.logger.exception("Unable to load model %s: %s", model_path, exc)
            raise

    def run(
        self,
        source: str,
        show_window: bool = False,
        save_output: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """Run inference on an image, webcam, or video source."""
        frame_source = Path(source)
        if frame_source.exists() and frame_source.is_file():
            frame = cv2.imread(str(frame_source))
            if frame is None:
                self.logger.error("Unable to read image %s", frame_source)
                return
            self._process_frame(frame, source_name=str(frame_source), save_output=save_output)
            return

        capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not capture.isOpened():
            self.logger.warning("Unable to open source %s; falling back to the demo image.", source)
            fallback_image = config.PROJECT_ROOT / "dataset" / "images" / "zidane.jpg"
            if fallback_image.exists():
                frame = cv2.imread(str(fallback_image))
                self._process_frame(frame, source_name=str(fallback_image), save_output=save_output)
            return

        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index += 1
            self._process_frame(frame, source_name=f"frame_{frame_index}", save_output=save_output)
            if show_window:
                cv2.imshow("ATM Theft Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and frame_index >= max_frames:
                break
        capture.release()
        if show_window:
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray, source_name: str, save_output: bool) -> None:
        """Perform detection, behavior analysis, threat scoring, and alerting."""
        start_time = time.time()
        detections = self._detect_objects(frame)
        behavior_reasons = self.behavior_analyzer.analyze(detections, frame.shape)
        threat_level, threat_reasons = self.threat_analyzer.analyze(detections, behavior_reasons)

        for detection in detections:
            self.db_manager.log_detection(detection.name, detection.confidence)

        if threat_level != "LOW" or threat_reasons:
            annotated_frame = self._draw_results(frame, detections, threat_level, threat_reasons)
            image_path, _ = self.alert_manager.create_alert(annotated_frame, threat_level, ", ".join(threat_reasons))
            self.db_manager.log_alert(threat_level, ", ".join(threat_reasons), image_path)
        else:
            annotated_frame = self._draw_results(frame, detections, threat_level, [])

        elapsed_seconds = time.time() - start_time
        self.logger.info(
            "Processed %s in %.2f seconds with threat level %s and %d detections",
            source_name,
            elapsed_seconds,
            threat_level,
            len(detections),
        )

        if save_output and threat_level != "LOW":
            output_path = config.EVIDENCE_DIR / f"{Path(source_name).stem}_{threat_level.lower()}_annotated.jpg"
            cv2.imwrite(str(output_path), annotated_frame)

    def _detect_objects(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run YOLOv8 inference and normalize detections to our local structure."""
        results = self.model(frame, stream=False, conf=self.confidence_threshold, imgsz=640, agnostic_nms=True)
        detections: list[DetectionResult] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                label = str(self.model.names[cls_id]).lower()
                normalized_label = self._normalize_label(label)
                if normalized_label not in config.TARGET_OBJECTS and normalized_label not in {"person"}:
                    continue
                detections.append(
                    DetectionResult(
                        name=normalized_label,
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                    )
                )
        return detections

    def _normalize_label(self, label: str) -> str:
        """Normalize labels so that similar names map to the supported object names."""
        lowered = label.strip().lower()
        aliases = {
            "face mask": "face mask",
            "mask": "face mask",
            "helmet": "helmet",
            "person": "person",
            "motorbike": "motorcycle",
            "motorcycle": "motorcycle",
            "backpack": "backpack",
            "cell phone": "mobile phone",
            "cellphone": "mobile phone",
            "mobile phone": "mobile phone",
            "phone": "mobile phone",
        }
        return aliases.get(lowered, lowered)

    def _draw_results(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        threat_level: str,
        threat_reasons: list[str],
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and threat information on the frame."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            color = (0, 255, 0)
            if threat_level in {"HIGH", "CRITICAL"}:
                color = (0, 0, 255)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                annotated,
                f"{detection.name} {detection.confidence:.2f}",
                (int(x1), max(10, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            f"Threat: {threat_level}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if threat_level in {"HIGH", "CRITICAL"} else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        text = " | ".join(threat_reasons) if threat_reasons else "No suspicious activity detected"
        cv2.putText(
            annotated,
            text[:90],
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated
