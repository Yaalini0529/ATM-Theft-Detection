"""Detection/inference entry point for the ATM theft detection system."""

from __future__ import annotations

import logging
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
from hand_detector import HandDetector


@dataclass
class DetectionResult:
    """Represents a single object detection result."""

    name: str
    confidence: float
    box: tuple[float, float, float, float]


class Detector:
    """Run YOLO inference, MediaPipe hand detection, behavior analysis, and threat scoring."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = config.DEFAULT_CONFIDENCE,
        db_manager: Optional[DatabaseManager] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:

        self.logger = logging.getLogger(
            "atm_detector.detect"
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.db_manager = (
            db_manager or DatabaseManager()
        )

        self.alert_manager = (
            alert_manager or AlertManager()
        )

        # Existing behavior analyzer
        self.behavior_analyzer = BehaviorAnalyzer()

        # Existing threat analyzer
        # DO NOT modify threat.py
        self.threat_analyzer = ThreatAnalyzer()

        # NEW: MediaPipe hand detector
        self.hand_detector = HandDetector()

        self.model_path = (
            model_path or config.DEFAULT_MODEL
        )

        self.model = self.load_model(
            self.model_path
        )

    # =========================================================
    # LOAD YOLO
    # =========================================================

    def load_model(self, model_path: str):

        try:

            return YOLO(model_path)

        except Exception as exc:

            self.logger.exception(
                "Unable to load model %s: %s",
                model_path,
                exc,
            )

            raise

    # =========================================================
    # RUN
    # =========================================================

    def run(
        self,
        source: str,
        show_window: bool = False,
        save_output: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """Run inference on image, webcam, or video."""

        frame_source = Path(source)

        # -----------------------------------------------------
        # IMAGE
        # -----------------------------------------------------

        if (
            frame_source.exists()
            and frame_source.is_file()
        ):

            frame = cv2.imread(
                str(frame_source)
            )

            if frame is None:

                self.logger.error(
                    "Unable to read image %s",
                    frame_source,
                )

                return

            self._process_frame(
                frame,
                source_name=str(frame_source),
                save_output=save_output,
            )

            return

        # -----------------------------------------------------
        # VIDEO / WEBCAM
        # -----------------------------------------------------

        capture = cv2.VideoCapture(
            int(source)
            if source.isdigit()
            else source
        )

        if not capture.isOpened():

            self.logger.warning(
                "Unable to open source %s",
                source,
            )

            fallback_image = (
                config.PROJECT_ROOT
                / "dataset"
                / "images"
                / "zidane.jpg"
            )

            if fallback_image.exists():

                frame = cv2.imread(
                    str(fallback_image)
                )

                self._process_frame(
                    frame,
                    source_name=str(
                        fallback_image
                    ),
                    save_output=save_output,
                )

            return

        frame_index = 0

        while True:

            success, frame = capture.read()

            if not success:
                break

            frame_index += 1

            # Process frame
            infer_result = self.infer_frame(
                frame,
                source_name=f"frame_{frame_index}",
                save_output=save_output,
            )

            # IMPORTANT:
            # Display the ANNOTATED frame.
            if show_window:

                cv2.imshow(
                    "ATM Theft Detection",
                    infer_result["annotated"],
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    break

            if (
                max_frames
                and frame_index >= max_frames
            ):
                break

        capture.release()

        if show_window:
            cv2.destroyAllWindows()

    # =========================================================
    # PROCESS FRAME
    # =========================================================

    def _process_frame(
        self,
        frame: np.ndarray,
        source_name: str,
        save_output: bool,
    ) -> None:

        start_time = time.time()

        infer_result = self.infer_frame(
            frame,
            source_name=source_name,
            save_output=save_output,
        )

        elapsed_seconds = (
            time.time() - start_time
        )

        self.logger.info(
            "Processed %s in %.2f seconds "
            "with threat level %s, score %d "
            "and %d detections",
            source_name,
            elapsed_seconds,
            infer_result["threat_level"],
            infer_result.get(
                "threat_score",
                0,
            ),
            len(
                infer_result.get(
                    "detections",
                    [],
                )
            ),
        )

    # =========================================================
    # YOLO DETECTION
    # =========================================================

    def _detect_objects(
        self,
        frame: np.ndarray,
    ) -> list[DetectionResult]:
        """Run YOLO inference."""

        results = self.model(
            frame,
            stream=False,
            conf=self.confidence_threshold,
            imgsz=640,
            agnostic_nms=True,
        )

        detections: list[
            DetectionResult
        ] = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                cls_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = [
                    float(value)
                    for value
                    in box.xyxy[0].tolist()
                ]

                label = str(
                    self.model.names[
                        cls_id
                    ]
                ).lower()

                normalized_label = (
                    self._normalize_label(
                        label
                    )
                )

                if (
                    normalized_label
                    not in config.TARGET_OBJECTS
                    and normalized_label
                    != "person"
                ):
                    continue

                detections.append(
                    DetectionResult(
                        name=normalized_label,
                        confidence=confidence,
                        box=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    )
                )

        return detections

    # =========================================================
    # MAIN INFERENCE
    # =========================================================

    def infer_frame(
        self,
        frame: np.ndarray,
        source_name: str = "frame",
        save_output: bool = True,
    ) -> dict:
        """
        Complete detection pipeline.

        YOLO
          ↓
        Person detection

        MediaPipe
          ↓
        Hand detection

        Both
          ↓
        BehaviorAnalyzer

        BehaviorAnalyzer
          ↓
        face_obstructed

        ThreatAnalyzer
          ↓
        CRITICAL / HIGH / MEDIUM / LOW
        """

        # -----------------------------------------------------
        # STEP 1 — YOLO
        # -----------------------------------------------------

        detections = self._detect_objects(
            frame
        )

        # -----------------------------------------------------
        # STEP 2 — MEDIAPIPE
        # -----------------------------------------------------

        hands = self.hand_detector.detect(
            frame
        )

        # -----------------------------------------------------
        # STEP 3 — FACE OBSTRUCTION
        #
        # IMPORTANT:
        # Call this ONLY ONCE per frame.
        # -----------------------------------------------------

        face_obstruction = (
            self.behavior_analyzer
            .analyze_face_obstruction(
                detections,
                frame.shape,
                hands=hands,
                frame=frame,
            )
        )

        # -----------------------------------------------------
        # STEP 4 — OTHER BEHAVIOR
        #
        # analyze() no longer checks face obstruction.
        # -----------------------------------------------------

        behavior_reasons = (
            self.behavior_analyzer.analyze(
                detections,
                frame.shape,
                hands=hands,
                frame=frame,
            )
        )

        # -----------------------------------------------------
        # STEP 5 — THREAT ANALYSIS
        #
        # threat.py remains unchanged.
        # -----------------------------------------------------

        threat_level, threat_reasons, threat_score = (
            self.threat_analyzer.analyze(
                detections,
                behavior_reasons,
                face_obstructed=bool(
                    face_obstruction[
                        "obstructed"
                    ]
                ),
                hand_detected=bool(hands),
            )
        )

        # -----------------------------------------------------
        # STEP 6 — DATABASE
        # -----------------------------------------------------

        for detection in detections:

            self.db_manager.log_detection(
                detection.name,
                detection.confidence,
            )

        # -----------------------------------------------------
        # STEP 7 — DRAW
        # -----------------------------------------------------

        annotated_frame = (
            self._draw_results(
                frame,
                detections,
                threat_level,
                threat_reasons,
                threat_score=threat_score,
                hands=hands,
            )
        )

        image_path = None

        # -----------------------------------------------------
        # STEP 8 — ALERT
        # -----------------------------------------------------

        if (
            threat_level != "LOW"
            or threat_reasons
        ):

            image_path, _ = (
                self.alert_manager.create_alert(
                    annotated_frame,
                    threat_level,
                    ", ".join(
                        threat_reasons
                    ),
                )
            )

            self.db_manager.log_alert(
                threat_level,
                f"score={threat_score}; "
                + ", ".join(
                    threat_reasons
                ),
                image_path,
            )

            if save_output:

                output_path = (
                    config.EVIDENCE_DIR
                    / f"{Path(source_name).stem}_"
                    f"{threat_level.lower()}_annotated.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    annotated_frame,
                )

        # -----------------------------------------------------
        # STEP 9 — RETURN
        # -----------------------------------------------------

        return {
            "annotated": annotated_frame,

            "detections": detections,

            "hands": hands,

            "threat_level": threat_level,

            "threat_reasons": threat_reasons,

            "threat_score": threat_score,

            "image_path": image_path,

            "face_obstruction": (
                face_obstruction[
                    "obstructed"
                ]
            ),

            "face_obstruction_confidence": (
                face_obstruction[
                    "confidence"
                ]
            ),
        }

    # =========================================================
    # LABEL NORMALIZATION
    # =========================================================

    def _normalize_label(
        self,
        label: str,
    ) -> str:

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

        return aliases.get(
            lowered,
            lowered,
        )

    # =========================================================
    # DRAW RESULTS
    # =========================================================

    def _draw_results(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        threat_level: str,
        threat_reasons: list[str],
        threat_score: int = 0,
        hands: list[object] | None = None,
    ) -> np.ndarray:
        """Draw detection and threat information."""

        annotated = frame.copy()

        height, width = (
            annotated.shape[:2]
        )

        # -----------------------------------------------------
        # YOLO BOXES
        # -----------------------------------------------------

        for detection in detections:

            x1, y1, x2, y2 = (
                detection.box
            )

            color = (0, 255, 0)

            if threat_level in {
                "HIGH",
                "CRITICAL",
            }:

                color = (
                    0,
                    0,
                    255,
                )

            cv2.rectangle(
                annotated,
                (
                    int(x1),
                    int(y1),
                ),
                (
                    int(x2),
                    int(y2),
                ),
                color,
                2,
            )

            cv2.putText(
                annotated,
                f"{detection.name} "
                f"{detection.confidence * 100:.1f}%",
                (
                    int(x1),
                    max(
                        10,
                        int(y1) - 5,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # -----------------------------------------------------
        # THREAT LEVEL
        # -----------------------------------------------------

        threat_color = (
            (0, 0, 255)
            if threat_level
            in {
                "HIGH",
                "CRITICAL",
            }
            else
            (0, 255, 0)
        )

        cv2.putText(
            annotated,
            f"Threat: {threat_level} "
            f" Score: {threat_score}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            threat_color,
            2,
            cv2.LINE_AA,
        )

        # -----------------------------------------------------
        # CRITICAL MESSAGE
        # -----------------------------------------------------

        if threat_level == "CRITICAL":

            cv2.putText(
                annotated,
                "CRITICAL THREAT: "
                "FACE OBSTRUCTION DETECTED",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # -----------------------------------------------------
        # REASON
        # -----------------------------------------------------

        text = (
            " | ".join(
                threat_reasons
            )
            if threat_reasons
            else
            "No suspicious activity detected"
        )

        cv2.putText(
            annotated,
            text[:90],
            (
                10,
                height - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated