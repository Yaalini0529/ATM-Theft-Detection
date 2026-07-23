"""Behavior analysis module for suspicious activity detection."""

from __future__ import annotations

import time
from typing import List


class BehaviorAnalyzer:
    """Simple behavioral heuristics for suspicious ATM activity."""

    def __init__(self) -> None:
        self.last_centers: list[tuple[float, float]] = []
        self.loiter_start_time: float | None = None
        self.loitering_detected = False

    def analyze(self, detections: List[object], frame_shape: tuple[int, int, int]) -> list[str]:
        """Inspect detections and return a list of behavior-based reasons."""
        reasons: list[str] = []
        height, width = frame_shape[0], frame_shape[1]
        person_boxes = [d for d in detections if d.name.lower() == "person"]

        if len(person_boxes) >= 2:
            reasons.append("Multiple people entering together")

        if person_boxes and self.loiter_start_time is None:
            self.loiter_start_time = time.monotonic()
        if person_boxes and self.loiter_start_time is not None:
            if time.monotonic() - self.loiter_start_time > 30 and not self.loitering_detected:
                reasons.append("Person loitering near ATM")
                self.loitering_detected = True
        elif self.loiter_start_time is not None:
            self.loiter_start_time = None
            self.loitering_detected = False

        if self.last_centers and person_boxes:
            current_centers = [(d.box[0] + d.box[2]) / 2.0 for d in person_boxes]
            max_shift = max(abs(curr - prev) for curr, prev in zip(current_centers, self.last_centers[: len(current_centers)]))
            if max_shift > width * 0.25:
                reasons.append("Sudden running after entering")

        self.last_centers = [(d.box[0] + d.box[2]) / 2.0 for d in person_boxes]

        if person_boxes:
            for detection in person_boxes:
                x1, y1, x2, y2 = detection.box
                if x1 < width * 0.20 and y1 < height * 0.25 and x2 > width * 0.35:
                    reasons.append("Person covering camera")
                    break

        if person_boxes:
            for detection in person_boxes:
                x1, y1, x2, y2 = detection.box
                if x1 < width * 0.25 and y2 > height * 0.70:
                    reasons.append("ATM vandalism")
                    break

        if len(person_boxes) >= 2:
            for index, first in enumerate(person_boxes[:-1]):
                for second in person_boxes[index + 1 :]:
                    x1, y1, x2, y2 = first.box
                    x3, y3, x4, y4 = second.box
                    overlap = not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)
                    if overlap:
                        reasons.append("Aggressive behaviour")
                        break
                if "Aggressive behaviour" in reasons:
                    break

        backpack_boxes = [d for d in detections if d.name.lower() in {"backpack", "mobile phone", "cell phone", "phone"}]
        if backpack_boxes and not person_boxes:
            reasons.append("Object left unattended")

        return reasons
