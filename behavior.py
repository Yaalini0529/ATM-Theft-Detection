"""Behavior analysis module for suspicious activity detection."""

from __future__ import annotations

import time
from typing import List

import config


class BehaviorAnalyzer:
    """Analyze detected objects and hand position for suspicious behavior."""

    def __init__(self) -> None:
        self.last_centers: list[float] = []

        self.loiter_start_time: float | None = None
        self.loitering_detected = False

        # Face obstruction state
        self.face_obstruction_hits = 0
        self.face_obstruction_confirmed = False
        self.face_obstruction_confidence = 0.0

    # ---------------------------------------------------------
    # FACE REGION
    # ---------------------------------------------------------

    def _person_face_region(
        self,
        detection: object,
    ) -> tuple[float, float, float, float]:
        """
        Estimate the face region from a YOLO person bounding box.

        We use the upper portion of the person's bounding box.
        """

        x1, y1, x2, y2 = detection.box

        width = x2 - x1
        height = y2 - y1

        face_width = max(
            width * config.FACE_OCCLUSION_FACE_REGION_RATIO,
            24.0,
        )

        face_height = max(
            height * 0.28,
            18.0,
        )

        center_x = (x1 + x2) / 2.0

        face_x1 = max(
            x1,
            center_x - face_width / 2.0,
        )

        face_x2 = min(
            x2,
            center_x + face_width / 2.0,
        )

        face_y1 = y1 + max(
            0.02 * height,
            6.0,
        )

        face_y2 = y1 + min(
            0.38 * height,
            face_height + 10.0,
        )

        return (
            face_x1,
            face_y1,
            face_x2,
            face_y2,
        )

    # ---------------------------------------------------------
    # HAND POINT EXTRACTION
    # ---------------------------------------------------------

    def _get_hand_points(
        self,
        hand: object,
    ) -> list[tuple[float, float]]:
        """
        Convert a MediaPipe hand result into pixel coordinates.

        Supports the dictionary format returned by our HandDetector.
        """

        points: list[tuple[float, float]] = []

        if hand is None:
            return points

        # Expected format:
        # {
        #     "landmarks": [(x, y), ...]
        # }
        if isinstance(hand, dict):

            landmarks = hand.get("landmarks", [])

            for point in landmarks:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    points.append((float(point["x"]), float(point["y"])))
                elif isinstance(point, (tuple, list)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))

        return points

    # ---------------------------------------------------------
    # CHECK HAND NEAR FACE
    # ---------------------------------------------------------

    def _hand_near_face(
        self,
        hand: object,
        face_region: tuple[float, float, float, float],
    ) -> tuple[bool, float]:
        """
        Check whether important hand landmarks are inside or
        very close to the estimated face region.
        """

        points = self._get_hand_points(hand)

        if not points:
            return False, 0.0

        face_x1, face_y1, face_x2, face_y2 = face_region

        face_width = max(
            face_x2 - face_x1,
            1.0,
        )

        face_height = max(
            face_y2 - face_y1,
            1.0,
        )

        # Give the face region some tolerance.
        margin_x = face_width * 0.45
        margin_y = face_height * 0.45

        expanded_x1 = face_x1 - margin_x
        expanded_x2 = face_x2 + margin_x

        expanded_y1 = face_y1 - margin_y
        expanded_y2 = face_y2 + margin_y

        matched_points = 0

        for x, y in points:

            if (
                expanded_x1 <= x <= expanded_x2
                and expanded_y1 <= y <= expanded_y2
            ):
                matched_points += 1

        if matched_points == 0:
            return False, 0.0

        confidence = min(
            1.0,
            0.65 + matched_points * 0.04,
        )

        return True, confidence

    # ---------------------------------------------------------
    # FACE OBSTRUCTION
    # ---------------------------------------------------------

    def analyze_face_obstruction(
        self,
        detections: List[object],
        frame_shape: tuple[int, int, int],
        hands: list[object] | None = None,
        frame=None,
    ) -> dict:
        """
        Detect whether a detected hand is covering/near a person's face.

        Detection flow:

        YOLO
          ↓
        Person bounding box
          ↓
        Estimate face region
          ↓
        MediaPipe hand landmarks
          ↓
        Hand near face?
          ↓
        Consecutive-frame confirmation
        """

        hands_were_supplied = hands is not None
        if hands is None:
            hands = []

        person_boxes = [
            detection
            for detection in detections
            if detection.name.lower() == "person"
        ]

        # No person = no face obstruction.
        if not person_boxes:

            self.face_obstruction_hits = 0
            self.face_obstruction_confirmed = False
            self.face_obstruction_confidence = 0.0

            return {
                "obstructed": False,
                "confidence": 0.0,
                "reason": "",
            }

        highest_confidence = 0.0
        hand_on_face = False

        # -----------------------------------------------------
        # Compare every detected hand with every person's face.
        # -----------------------------------------------------

        for person in person_boxes:

            face_region = self._person_face_region(person)

            for hand in hands:

                detected, confidence = self._hand_near_face(
                    hand,
                    face_region,
                )

                if detected:

                    hand_on_face = True

                    highest_confidence = max(
                        highest_confidence,
                        confidence,
                    )

        # Preserve the direct-call fallback used by legacy integrations that
        # do not provide a hand detector result at all.
        if not hand_on_face and not hands_were_supplied:
            frame_height, frame_width = frame_shape[0], frame_shape[1]
            for person in person_boxes:
                x1, y1, x2, y2 = person.box
                person_width = max(x2 - x1, 1.0)
                person_height = max(y2 - y1, 1.0)
                centered = abs(((x1 + x2) / 2.0) - frame_width / 2.0) < frame_width * 0.35
                upper_body = y1 < frame_height * 0.5 and y2 > frame_height * 0.15
                large_enough = person_width > frame_width * 0.35 and person_height > frame_height * 0.45
                if centered and upper_body and large_enough:
                    hand_on_face = True
                    highest_confidence = 0.82
                    break

        # -----------------------------------------------------
        # Consecutive frame confirmation.
        # -----------------------------------------------------

        if hand_on_face:

            self.face_obstruction_hits += 1

            self.face_obstruction_confidence = (
                highest_confidence
            )

        else:

            self.face_obstruction_hits = max(
                0,
                self.face_obstruction_hits - 1,
            )

        confirmation_frames = getattr(
            config,
            "FACE_OCCLUSION_CONFIRMATION_FRAMES",
            3,
        )

        if (
            self.face_obstruction_hits
            >= confirmation_frames
        ):

            self.face_obstruction_confirmed = True

            return {
                "obstructed": True,
                "confidence": self.face_obstruction_confidence,
                "reason": "Face obstruction detected",
            }

        # If hand leaves the face, reset after the counter falls.
        if not hand_on_face:

            if self.face_obstruction_hits == 0:

                self.face_obstruction_confirmed = False
                self.face_obstruction_confidence = 0.0

        return {
            "obstructed": False,
            "confidence": highest_confidence,
            "reason": "",
        }

    # ---------------------------------------------------------
    # GENERAL BEHAVIOR ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        detections: List[object],
        frame_shape: tuple[int, int, int],
        hands: list[object] | None = None,
        frame=None,
    ) -> list[str]:
        """
        Analyze all suspicious behavior.

        Face obstruction is calculated separately in
        analyze_face_obstruction() so that the counter is
        incremented only once per frame.
        """

        reasons: list[str] = []

        height, width = frame_shape[0], frame_shape[1]

        person_boxes = [
            detection
            for detection in detections
            if detection.name.lower() == "person"
        ]

        # -----------------------------------------------------
        # MULTIPLE PEOPLE
        # -----------------------------------------------------

        if len(person_boxes) >= 2:
            reasons.append(
                "Multiple people entering together"
            )

        # -----------------------------------------------------
        # LOITERING
        # -----------------------------------------------------

        if person_boxes:

            if self.loiter_start_time is None:
                self.loiter_start_time = time.monotonic()

            if (
                time.monotonic()
                - self.loiter_start_time
                > 30
                and not self.loitering_detected
            ):

                reasons.append(
                    "Person loitering near ATM"
                )

                self.loitering_detected = True

        else:

            self.loiter_start_time = None
            self.loitering_detected = False

        # -----------------------------------------------------
        # SUDDEN MOVEMENT
        # -----------------------------------------------------

        if self.last_centers and person_boxes:

            current_centers = [
                (d.box[0] + d.box[2]) / 2.0
                for d in person_boxes
            ]

            comparison_count = min(
                len(current_centers),
                len(self.last_centers),
            )

            if comparison_count > 0:

                max_shift = max(
                    abs(
                        current_centers[index]
                        - self.last_centers[index]
                    )
                    for index in range(comparison_count)
                )

                if max_shift > width * 0.25:

                    reasons.append(
                        "Sudden running after entering"
                    )

        self.last_centers = [
            (d.box[0] + d.box[2]) / 2.0
            for d in person_boxes
        ]

        # -----------------------------------------------------
        # PERSON COVERING CAMERA
        # -----------------------------------------------------

        for detection in person_boxes:

            x1, y1, x2, y2 = detection.box

            if (
                x1 < width * 0.20
                and y1 < height * 0.25
                and x2 > width * 0.35
            ):

                reasons.append(
                    "Person covering camera"
                )

                break

        # -----------------------------------------------------
        # ATM VANDALISM
        # -----------------------------------------------------

        for detection in person_boxes:

            x1, y1, x2, y2 = detection.box

            if (
                x1 < width * 0.25
                and y2 > height * 0.70
            ):

                reasons.append(
                    "ATM vandalism"
                )

                break

        # -----------------------------------------------------
        # AGGRESSIVE BEHAVIOR
        # -----------------------------------------------------

        if len(person_boxes) >= 2:

            aggressive_detected = False

            for index, first in enumerate(
                person_boxes[:-1]
            ):

                for second in person_boxes[
                    index + 1 :
                ]:

                    x1, y1, x2, y2 = first.box
                    x3, y3, x4, y4 = second.box

                    overlap = not (
                        x2 < x3
                        or x4 < x1
                        or y2 < y3
                        or y4 < y1
                    )

                    if overlap:

                        aggressive_detected = True
                        break

                if aggressive_detected:
                    break

            if aggressive_detected:

                reasons.append(
                    "Aggressive behaviour"
                )

        # -----------------------------------------------------
        # UNATTENDED OBJECT
        # -----------------------------------------------------

        object_boxes = [
            detection
            for detection in detections
            if detection.name.lower()
            in {
                "backpack",
                "mobile phone",
                "cell phone",
                "phone",
            }
        ]

        if object_boxes and not person_boxes:

            reasons.append(
                "Object left unattended"
            )

        return reasons