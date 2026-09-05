import cv2
import mediapipe as mp

import config


class HandDetector:
    def __init__(self):
        model_path = str(config.HAND_MODEL_PATH)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )

    def detect(self, frame):
        """
        Detect hands in an OpenCV BGR frame.

        Returns:
            List of hands.
            Each hand contains 21 landmarks.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.detector.detect(mp_image)

        detected_hands = []

        if result.hand_landmarks:

            height, width, _ = frame.shape

            for hand in result.hand_landmarks:

                landmarks = []

                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    landmarks.append({
                        "x": x,
                        "y": y,
                        "z": landmark.z,
                    })

                detected_hands.append(landmarks)

        return detected_hands

    def draw(self, frame, hands):
        """
        Draw hand landmarks and connections.
        """

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        for landmarks in hands:

            for start_idx, end_idx in connections:

                start = landmarks[start_idx]
                end = landmarks[end_idx]

                cv2.line(
                    frame,
                    (start["x"], start["y"]),
                    (end["x"], end["y"]),
                    (0, 255, 0),
                    2,
                )

            for point in landmarks:

                cv2.circle(
                    frame,
                    (point["x"], point["y"]),
                    4,
                    (0, 0, 255),
                    -1,
                )

        return frame