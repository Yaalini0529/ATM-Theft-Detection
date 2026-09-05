import unittest

from behavior import BehaviorAnalyzer


class DummyDetection:
    def __init__(self, name, box, confidence):
        self.name = name
        self.box = box
        self.confidence = confidence


class FaceObstructionTests(unittest.TestCase):
    def test_sustained_hand_cover_without_detected_hand_still_triggers(self):
        analyzer = BehaviorAnalyzer()
        person = DummyDetection("person", (120, 80, 520, 420), 0.96)

        for _ in range(3):
            result = analyzer.analyze_face_obstruction([person], (480, 640, 3))
            if result["obstructed"]:
                break

        self.assertTrue(result["obstructed"])


if __name__ == "__main__":
    unittest.main()
