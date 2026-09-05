import cv2

from hand_detector import HandDetector


def main():

    detector = HandDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    print("Camera started.")
    print("Show your hand to the camera.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        hands = detector.detect(frame)

        detector.draw(frame, hands)

        cv2.putText(
            frame,
            f"Hands detected: {len(hands)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand Detection Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()