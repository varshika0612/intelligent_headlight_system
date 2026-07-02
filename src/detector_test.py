"""
detector_test.py — Live Visual Sanity Check
============================================
Quick standalone script to verify YOLO detections on your webcam
BEFORE integrating with the full pipeline.

Run:
    python src/detector_test.py

What you'll see:
    - Live webcam feed with bounding boxes drawn
    - Label + confidence + distance printed in terminal
    - Press Q to quit

This is NOT a pytest test — it's a human eyeball check.
Run this first every time you change detector.py.
"""

import sys
import cv2
from detector import VehicleDetector

def main():
    detector = VehicleDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[detector_test] Cannot open webcam. Check it's connected.")
        sys.exit(1)

    print("[detector_test] Running — press Q to quit.")
    print(f"{'Label':<12} {'Conf':>6}  {'Distance':>10}  {'BBox'}")
    print("─" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            dist = d["distance"]
            dist_str = f"{dist:.1f}m" if dist else "   ?m"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)

            # Label above box
            text = f"{d['label']} {d['confidence']:.2f}  {dist_str}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 80), -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Terminal output
            print(f"{d['label']:<12} {d['confidence']:>6.2f}  {dist_str:>10}  {d['bbox']}")

        cv2.imshow("Detector Test — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[detector_test] Done.")

if __name__ == "__main__":
    main()
