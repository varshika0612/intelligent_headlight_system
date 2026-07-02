"""
focal_calibration.py — One-Time Focal Length Calibration
==========================================================
Run this ONCE to calibrate the focal length for your specific webcam.

Steps:
    1. Stand exactly 2 metres from the webcam.
    2. Run: python calibration/focal_calibration.py
    3. Press SPACE when your bounding box is visible on screen.
    4. Copy the printed FOCAL_LENGTH value into src/detector.py.

Physics:
    focal_length = (pixel_height × known_distance) / real_height
"""

import json
import os
import sys

try:
    import cv2
except ImportError:
    print("OpenCV is required: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics is required: pip install ultralytics")
    sys.exit(1)

# ── Calibration parameters ────────────────────────────────────────────────────
KNOWN_DISTANCE_M: float = 2.0    # Stand exactly this far from the camera (metres)
PERSON_HEIGHT_M:  float = 1.75   # Assumed standing person height (metres)
OUTPUT_FILE:      str   = os.path.join(os.path.dirname(__file__), "calibration.json")


def main() -> None:
    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Calibration] Cannot open webcam.")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════╗
║         FOCAL LENGTH CALIBRATION                     ║
║  Stand EXACTLY {KNOWN_DISTANCE_M:.1f} metres from the webcam.       ║
║  Press SPACE when your full body is detected.        ║
║  Press Q to quit without saving.                     ║
╚══════════════════════════════════════════════════════╝
    """)

    captured_focal: list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        display = frame.copy()

        best_box = None
        best_area = 0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label  = result.names.get(cls_id, "").lower()
                if label != "person":
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box  = (x1, y1, x2, y2)

        if best_box:
            x1, y1, x2, y2 = best_box
            pixel_h = y2 - y1
            focal   = (pixel_h * KNOWN_DISTANCE_M) / PERSON_HEIGHT_M
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Focal est: {focal:.1f}px | Press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Focal Calibration — Stand 2m away", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and best_box:
            x1, y1, x2, y2 = best_box
            pixel_h = y2 - y1
            focal   = (pixel_h * KNOWN_DISTANCE_M) / PERSON_HEIGHT_M
            captured_focal.append(focal)
            print(f"  ✓ Captured  pixel_height={pixel_h}px  →  focal={focal:.2f}px")

            if len(captured_focal) >= 3:
                avg_focal = sum(captured_focal) / len(captured_focal)
                _save_and_report(avg_focal, captured_focal)
                break

        elif key == ord("q"):
            print("[Calibration] Aborted — no calibration saved.")
            break

    cap.release()
    cv2.destroyAllWindows()


def _save_and_report(focal: float, samples: list) -> None:
    data = {
        "focal_length_px": round(focal, 2),
        "samples":         [round(s, 2) for s in samples],
        "known_distance_m": KNOWN_DISTANCE_M,
        "person_height_m":  PERSON_HEIGHT_M,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"""
╔══════════════════════════════════════════════════════╗
║  CALIBRATION COMPLETE                                ║
║                                                      ║
║  Samples:       {samples}
║  Average focal: {focal:.2f} px                            
║                                                      ║
║  → Copy this into src/detector.py:                   ║
║    FOCAL_LENGTH = {focal:.1f}                           
║                                                      ║
║  Saved to: {OUTPUT_FILE}
╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
