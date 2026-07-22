"""
main.py — Intelligent Headlight System Entry Point
====================================================
Pipeline:
    webcam → detector → tracker → mask_gen → led_display
                                           → serial_writer (Week 2)

Displays side-by-side: camera feed (left) | LED matrix (right).

Usage:
    python src/main.py                          # live webcam
    python src/main.py --source assets/video.mp4   # sample video file
    python src/main.py --stub                   # no webcam, synthetic data
"""

import argparse
import sys
import time
import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    print("[main] OpenCV not installed — pip install opencv-python")

from detector    import VehicleDetector
from tracker     import IOUTracker
from mask_gen    import generate_mask, mask_summary, FRAME_WIDTH, FRAME_HEIGHT
from led_display import LEDDisplay

WINDOW_NAME = "Intelligent Headlight System  |  Camera          LED Matrix"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intelligent Headlight System")
    p.add_argument("--source",     default=0,     help="0 = webcam, or path to video file")
    p.add_argument("--stub",       action="store_true", help="Synthetic detections, no webcam")
    p.add_argument("--no-display", action="store_true", help="Headless / CI mode")
    p.add_argument("--model", default=None, help="YOLOv8 weights path (default: uses MODEL_PATH from detector.py)")
    p.add_argument("--device",     default="cuda",  help="cpu | cuda | mps")
    return p.parse_args()


# ── Per-label colours (BGR) ───────────────────────────────────────────────────
# Each detected class gets its own distinct colour when actively tracked.
# When coasting (YOLO missed it, tracker keeping alive) the colour fades to grey.

LABEL_COLOURS = {
    "person":     (255, 100,  50),   # blue
    "car":        (0,   220,   0),   # green
    "truck":      (0,   140, 255),   # orange
    "bus":        (180,   0, 255),   # purple
    "motorcycle": (0,   255, 220),   # yellow-green
    "bicycle":    (255,   0, 150),   # pink
}
COASTING_COLOUR = (130, 130, 130)    # grey — tracker alive, YOLO missed this frame
DEFAULT_COLOUR  = (200, 200,   0)    # cyan fallback for any unknown label


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Draw bounding boxes + labels + distance on the camera frame.
    Each object class gets its own colour. Coasting objects turn grey."""
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        dist     = t.get("distance")
        label    = t["label"]
        track_id = t["id"]
        lost     = t.get("lost_frames", 0)

        # Active: use per-label colour. Coasting: grey.
        if lost == 0:
            color = LABEL_COLOURS.get(label, DEFAULT_COLOUR)
        else:
            color = COASTING_COLOUR

        # Bounding box — thicker for close objects (distance < 5m)
        thickness = 3 if (dist and dist < 5.0) else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        dist_str = f"{dist:.1f}m" if dist else "?m"
        text     = f"ID{track_id} {label} {dist_str}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Filled label background so text is always readable
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw a small colour legend in the top-right corner of the camera feed."""
    out    = frame.copy()
    x      = out.shape[1] - 160
    y      = 18
    gap    = 22

    cv2.rectangle(out, (x - 8, 4), (out.shape[1] - 4, y + gap * len(LABEL_COLOURS) - 4),
                  (30, 30, 30), -1)

    for label, colour in LABEL_COLOURS.items():
        cv2.rectangle(out, (x, y - 10), (x + 14, y + 2), colour, -1)
        cv2.putText(out, label, (x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        y += gap

    return out


def make_combined(camera: np.ndarray, led: np.ndarray) -> np.ndarray:
    """Resize both panels to the same height and place side by side."""
    target_h = 480

    # Camera side
    ch, cw = camera.shape[:2]
    cam_w  = int(cw * target_h / ch)
    cam    = cv2.resize(camera, (cam_w, target_h))

    # LED side — keep square aspect ratio
    lh, lw = led.shape[:2]
    led_w  = int(lw * target_h / lh)
    led_r  = cv2.resize(led, (led_w, target_h))

    return np.hstack([cam, led_r])


# ── Main loops ────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    from detector import MODEL_PATH
    detector = VehicleDetector(
    model_path = args.model if args.model else MODEL_PATH,
    device     = args.device,
    )
    tracker  = IOUTracker()
    display  = LEDDisplay()

    if args.stub:
        _run_stub(detector, tracker, display, args.no_display)
        return

    if not _CV2:
        print("[main] OpenCV required for webcam.  Exiting.")
        sys.exit(1)

    # Accept integer (webcam index) or string (file path)
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[main] Cannot open source: {source}")
        sys.exit(1)

    # Read one frame to get real frame dimensions for mask_gen
    ret, first = cap.read()
    if not ret:
        print("[main] Camera returned no frame.")
        sys.exit(1)

    fh, fw = first.shape[:2]
    print(f"[main] Frame size: {fw}×{fh}")
    print("[main] Pipeline running — press Q to quit.")

    fps_start   = time.time()
    frame_count = 0
    frame       = first   # process the first frame too

    try:
        while True:
            # ── Core pipeline ──────────────────────────────────────────────
            detections = detector.detect(frame)
            tracks     = tracker.update(detections)
            bboxes     = [t["bbox"] for t in tracks]
            mask       = generate_mask(bboxes, frame_width=fw, frame_height=fh)
            led_panel  = display.render(mask)

            # ── Display ────────────────────────────────────────────────────
            if not args.no_display:
                annotated = draw_tracks(frame, tracks)
                annotated = draw_legend(annotated)
                combined  = make_combined(annotated, led_panel)

                frame_count += 1
                fps = frame_count / max(0.001, time.time() - fps_start)
                cv2.putText(combined, f"FPS: {fps:.1f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Day / night mode indicator
                mode_text  = "BDD100K MODEL"
                mode_color = (0, 255, 180)   # teal — single model indicator
                cv2.putText(combined, mode_text,
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, mode_color, 2, cv2.LINE_AA)

                # Object count bottom-left
                cv2.putText(combined, f"Objects: {len([t for t in tracks if t['lost_frames']==0])}",
                            (10, combined.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow(WINDOW_NAME, combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── Week 2 hook ────────────────────────────────────────────────
            # serial_writer.send(mask)

            # Read next frame
            ret, frame = cap.read()
            if not ret:
                print("[main] End of stream.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[main] Stopped. Processed {frame_count} frames.")


def _run_stub(detector, tracker, display, no_display: bool) -> None:
    """Headless mode — 30 synthetic frames, prints ASCII masks to terminal."""
    print("[stub] Running 30 frames with synthetic detections...\n")
    dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    for i in range(30):
        dets   = detector.detect(dummy)
        tracks = tracker.update(dets)
        bboxes = [t["bbox"] for t in tracks]
        mask   = generate_mask(bboxes)

        if i % 10 == 0:
            print(f"── Frame {i:02d} " + "─" * 36)
            for t in tracks:
                d = t.get("distance")
                print(f"  ID={t['id']}  {t['label']:<10}  dist={d}m  lost={t['lost_frames']}")
            print(mask_summary(mask))

    print("\n[stub] Done.")


if __name__ == "__main__":
    run(parse_args())
