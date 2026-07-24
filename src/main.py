"""
main.py — Intelligent Headlight System Entry Point
====================================================
Pipeline:
    webcam → detector (Model 1 + Model 2) → tracker → mask_gen → led_display

Model 1: BDD100K YOLOv8n — detects vehicles and people
Model 2: front/rear classifier — determines if vehicle is oncoming or away

Colour coding:
    Each class has its own colour (green=car, blue=person, etc.)
    Rear vehicles (driving away) → WHITE box
    Coasting (tracker holding)   → GREY box

Only FRONT vehicles and people dim the LEDs.
REAR vehicles keep LEDs on.

Usage:
    python src/main.py                           # live webcam
    python src/main.py --source assets/video.mp4 # video file
    python src/main.py --stub                    # synthetic data
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

from detector    import VehicleDetector, MODEL_PATH
from tracker     import IOUTracker
from mask_gen    import generate_mask, mask_summary, FRAME_WIDTH, FRAME_HEIGHT
from led_display import LEDDisplay

WINDOW_NAME = "Intelligent Headlight System  |  Camera          LED Matrix"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intelligent Headlight System")
    p.add_argument("--source",     default=0,    help="0 = webcam, or path to video file")
    p.add_argument("--stub",       action="store_true", help="Synthetic detections, no webcam")
    p.add_argument("--no-display", action="store_true", help="Headless / CI mode")
    p.add_argument("--model",      default=None, help="Override BDD100K weights path")
    p.add_argument("--device",     default="cuda", help="cpu | cuda | mps")
    return p.parse_args()


# ── Per-label colours (BGR) ───────────────────────────────────────────────────

LABEL_COLOURS = {
    "person":     (255, 100,  50),   # blue
    "car":        (0,   220,   0),   # green
    "truck":      (0,   140, 255),   # orange
    "bus":        (180,   0, 255),   # purple
    "motorcycle": (0,   255, 220),   # yellow-green
    "bicycle":    (255,   0, 150),   # pink
}
REAR_COLOUR     = (255, 255, 255)   # white  — moving away, LEDs stay on
COASTING_COLOUR = (130, 130, 130)   # grey   — tracker holding, YOLO missed
DEFAULT_COLOUR  = (200, 200,   0)   # cyan   — unknown label fallback


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Draw bounding boxes with direction arrows and colour coding.

    Colour logic (priority order):
        1. Coasting (lost_frames > 0)  → GREY
        2. Rear vehicles               → WHITE
        3. Active, class colour        → per-label colour
    """
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        dist      = t.get("distance")
        label     = t["label"]
        track_id  = t["id"]
        lost      = t.get("lost_frames", 0)
        direction = t.get("direction", "unknown")

        # ── Colour selection ──────────────────────────────────────────────────
        if lost > 0:
            color = COASTING_COLOUR          # grey — coasting
        elif direction == "rear":
            color = REAR_COLOUR              # white — moving away
        else:
            color = LABEL_COLOURS.get(label, DEFAULT_COLOUR)  # class colour

        # Thicker box for close objects
        thickness = 3 if (dist and dist < 5.0) else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # ── Label text ────────────────────────────────────────────────────────
        dist_str = f"{dist:.1f}m" if dist else "?m"
        arrow    = "▶" if direction == "front" else "◀" if direction == "rear" else "?"
        text     = f"ID{track_id} {label} {dist_str} {arrow}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Label background — black for white boxes, colour for others
        bg_color = (40, 40, 40) if direction == "rear" else color
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), bg_color, -1)

        # Text — white for coloured backgrounds, black for white background
        txt_color = (255, 255, 255) if direction == "rear" else (0, 0, 0)
        cv2.putText(out, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)

    return out


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Colour legend in top-right corner including rear/coasting."""
    out = frame.copy()
    x   = out.shape[1] - 165
    y   = 18
    gap = 22

    all_entries = list(LABEL_COLOURS.items()) + [
        ("rear (away)", REAR_COLOUR),
        ("coasting",    COASTING_COLOUR),
    ]

    cv2.rectangle(out, (x - 8, 4),
                  (out.shape[1] - 4, y + gap * len(all_entries) - 4),
                  (30, 30, 30), -1)

    for lbl, colour in all_entries:
        cv2.rectangle(out, (x, y - 10), (x + 14, y + 2), colour, -1)
        cv2.putText(out, lbl, (x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)
        y += gap

    return out


def make_combined(camera: np.ndarray, led: np.ndarray) -> np.ndarray:
    target_h = 480
    ch, cw   = camera.shape[:2]
    cam      = cv2.resize(camera, (int(cw * target_h / ch), target_h))
    lh, lw   = led.shape[:2]
    led_r    = cv2.resize(led, (int(lw * target_h / lh), target_h))
    return np.hstack([cam, led_r])


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    detector = VehicleDetector(
        model_path = args.model if args.model else MODEL_PATH,
        device     = args.device,
    )
    tracker = IOUTracker()
    display = LEDDisplay()

    if args.stub:
        _run_stub(detector, tracker, display, args.no_display)
        return

    if not _CV2:
        print("[main] OpenCV required. Exiting.")
        sys.exit(1)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[main] Cannot open source: {source}")
        sys.exit(1)

    ret, first = cap.read()
    if not ret:
        print("[main] Camera returned no frame.")
        sys.exit(1)

    fh, fw = first.shape[:2]
    print(f"[main] Frame size: {fw}×{fh}")
    print("[main] Pipeline running — press Q to quit.")
    print("[main] ▶ coloured = oncoming (dim)   ◀ white = moving away (keep on)")

    fps_start   = time.time()
    frame_count = 0
    frame       = first

    try:
        while True:
            # ── Core pipeline ──────────────────────────────────────────────
            detections = detector.detect(frame)
            tracks     = tracker.update(detections)

            # ── Active tracks only (lost_frames == 0) ──────────────────────
            active_tracks = [t for t in tracks if t["lost_frames"] == 0]

            # ── Direction filter on ACTIVE tracks only ─────────────────────
            # This fixes the "dimmed > total" bug —
            # coasting tracks are excluded from both counts
            bboxes = [
                t["bbox"] for t in active_tracks
                if t["label"] == "person"
                or t.get("direction") == "front"
                or t.get("direction") == "unknown"
            ]

            mask      = generate_mask(bboxes, frame_width=fw, frame_height=fh)
            led_panel = display.render(mask)

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

                cv2.putText(combined, "BDD100K + Front/Rear",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 180), 2, cv2.LINE_AA)

                # Objects = active only, Dimmed = from active only
                # These two numbers are now always consistent
                cv2.putText(
                    combined,
                    f"Objects: {len(active_tracks)}  Dimmed: {len(bboxes)}",
                    (10, combined.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                )

                cv2.imshow(WINDOW_NAME, combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── serial_writer.send(mask)  ← Week 2 hook ───────────────────

            ret, frame = cap.read()
            if not ret:
                print("[main] End of stream.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[main] Stopped. Processed {frame_count} frames.")


def _run_stub(detector, tracker, display, no_display: bool) -> None:
    print("[stub] Running 30 frames with synthetic detections...\n")
    dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    for i in range(30):
        dets          = detector.detect(dummy)
        tracks        = tracker.update(dets)
        active_tracks = [t for t in tracks if t["lost_frames"] == 0]
        bboxes        = [
            t["bbox"] for t in active_tracks
            if t["label"] == "person"
            or t.get("direction") == "front"
            or t.get("direction") == "unknown"
        ]
        mask = generate_mask(bboxes)

        if i % 10 == 0:
            print(f"── Frame {i:02d} " + "─" * 36)
            for t in active_tracks:
                print(f"  ID={t['id']}  {t['label']:<10}  "
                      f"dir={t.get('direction','?'):<10}  "
                      f"dist={t.get('distance')}m")
            print(mask_summary(mask))

    print("\n[stub] Done.")


if __name__ == "__main__":
    run(parse_args())