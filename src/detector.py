"""
detector.py — YOLOv8 Inference + Distance Estimation
=====================================================
Single BDD100K fine-tuned model handles all lighting conditions.
Day, night, dusk — one model, no switching logic.

Classes: person, rider, car, truck, bus, motorcycle, bicycle
Distance estimated using pinhole camera model.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional

# ── Calibration ───────────────────────────────────────────────────────────────
FOCAL_LENGTH: float = 615.0   # update after running focal_calibration.py

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH: str = "best.pt"   # BDD100K fine-tuned — all conditions

# ── Real-world heights (metres) for distance estimation ───────────────────────
REAL_HEIGHTS: Dict[str, float] = {
    "person":     1.75,
    "rider":      1.75,
    "bicycle":    1.10,
    "motorcycle": 1.20,
    "car":        1.50,
    "truck":      2.80,
    "bus":        3.20,
}

# ── BDD100K label → pipeline label ────────────────────────────────────────────
# Your BDD100K data.yaml classes:
#   0:person  1:rider  2:car  3:truck  4:bus  5:motorcycle  6:bicycle
LABEL_MAP: Dict[str, str] = {
    "person":     "person",
    "rider":      "person",      # rider = person on vehicle
    "car":        "car",
    "truck":      "truck",
    "bus":        "bus",
    "motorcycle": "motorcycle",
    "bicycle":    "bicycle",
}

CONFIDENCE_THRESHOLD: float = 0.35
HORIZON_RATIO:        float = 0.0

class VehicleDetector:
    """Single-model detector using BDD100K fine-tuned YOLOv8n."""

    def __init__(
        self,
        model_path:   str   = MODEL_PATH,
        confidence:   float = CONFIDENCE_THRESHOLD,
        focal_length: float = FOCAL_LENGTH,
        device:       str   = "cpu",
    ) -> None:
        self.confidence   = confidence
        self.focal_length = focal_length
        self.device       = device

        print("[Detector] Loading BDD100K model...")
        self.model = self._load_model(model_path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: str):
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            model.to(self.device)
            print(f"  ✓ Loaded '{path}' on {self.device}")
            return model
        except ImportError:
            print("  ✗ ultralytics not installed — stub mode.")
            return None
        except Exception as e:
            print(f"  ✗ Could not load '{path}': {e}")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on a single BGR frame.

        Works for all lighting conditions — day, night, dusk.
        Does NOT draw anything on the frame.
        All drawing happens in main.py.

        Returns list of dicts:
            {
                label:      str,
                confidence: float,
                bbox:       (x1, y1, x2, y2),
                distance:   float | None,
                mode:       'auto'
            }
        """
        if self.model:
            return self._run_inference(frame)
        return self._stub_detections(frame)

    def current_mode(self) -> str:
        """Single unified model — always returns 'auto'."""
        return "auto"

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> List[Dict]:
        h, w      = frame.shape[:2]
        horizon_y = int(h * HORIZON_RATIO)

        results = self.model.predict(
            source  = frame,
            conf    = self.confidence,
            verbose = False,
            device  = self.device,
        )

        detections: List[Dict] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id    = int(box.cls[0].item())
                raw_label = result.names.get(cls_id, "unknown").lower()

                label = LABEL_MAP.get(raw_label)
                if label is None:
                    continue   # ignore irrelevant classes

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                # Ignore detections above horizon — signs, bridges, sky
                if y2 < horizon_y:
                    continue

                conf     = float(box.conf[0].item())
                distance = self._estimate_distance(label, max(1, y2 - y1))

                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "distance":   distance,
                    "mode":       "auto",
                })

        return detections

    # ── Distance estimation ───────────────────────────────────────────────────

    def _estimate_distance(self, label: str, pixel_height: int) -> Optional[float]:
        """Pinhole model: distance = (real_height × focal_length) / pixel_height.
        Accuracy ±20% after focal calibration."""
        real_h = REAL_HEIGHTS.get(label)
        if real_h is None or pixel_height <= 0:
            return None
        return round((real_h * self.focal_length) / pixel_height, 2)

    # ── Stub ──────────────────────────────────────────────────────────────────

    def _stub_detections(self, frame: np.ndarray) -> List[Dict]:
        """Synthetic detections when no model is loaded — for tests and CI."""
        h, w = frame.shape[:2]
        return [
            {
                "label":      "car",
                "confidence": 0.91,
                "bbox":       (int(w*0.1), int(h*0.3), int(w*0.4), int(h*0.7)),
                "distance":   9.5,
                "mode":       "stub",
            },
            {
                "label":      "person",
                "confidence": 0.84,
                "bbox":       (int(w*0.7), int(h*0.2), int(w*0.8), int(h*0.8)),
                "distance":   4.2,
                "mode":       "stub",
            },
        ]