"""
detector.py — YOLOv8 Two-Model Pipeline
========================================
Model 1 (BDD100K): Detects vehicles, people, riders in all lighting conditions.
Model 2 (front_rear_v1): Classifies each detected vehicle crop as front or rear.

Pipeline per frame:
    1. Model 1 finds all vehicles/people with bounding boxes
    2. Each vehicle bbox is cropped from the frame
    3. Model 2 classifies the crop as 'front' or 'rear'
    4. direction field is set on each detection
    5. main.py dims LEDs only for 'front' detections and people
"""

import cv2
import numpy as np
from typing import List, Dict, Optional

# ── Calibration ───────────────────────────────────────────────────────────────
FOCAL_LENGTH: float = 615.0   # update after running focal_calibration.py

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATH:      str = "best.pt"           # BDD100K — all lighting conditions
CLASSIFIER_PATH: str = "front_rear_v1.pt"  # front/rear classifier

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
LABEL_MAP: Dict[str, str] = {
    "person":     "person",
    "rider":      "person",
    "car":        "car",
    "truck":      "truck",
    "bus":        "bus",
    "motorcycle": "motorcycle",
    "bicycle":    "bicycle",
}

CONFIDENCE_THRESHOLD: float = 0.35
HORIZON_RATIO:        float = 0.0
MIN_CROP_SIZE:        int   = 20   # pixels — crops smaller than this are skipped


class VehicleDetector:
    """Two-model pipeline: BDD100K detector + front/rear classifier."""

    def __init__(
        self,
        model_path:      str   = MODEL_PATH,
        classifier_path: str   = CLASSIFIER_PATH,
        confidence:      float = CONFIDENCE_THRESHOLD,
        focal_length:    float = FOCAL_LENGTH,
        device:          str   = "cpu",
    ) -> None:
        self.confidence   = confidence
        self.focal_length = focal_length
        self.device       = device

        print("[Detector] Loading BDD100K model (Model 1)...")
        self.model = self._load_model(model_path)

        print("[Detector] Loading front/rear classifier (Model 2)...")
        self.classifier = self._load_model(classifier_path)

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
        """Run two-model pipeline on a single BGR frame.

        Returns list of dicts:
            {
                label:      str,
                confidence: float,
                bbox:       (x1, y1, x2, y2),
                distance:   float | None,
                direction:  'front' | 'rear' | 'unknown',
                mode:       'auto'
            }
        """
        if self.model:
            return self._run_inference(frame)
        return self._stub_detections(frame)

    def current_mode(self) -> str:
        return "auto"

    # ── Model 1: detection ────────────────────────────────────────────────────

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
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                if y2 < horizon_y:
                    continue

                # ── Model 2: classify front vs rear ──────────────────────────
                if label == "person":
                    # Always dim for people regardless of direction
                    direction = "front"
                else:
                    direction = self._classify_direction(frame, x1, y1, x2, y2, w, h)

                conf     = float(box.conf[0].item())
                distance = self._estimate_distance(label, max(1, y2 - y1))

                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "distance":   distance,
                    "direction":  direction,
                    "mode":       "auto",
                })

        return detections

    # ── Model 2: front/rear classifier ───────────────────────────────────────

    def _classify_direction(
        self, frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        fw: int, fh: int,
    ) -> str:
        """Crop the vehicle from the frame and classify as front or rear."""
        if self.classifier is None:
            return "unknown"

        # Clamp crop to frame bounds
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(fw, x2)
        cy2 = min(fh, y2)

        crop = frame[cy1:cy2, cx1:cx2]

        # Reject crops too small to classify
        if crop is None or crop.size == 0:
            return "unknown"
        ch, cw = crop.shape[:2]
        if ch < MIN_CROP_SIZE or cw < MIN_CROP_SIZE:
            return "unknown"

        results = self.classifier.predict(
            source  = crop,
            conf    = 0.45,
            verbose = False,
            device  = self.device,
        )

        # Take highest confidence detection
        best_label = "unknown"
        best_conf  = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf      = float(box.conf[0].item())
                cls_id    = int(box.cls[0].item())
                det_label = result.names.get(cls_id, "unknown").lower()

                if conf > best_conf and det_label in ("front", "rear"):
                    best_conf  = conf
                    best_label = det_label

        return best_label

    # ── Distance estimation ───────────────────────────────────────────────────

    def _estimate_distance(self, label: str, pixel_height: int) -> Optional[float]:
        real_h = REAL_HEIGHTS.get(label)
        if real_h is None or pixel_height <= 0:
            return None
        return round((real_h * self.focal_length) / pixel_height, 2)

    # ── Stub ──────────────────────────────────────────────────────────────────

    def _stub_detections(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        return [
            {
                "label": "car", "confidence": 0.91,
                "bbox":  (int(w*0.1), int(h*0.3), int(w*0.4), int(h*0.7)),
                "distance": 9.5, "direction": "front", "mode": "stub",
            },
            {
                "label": "person", "confidence": 0.84,
                "bbox":  (int(w*0.7), int(h*0.2), int(w*0.8), int(h*0.8)),
                "distance": 4.2, "direction": "front", "mode": "stub",
            },
        ]
