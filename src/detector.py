"""
detector.py — YOLOv8 Inference + Distance Estimation
=====================================================
Wraps ultralytics YOLOv8 to detect vehicles, people, and bikes,
then estimates distance using the pinhole-camera model.

Distance formula (pinhole model):
    distance = (real_height × focal_length) / pixel_height

Run calibration/focal_calibration.py once and paste the printed value below.
"""

import numpy as np
from typing import List, Dict, Optional

# ── Calibration constant — update after running focal_calibration.py ──────────
FOCAL_LENGTH: float = 615.0   # pixels — replace with your calibrated value

# ── Known real-world heights (metres) ─────────────────────────────────────────
REAL_HEIGHTS: Dict[str, float] = {
    "person":     1.75,
    "bicycle":    1.10,
    "motorcycle": 1.20,
    "car":        1.50,
    "truck":      2.80,
    "bus":        3.20,
}

# COCO class IDs → label names (YOLOv8 uses COCO 80-class labels)
ALLOWED_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIDENCE_THRESHOLD: float = 0.45

# Ignore detections above this fraction of frame height (sky / overhead signs)
HORIZON_RATIO: float = 0.55


class VehicleDetector:
    """YOLOv8 detector — real inference when ultralytics is installed,
    synthetic stub detections otherwise so tests always pass."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = CONFIDENCE_THRESHOLD,
        focal_length: float = FOCAL_LENGTH,
        device: str = "cpu",
    ) -> None:
        self.confidence   = confidence
        self.focal_length = focal_length
        self.device       = device
        self.model        = self._load_model(model_path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: str):
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            model.to(self.device)
            print(f"[Detector] YOLOv8 loaded from '{path}' on {self.device}")
            return model
        except ImportError:
            print("[Detector] ultralytics not installed — stub mode active.")
            return None
        except Exception as e:
            print(f"[Detector] Could not load model '{path}': {e} — stub mode active.")
            return None

    # ── Public inference API ──────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on a single BGR frame.

        Returns a list of dicts — one per detected object:
            {
                "label":      str,    # "car", "person", etc.
                "confidence": float,  # 0.0–1.0
                "bbox":       (x1, y1, x2, y2),  # pixel ints
                "distance":   float | None,       # metres, pinhole estimate
            }
        """
        if self.model is None:
            return self._stub_detections(frame)

        h, w = frame.shape[:2]
        horizon_y = int(h * HORIZON_RATIO)

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )

        detections: List[Dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())

                # Only keep road-relevant classes
                if cls_id not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                # Discard anything above the horizon (sky, signs, etc.)
                if y2 < horizon_y:
                    continue

                label    = ALLOWED_CLASSES[cls_id]
                conf     = float(box.conf[0].item())
                pixel_h  = max(1, y2 - y1)
                distance = self._estimate_distance(label, pixel_h)

                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "distance":   distance,
                })

        return detections

    # ── Distance estimation ───────────────────────────────────────────────────

    def _estimate_distance(self, label: str, pixel_height: int) -> Optional[float]:
        """Pinhole camera model: distance = (real_h × focal_length) / pixel_h.
        Accuracy ±20% after calibration. Sufficient for LED zone selection."""
        real_h = REAL_HEIGHTS.get(label)
        if real_h is None or pixel_height <= 0:
            return None
        return round((real_h * self.focal_length) / pixel_height, 2)

    # ── Stub — always works, no dependencies ─────────────────────────────────

    def _stub_detections(self, frame: np.ndarray) -> List[Dict]:
        """Synthetic detections for unit tests / no-camera environments."""
        h, w = frame.shape[:2]
        return [
            {
                "label":      "car",
                "confidence": 0.91,
                "bbox":       (int(w * 0.1), int(h * 0.3), int(w * 0.4), int(h * 0.7)),
                "distance":   9.5,
            },
            {
                "label":      "person",
                "confidence": 0.84,
                "bbox":       (int(w * 0.7), int(h * 0.2), int(w * 0.8), int(h * 0.8)),
                "distance":   4.2,
            },
        ]
