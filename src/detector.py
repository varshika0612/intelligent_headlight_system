"""
detector.py — YOLOv8 Inference + Distance Estimation
"""

from pathlib import Path
import numpy as np
from typing import List, Dict, Optional

# ── Calibration ───────────────────────────────────────────────────────────────
FOCAL_LENGTH: float     = 615.0
PROJECT_ROOT: Path      = Path(__file__).resolve().parent.parent
NIGHT_MODEL_PATH: str   = str(PROJECT_ROOT / "best.pt")
NIGHT_THRESHOLD: float  = 80.0

# ── Real-world heights (metres) ───────────────────────────────────────────────
REAL_HEIGHTS: Dict[str, float] = {
    # Day model (COCO)
    "person":     1.75,
    "bicycle":    1.10,
    "motorcycle": 1.20,
    "car":        1.50,
    "truck":      2.80,
    "bus":        3.20,
    # Night model — map to same heights
    "car by night":   1.50,
    "headlight":      1.20,   # treated as car proxy
    "high beam":      1.20,   # treated as car proxy
    "low beam":       1.20,   # treated as car proxy
    "truck by night": 2.80,
}

# ── Night model label → pipeline label mapping ────────────────────────────────
# Maps your fine-tuned class names → standard pipeline names
# so the rest of your system (LED grid, distance) stays unchanged
NIGHT_LABEL_MAP: Dict[str, str] = {
    "person":         "person",
    "car by night":   "car",
    "headlight":      "car",        # headlight = car is present
    "high beam":      "car",        # high beam = car is present
    "low beam":       "car",        # low beam  = car is present
    "truck by night": "truck",
}

# Day model COCO classes
ALLOWED_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIDENCE_THRESHOLD: float = 0.45
HORIZON_RATIO: float        = 0.55


class VehicleDetector:

    def __init__(
        self,
        model_path: str       = NIGHT_MODEL_PATH,
        night_model_path: str = NIGHT_MODEL_PATH,
        confidence: float     = CONFIDENCE_THRESHOLD,
        focal_length: float   = FOCAL_LENGTH,
        device: str           = "cpu",
        force_night: bool     = True,
    ) -> None:
        self.confidence   = confidence
        self.focal_length = focal_length
        self.device       = device
        self.force_night  = force_night
        self.model        = None if force_night else self._load_model(model_path)
        self.night_model  = self._load_model(night_model_path)

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self, path: str):
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            model.to(self.device)
            print(f"[Detector] Loaded '{path}' on {self.device}")
            return model
        except ImportError:
            print("[Detector] ultralytics not installed — stub mode.")
            return None
        except Exception as e:
            print(f"[Detector] Could not load '{path}': {e} — stub mode.")
            return None

    # ── Night check ───────────────────────────────────────────────────────────
    def _is_night(self, frame: np.ndarray) -> bool:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < NIGHT_THRESHOLD

    # ── Main detect ───────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Dict]:
        force_night = getattr(self, "force_night", True)
        if force_night and self.night_model:
            return self._detect_night(frame)
        if force_night:
            return self._stub_detections(frame)

        is_night = self._is_night(frame)
        if is_night and self.night_model:
            return self._detect_night(frame)
        if self.model:
            return self._detect_day(frame)
        return self._stub_detections(frame)

    # ── Day detection (COCO YOLOv8) ───────────────────────────────────────────
    def _detect_day(self, frame: np.ndarray) -> List[Dict]:
        h, w      = frame.shape[:2]
        horizon_y = int(h * HORIZON_RATIO)

        results    = self.model.predict(
            source=frame, conf=self.confidence,
            verbose=False, device=self.device
        )
        detections: List[Dict] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
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

    # ── Night detection (your fine-tuned model) ───────────────────────────────
    def _detect_night(self, frame: np.ndarray) -> List[Dict]:
        h, w      = frame.shape[:2]
        horizon_y = int(h * HORIZON_RATIO)

        results = self.night_model.predict(
            source=frame, conf=self.confidence,
            verbose=False, device=self.device
        )
        detections: List[Dict] = []

        # Track seen bboxes to avoid duplicate car detections
        # (headlight + car by night on same vehicle)
        seen_boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id    = int(box.cls[0].item())
                raw_label = result.names.get(cls_id, "unknown").lower()

                # Map to pipeline label
                label = NIGHT_LABEL_MAP.get(raw_label)
                if label is None:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if y2 < horizon_y:
                    continue

                # Deduplicate — skip if a very similar box already added
                # (handles headlight + car by night on same vehicle)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                duplicate = False
                for (sx, sy, slabel) in seen_boxes:
                    if slabel == label and abs(cx - sx) < 50 and abs(cy - sy) < 50:
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen_boxes.append((cx, cy, label))

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
        real_h = REAL_HEIGHTS.get(label)
        if real_h is None or pixel_height <= 0:
            return None
        return round((real_h * self.focal_length) / pixel_height, 2)

    # ── Stub ──────────────────────────────────────────────────────────────────
    def _stub_detections(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        return [
            {
                "label":      "car",
                "confidence": 0.91,
                "bbox":       (int(w*0.1), int(h*0.3), int(w*0.4), int(h*0.7)),
                "distance":   9.5,
            },
            {
                "label":      "person",
                "confidence": 0.84,
                "bbox":       (int(w*0.7), int(h*0.2), int(w*0.8), int(h*0.8)),
                "distance":   4.2,
            },
        ]