import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List

ALLOWED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CONFIDENCE_THRESHOLD = 0.45
HORIZON_RATIO = 0.55

# --- distance estimation constants ---
FOCAL_LENGTH = 182.0   # replace with YOUR calibrated value ##UPDATE

KNOWN_REAL_HEIGHTS = {   ## UPDATE WITH OTHER VALUES
    0: 1.7,   # person
    1: 1.1,   # bicycle
    2: 1.5,   # car
    3: 1.1,   # motorcycle
    5: 3.0,   # bus
    7: 2.5,   # truck
}

@dataclass
class Detection:
    bbox: tuple           # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    distance: float       


class Detector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        print("Detector ready")

    def _estimate_distance(self, class_id: int, bbox: tuple) -> float:
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1   # height of bounding box in pixels

        if bbox_height == 0:
            return 999.0   # avoid division by zero

        real_height = KNOWN_REAL_HEIGHTS.get(class_id, 1.5)
        distance = (real_height * FOCAL_LENGTH) / bbox_height
        return round(distance, 1)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        horizon_y = int(h * HORIZON_RATIO)

        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in ALLOWED_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if y2 < horizon_y:
                continue

            distance = self._estimate_distance(cls_id, (x1, y1, x2, y2))

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=ALLOWED_CLASSES[cls_id],
                confidence=conf,
                distance=distance
            ))
        
        return detections