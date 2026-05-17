import numpy as np
from typing import List
from tracker import TrackedObject

GRID_ROWS = 8
GRID_COLS = 8
MARGIN = 1   # extra LEDs to pad around each bounding box

class MaskGenerator:
    def __init__(self, frame_width: int, frame_height: int,
                 calibration_path: str = None):
        self.fw = frame_width
        self.fh = frame_height
        self.calibration = None

        if calibration_path:
            import json
            with open(calibration_path) as f:
                self.calibration = json.load(f)
            print(f"Calibration loaded from {calibration_path}")
        else:
            print("Using linear mapping (no calibration file)")

    def _bbox_to_led_indices(self, bbox):
        x1, y1, x2, y2 = bbox

        if self.calibration:
            # use lookup table (Week 3 upgrade)
            return self._calibrated_mapping(x1, y1, x2, y2)
        else:
            # linear mapping: pixel coords → LED grid indices
            col1 = int(x1 / self.fw * GRID_COLS) - MARGIN
            col2 = int(x2 / self.fw * GRID_COLS) + MARGIN
            row1 = int(y1 / self.fh * GRID_ROWS) - MARGIN
            row2 = int(y2 / self.fh * GRID_ROWS) + MARGIN
            # clamp to grid
            col1, col2 = max(0, col1), min(GRID_COLS-1, col2)
            row1, row2 = max(0, row1), min(GRID_ROWS-1, row2)
            return row1, row2, col1, col2

    def _calibrated_mapping(self, x1, y1, x2, y2):
        # placeholder — filled in Week 3
        return self._bbox_to_led_indices((x1, y1, x2, y2))

    def generate(self, tracked_objects: List[TrackedObject]) -> np.ndarray:
        mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        for obj in tracked_objects:
            r1, r2, c1, c2 = self._bbox_to_led_indices(obj.bbox)
            mask[r1:r2+1, c1:c2+1] = True
        return mask