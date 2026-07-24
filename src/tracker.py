"""
tracker.py — IOU Object Tracker with Direction Detection
=========================================================
Assigns stable integer IDs to detections across frames using
Intersection-over-Union (IOU) matching. No external library required.

Direction detection via bbox area growth (backup to Model 2 classifier).
Only updates history on actual YOLO matches — not during coasting.
"""

import numpy as np
from typing import List, Dict, Tuple

IOU_THRESHOLD:    float = 0.3
MAX_LOST_FRAMES:  int   = 10    # increased — gives more history for direction
HISTORY_SIZE:     int   = 10    # keep last N detected frames
MIN_HISTORY:      int   = 5     # need at least this many to decide direction
GROWTH_THRESHOLD: float = 0.08  # >8% area growth = oncoming

Detection = Dict
Track     = Dict


# ── IOU helper ────────────────────────────────────────────────────────────────

def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ── Direction detection ───────────────────────────────────────────────────────

def _bbox_area(bbox: Tuple) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _get_direction(history: List[Tuple]) -> str:
    """Backup direction detection via bbox area growth.

    Used when Model 2 (front/rear classifier) returns 'unknown'.
    Compares first 2 vs last 2 frames in detection history.

    Returns: 'oncoming', 'away', or 'unknown'
    """
    if len(history) < MIN_HISTORY:
        return "unknown"

    early = sum(_bbox_area(b) for b in history[:2]) / 2
    late  = sum(_bbox_area(b) for b in history[-2:]) / 2

    if early == 0:
        return "unknown"

    growth = (late - early) / early

    if growth > GROWTH_THRESHOLD:
        return "oncoming"
    elif growth < -GROWTH_THRESHOLD:
        return "away"
    else:
        return "unknown"


# ── Tracker class ─────────────────────────────────────────────────────────────

class IOUTracker:
    """Greedy IOU tracker with direction detection."""

    def __init__(
        self,
        iou_threshold:   float = IOU_THRESHOLD,
        max_lost_frames: int   = MAX_LOST_FRAMES,
    ) -> None:
        self.iou_threshold   = iou_threshold
        self.max_lost_frames = max_lost_frames
        self._tracks: List[Track] = []
        self._next_id: int = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        if not self._tracks and not detections:
            return []

        matched_track_ids:     set = set()
        matched_detection_ids: set = set()

        # ── Step 1: IOU matrix ────────────────────────────────────────────────
        if self._tracks and detections:
            iou_matrix = np.zeros(
                (len(self._tracks), len(detections)), dtype=np.float32
            )
            for t_idx, track in enumerate(self._tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = compute_iou(
                        track["bbox"], det["bbox"]
                    )

            # ── Step 2: greedy matching ───────────────────────────────────────
            candidates = sorted(
                [
                    (iou_matrix[t, d], t, d)
                    for t in range(len(self._tracks))
                    for d in range(len(detections))
                    if iou_matrix[t, d] >= self.iou_threshold
                ],
                key=lambda x: x[0],
                reverse=True,
            )

            for _, t_idx, d_idx in candidates:
                if t_idx in matched_track_ids or d_idx in matched_detection_ids:
                    continue

                det = detections[d_idx]

                # Update core fields
                self._tracks[t_idx]["bbox"]        = det["bbox"]
                self._tracks[t_idx]["label"]       = det["label"]
                self._tracks[t_idx]["distance"]    = det.get("distance")
                self._tracks[t_idx]["confidence"]  = det.get("confidence", 1.0)
                self._tracks[t_idx]["lost_frames"] = 0
                self._tracks[t_idx]["age"]        += 1

                # Direction from Model 2 if available, else use bbox growth
                model2_direction = det.get("direction", "unknown")
                if model2_direction in ("front", "rear"):
                    self._tracks[t_idx]["direction"] = model2_direction
                else:
                    # Fall back to bbox growth method
                    history = self._tracks[t_idx]["bbox_history"]
                    history.append(det["bbox"])
                    if len(history) > HISTORY_SIZE:
                        history.pop(0)
                    self._tracks[t_idx]["direction"] = _get_direction(history)

                matched_track_ids.add(t_idx)
                matched_detection_ids.add(d_idx)

        # ── Step 3: new tracks ────────────────────────────────────────────────
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detection_ids:
                self._tracks.append({
                    "id":           self._next_id,
                    "label":        det["label"],
                    "bbox":         det["bbox"],
                    "distance":     det.get("distance"),
                    "confidence":   det.get("confidence", 1.0),
                    "lost_frames":  0,
                    "age":          1,
                    "bbox_history": [det["bbox"]],
                    "direction":    det.get("direction", "unknown"),
                })
                self._next_id += 1

        # ── Step 4: age out unmatched ─────────────────────────────────────────
        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_track_ids:
                track["lost_frames"] += 1

        self._tracks = [
            t for t in self._tracks
            if t["lost_frames"] <= self.max_lost_frames
        ]

        return list(self._tracks)

    def reset(self) -> None:
        self._tracks  = []
        self._next_id = 0

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t["lost_frames"] == 0]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tracker = IOUTracker()

    frames = [
        [{"label": "car", "confidence": 0.92,
          "bbox": (400, 300, 500, 380), "distance": 20.0, "direction": "front"}],
        [{"label": "car", "confidence": 0.91,
          "bbox": (380, 280, 530, 400), "distance": 17.0, "direction": "front"}],
        [{"label": "car", "confidence": 0.90,
          "bbox": (360, 260, 560, 420), "distance": 14.0, "direction": "front"}],
        [],   # missed frame — coasting
        [{"label": "car", "confidence": 0.88,
          "bbox": (300, 200, 640, 500), "distance": 8.0,  "direction": "front"}],
        [{"label": "car", "confidence": 0.87,
          "bbox": (250, 150, 700, 560), "distance": 5.0,  "direction": "unknown"}],
    ]

    print("Two-model pipeline smoke test:\n")
    for i, dets in enumerate(frames, start=1):
        tracks = tracker.update(dets)
        for t in tracks:
            print(f"  Frame {i:02d}  ID={t['id']}  "
                  f"dir={t.get('direction','?'):<10}  "
                  f"lost={t['lost_frames']}  "
                  f"hist={len(t.get('bbox_history',[]))}  "
                  f"dist={t['distance']}m")
        if not tracks:
            print(f"  Frame {i:02d}  no tracks (coasting)")
