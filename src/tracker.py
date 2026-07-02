"""
tracker.py — IOU Object Tracker
================================
Assigns stable integer IDs to detections across frames using
Intersection-over-Union (IOU) matching.  No external library required.

Design goals
------------
• Prevent LED flicker: a detection missing for 1–2 frames keeps its ID.
• Tracks disappear after `max_lost_frames` consecutive misses.
• Pure numpy — no scipy, no deep-sort.

Data structures
---------------
    Detection : dict with keys  label, confidence, bbox, distance
    Track     : dict with keys  id, label, bbox, distance, lost_frames, age
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

# Minimum IOU to consider two boxes the same object.
IOU_THRESHOLD: float = 0.3

# How many consecutive frames a track can go unmatched before being dropped.
MAX_LOST_FRAMES: int = 5

Detection = Dict   # {label, confidence, bbox: (x1,y1,x2,y2), distance}
Track     = Dict   # {id, label, bbox, distance, lost_frames, age}


# ── IOU helper ────────────────────────────────────────────────────────────────

def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute Intersection-over-Union between two axis-aligned bounding boxes.

    Args:
        box_a, box_b: (x_min, y_min, x_max, y_max)

    Returns:
        IOU in [0.0, 1.0].  Returns 0.0 for degenerate (zero-area) boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle.
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
    union   = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ── Tracker class ─────────────────────────────────────────────────────────────

class IOUTracker:
    """Greedy IOU tracker — O(D×T) per frame, sufficient for ≤20 objects."""

    def __init__(
        self,
        iou_threshold: float = IOU_THRESHOLD,
        max_lost_frames: int  = MAX_LOST_FRAMES,
    ) -> None:
        self.iou_threshold   = iou_threshold
        self.max_lost_frames = max_lost_frames
        self._tracks: List[Track] = []
        self._next_id: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Track]:
        """Match new detections to existing tracks and return active tracks.

        Algorithm (greedy):
        1. Build IOU matrix  (tracks × detections).
        2. Greedily assign each detection to the track with the highest IOU
           that exceeds `iou_threshold`.
        3. Unmatched detections → new tracks.
        4. Unmatched tracks    → increment lost_frames; drop if too high.

        Args:
            detections: List of Detection dicts from detector.py for this frame.

        Returns:
            All currently *active* tracks (including coasting tracks that were
            unmatched this frame but haven't timed out yet).
        """
        if not self._tracks and not detections:
            return []

        matched_track_ids:     set = set()
        matched_detection_ids: set = set()

        # ── Step 1: build IOU matrix ─────────────────────────────────────────
        if self._tracks and detections:
            iou_matrix = np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self._tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = compute_iou(track["bbox"], det["bbox"])

            # ── Step 2: greedy matching ───────────────────────────────────────
            # Sort candidate (track, detection) pairs by IOU descending.
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

            for iou_score, t_idx, d_idx in candidates:
                if t_idx in matched_track_ids or d_idx in matched_detection_ids:
                    continue
                # Update the matched track with fresh detection data.
                self._tracks[t_idx]["bbox"]        = detections[d_idx]["bbox"]
                self._tracks[t_idx]["label"]       = detections[d_idx]["label"]
                self._tracks[t_idx]["distance"]    = detections[d_idx].get("distance")
                self._tracks[t_idx]["confidence"]  = detections[d_idx].get("confidence", 1.0)
                self._tracks[t_idx]["lost_frames"] = 0
                self._tracks[t_idx]["age"]        += 1

                matched_track_ids.add(t_idx)
                matched_detection_ids.add(d_idx)

        # ── Step 3: unmatched detections → new tracks ─────────────────────────
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detection_ids:
                self._tracks.append({
                    "id":          self._next_id,
                    "label":       det["label"],
                    "bbox":        det["bbox"],
                    "distance":    det.get("distance"),
                    "confidence":  det.get("confidence", 1.0),
                    "lost_frames": 0,
                    "age":         1,
                })
                self._next_id += 1

        # ── Step 4: unmatched tracks → age them out ────────────────────────────
        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_track_ids:
                track["lost_frames"] += 1

        # Remove stale tracks.
        self._tracks = [
            t for t in self._tracks
            if t["lost_frames"] <= self.max_lost_frames
        ]

        return list(self._tracks)

    def reset(self) -> None:
        """Clear all tracks (call between video clips or on camera reconnect)."""
        self._tracks   = []
        self._next_id  = 0

    @property
    def active_tracks(self) -> List[Track]:
        """Tracks that were matched in the most recent frame (lost_frames == 0)."""
        return [t for t in self._tracks if t["lost_frames"] == 0]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tracker = IOUTracker()

    frame1 = [
        {"label": "car",    "confidence": 0.92, "bbox": (100, 200, 400, 500), "distance": 8.2},
        {"label": "person", "confidence": 0.87, "bbox": (800, 300, 900, 600), "distance": 5.1},
    ]
    frame2 = [
        {"label": "car",    "confidence": 0.90, "bbox": (105, 205, 405, 505), "distance": 7.9},
        {"label": "person", "confidence": 0.85, "bbox": (805, 305, 905, 605), "distance": 4.9},
    ]
    # Simulate a missed detection for the person in frame 3.
    frame3 = [
        {"label": "car",    "confidence": 0.88, "bbox": (110, 210, 410, 510), "distance": 7.5},
    ]

    for i, frame in enumerate([frame1, frame2, frame3], start=1):
        tracks = tracker.update(frame)
        print(f"Frame {i}: {len(tracks)} active tracks")
        for t in tracks:
            print(f"  ID={t['id']:02d}  label={t['label']:<8}  lost={t['lost_frames']}  bbox={t['bbox']}")
