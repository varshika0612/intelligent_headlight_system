import numpy as np
from dataclasses import dataclass
from typing import List
from detector import Detection

@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple        # (x1, y1, x2, y2)
    class_name: str
    last_seen: float   # timestamp

class SimpleTracker:
    """
    Lightweight IOU-based tracker. No extra dependencies beyond numpy.
    """
    def __init__(self, max_lost_frames=5):
        self.tracks = {}        # id -> TrackedObject
        self.next_id = 0
        self.max_lost = max_lost_frames
        self.lost_count = {}    # id -> frames since last matched

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def update(self, detections: List[Detection], timestamp: float) -> List[TrackedObject]:
        matched_ids = set()
        results = []

        for det in detections:
            best_id, best_iou = None, 0.3   # IOU threshold
            for tid, obj in self.tracks.items():
                iou = self._iou(det.bbox, obj.bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None:
                # update existing track
                self.tracks[best_id].bbox = det.bbox
                self.tracks[best_id].last_seen = timestamp
                self.lost_count[best_id] = 0
                matched_ids.add(best_id)
                results.append(self.tracks[best_id])
            else:
                # new track
                new_obj = TrackedObject(self.next_id, det.bbox, det.class_name, timestamp)
                self.tracks[self.next_id] = new_obj
                self.lost_count[self.next_id] = 0
                matched_ids.add(self.next_id)
                results.append(new_obj)
                self.next_id += 1

        # age unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                self.lost_count[tid] = self.lost_count.get(tid, 0) + 1
                if self.lost_count[tid] > self.max_lost:
                    del self.tracks[tid]
                    del self.lost_count[tid]
                else:
                    results.append(self.tracks[tid])  # keep for persistence

        return results