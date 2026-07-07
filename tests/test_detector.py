"""
tests/test_detector.py — Unit Tests for detector.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from detector import VehicleDetector, REAL_HEIGHTS, FOCAL_LENGTH


class TestDistanceEstimation:

    def setup_method(self):
        self.det = VehicleDetector()

    def test_known_car_height_at_10m(self):
        """A car at 10m with focal=615 should produce pixel_height ≈ 92."""
        # pixel_h = (real_h * focal) / distance = (1.5 * 615) / 10 = 92.25
        real_h    = REAL_HEIGHTS["car"]
        focal     = FOCAL_LENGTH
        distance  = 10.0
        pixel_h   = int((real_h * focal) / distance)

        estimated = self.det._estimate_distance("car", pixel_h)
        assert estimated is not None
        assert abs(estimated - distance) / distance < 0.05   # within 5%

    def test_zero_pixel_height_returns_none(self):
        assert self.det._estimate_distance("car", 0) is None

    def test_unknown_label_returns_none(self):
        assert self.det._estimate_distance("elephant", 100) is None

    def test_detect_forces_night_model_even_on_bright_frame(self):
        det = VehicleDetector.__new__(VehicleDetector)
        det.night_model = object()
        det.model = object()
        det._detect_night = lambda frame: [{"label": "car"}]
        det._detect_day = lambda frame: [{"label": "day"}]
        det._is_night = lambda frame: False

        detections = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert detections == [{"label": "car"}]

    def test_stub_returns_two_detections(self):
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dets = self.det._stub_detections(dummy)
        assert len(dets) == 2

    def test_stub_detection_has_required_keys(self):
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for det in self.det._stub_detections(dummy):
            assert "label"      in det
            assert "confidence" in det
            assert "bbox"       in det
            assert "distance"   in det
            assert len(det["bbox"]) == 4
