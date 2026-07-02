"""
tests/test_mask_gen.py — Unit Tests for mask_gen.py
=====================================================
Tests for the geometry / LED mapping module (Sri's primary role).

Run:  pytest tests/test_mask_gen.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mask_gen import (
    pixel_to_grid,
    bbox_to_grid_range,
    generate_mask,
    GRID_ROWS, GRID_COLS,
    FRAME_WIDTH, FRAME_HEIGHT,
)


# ══════════════════════════════════════════════════════════════════════════════
# pixel_to_grid
# ══════════════════════════════════════════════════════════════════════════════

class TestPixelToGrid:

    def test_origin_maps_to_zero_zero(self):
        """Top-left pixel (0,0) → grid cell (0,0)."""
        assert pixel_to_grid(0, 0) == (0, 0)

    def test_centre_pixel(self):
        """Frame centre → grid centre-ish."""
        row, col = pixel_to_grid(FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        assert row == GRID_ROWS // 2
        assert col == GRID_COLS // 2

    def test_bottom_right_pixel(self):
        """Last valid pixel (W-1, H-1) → last grid cell."""
        row, col = pixel_to_grid(FRAME_WIDTH - 1, FRAME_HEIGHT - 1)
        assert row == GRID_ROWS - 1
        assert col == GRID_COLS - 1

    def test_exact_frame_width_clamped(self):
        """x == FRAME_WIDTH (one beyond last) is clamped to last column."""
        row, col = pixel_to_grid(FRAME_WIDTH, 0)
        assert col == GRID_COLS - 1

    def test_exact_frame_height_clamped(self):
        """y == FRAME_HEIGHT (one beyond last) is clamped to last row."""
        row, col = pixel_to_grid(0, FRAME_HEIGHT)
        assert row == GRID_ROWS - 1

    def test_negative_coords_clamped(self):
        """Negative coordinates are clamped to (0, 0)."""
        row, col = pixel_to_grid(-100, -200)
        assert (row, col) == (0, 0)

    def test_formula_correctness(self):
        """Verify the exact formula: col = floor(x * cols / W)."""
        x, y = 480, 270   # 1/4 of 1920, 1/4 of 1080
        expected_col = int(x * GRID_COLS / FRAME_WIDTH)   # 2
        expected_row = int(y * GRID_ROWS / FRAME_HEIGHT)  # 2
        assert pixel_to_grid(x, y) == (expected_row, expected_col)

    def test_custom_resolution(self):
        """Custom frame size: 640×480 → 8×8 grid."""
        row, col = pixel_to_grid(320, 240, frame_width=640, frame_height=480)
        assert row == 4
        assert col == 4


# ══════════════════════════════════════════════════════════════════════════════
# bbox_to_grid_range
# ══════════════════════════════════════════════════════════════════════════════

class TestBBoxToGridRange:

    def test_small_box_in_top_left(self):
        """A small bbox in the top-left corner maps to cells near (0,0)."""
        bbox = (0, 0, 100, 100)
        r_min, c_min, r_max, c_max = bbox_to_grid_range(bbox, margin=0)
        assert r_min == 0 and c_min == 0

    def test_full_frame_spans_entire_grid(self):
        """A bbox covering the entire frame should span all 8×8 cells."""
        bbox = (0, 0, FRAME_WIDTH - 1, FRAME_HEIGHT - 1)
        r_min, c_min, r_max, c_max = bbox_to_grid_range(bbox, margin=0)
        assert r_min == 0 and c_min == 0
        assert r_max == GRID_ROWS - 1 and c_max == GRID_COLS - 1

    def test_margin_expands_range(self):
        """Margin=1 should expand the range by 1 cell on each side (when possible)."""
        # A bbox tightly in the middle.
        bbox = (960, 540, 1200, 700)
        r0, c0, r1, c1 = bbox_to_grid_range(bbox, margin=0)
        rm, cm, r1m, c1m = bbox_to_grid_range(bbox, margin=1)
        assert rm <= r0
        assert cm <= c0
        assert r1m >= r1
        assert c1m >= c1

    def test_margin_clamped_at_borders(self):
        """Margin on a corner bbox should not produce out-of-range indices."""
        bbox = (0, 0, 50, 50)
        r_min, c_min, r_max, c_max = bbox_to_grid_range(bbox, margin=5)
        assert r_min >= 0 and c_min >= 0
        assert r_max < GRID_ROWS and c_max < GRID_COLS

    def test_degenerate_zero_area_box(self):
        """A point bbox (x_min==x_max, y_min==y_max) should still return a valid range."""
        bbox = (960, 540, 960, 540)
        r_min, c_min, r_max, c_max = bbox_to_grid_range(bbox, margin=0)
        assert r_min <= r_max
        assert c_min <= c_max

    def test_inverted_bbox_handled(self):
        """A bbox where x_min > x_max should not crash (defensive handling)."""
        bbox = (1200, 700, 960, 540)   # inverted
        r_min, c_min, r_max, c_max = bbox_to_grid_range(bbox, margin=0)
        assert r_min <= r_max
        assert c_min <= c_max

    def test_result_within_grid_bounds(self):
        """All returned indices must be within [0, 7]."""
        test_bboxes = [
            (0, 0, 1919, 1079),
            (500, 200, 800, 600),
            (1800, 900, 1920, 1080),
        ]
        for bbox in test_bboxes:
            r_min, c_min, r_max, c_max = bbox_to_grid_range(bbox)
            assert 0 <= r_min <= r_max <= GRID_ROWS - 1, f"row out of range for {bbox}"
            assert 0 <= c_min <= c_max <= GRID_COLS - 1, f"col out of range for {bbox}"


# ══════════════════════════════════════════════════════════════════════════════
# generate_mask
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateMask:

    def test_empty_detections_all_on(self):
        """No objects → all LEDs on (all False)."""
        mask = generate_mask([])
        assert mask.shape == (GRID_ROWS, GRID_COLS)
        assert mask.dtype == bool
        assert not mask.any()   # no cell should be True (OFF)

    def test_full_frame_bbox_dims_all(self):
        """A bbox spanning the whole frame should dim every LED."""
        bbox = (0, 0, FRAME_WIDTH, FRAME_HEIGHT)
        mask = generate_mask([bbox])
        assert mask.all()   # every cell should be True (OFF)

    def test_left_half_dims_left_columns(self):
        """Bbox on left half → roughly left 4 columns dimmed."""
        bbox = (0, 0, FRAME_WIDTH // 2, FRAME_HEIGHT)
        mask = generate_mask([bbox], margin=0)
        # At minimum, column 0 should be OFF.
        assert mask[:, 0].any()

    def test_mask_shape_and_dtype(self):
        """Return value must always be (8,8) bool array."""
        mask = generate_mask([(100, 100, 500, 500)])
        assert mask.shape == (GRID_ROWS, GRID_COLS)
        assert mask.dtype == bool

    def test_multiple_bboxes_union(self):
        """Two non-overlapping bboxes dim both their zones."""
        left_bbox  = (0,    0, 400, FRAME_HEIGHT)
        right_bbox = (1500, 0, FRAME_WIDTH, FRAME_HEIGHT)
        mask_left  = generate_mask([left_bbox],  margin=0)
        mask_right = generate_mask([right_bbox], margin=0)
        mask_both  = generate_mask([left_bbox, right_bbox], margin=0)

        # Union: both-mask must have at least as many True cells as either alone.
        assert mask_both.sum() >= mask_left.sum()
        assert mask_both.sum() >= mask_right.sum()

    def test_two_cars_spanning_multiple_zones(self):
        """Bboxes that span multiple LED zones dim all covered cells."""
        # Car spanning columns ~1–5 (x 240–1200 of 1920).
        bbox = (240, 0, 1200, FRAME_HEIGHT)
        mask = generate_mask([bbox], margin=0)
        dimmed_cols = [c for c in range(GRID_COLS) if mask[:, c].any()]
        assert len(dimmed_cols) >= 4, (
            f"Expected ≥4 columns dimmed for wide bbox, got: {dimmed_cols}"
        )

    def test_single_pixel_bbox(self):
        """A 1×1 bbox should dim at least one LED cell."""
        bbox = (960, 540, 961, 541)
        mask = generate_mask([bbox], margin=0)
        assert mask.sum() >= 1

    def test_deterministic(self):
        """Same input always produces identical mask."""
        bboxes = [(300, 200, 900, 700), (1200, 100, 1700, 500)]
        mask1 = generate_mask(bboxes)
        mask2 = generate_mask(bboxes)
        np.testing.assert_array_equal(mask1, mask2)


# ══════════════════════════════════════════════════════════════════════════════
# Integration: pixel_to_grid + generate_mask consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_pixel_centre_is_in_dimmed_zone(self):
        """The pixel at the centre of a bbox should land in a dimmed LED zone."""
        x1, y1, x2, y2 = 400, 300, 800, 700
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        row, col = pixel_to_grid(cx, cy)

        mask = generate_mask([(x1, y1, x2, y2)], margin=0)
        assert mask[row, col], (
            f"Centre pixel ({cx},{cy}) → grid ({row},{col}) not in dimmed zone"
        )

    def test_non_overlapping_bboxes_independent(self):
        """Two far-apart bboxes should dim different, non-overlapping LED zones."""
        bbox_a = (0,    0, 200, 200)    # top-left corner
        bbox_b = (1800, 900, 1920, 1080)  # bottom-right corner

        mask_a = generate_mask([bbox_a], margin=0)
        mask_b = generate_mask([bbox_b], margin=0)

        # The intersection of their dimmed cells should be empty.
        overlap = mask_a & mask_b
        assert not overlap.any(), "Non-overlapping bboxes should not share dimmed cells"
