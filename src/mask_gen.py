"""
mask_gen.py — Bounding Box → LED Grid Mapping
==============================================
Converts tracked object bounding boxes into an 8×8 boolean LED mask.

Convention:
    True  = LED OFF  (object detected in this zone → prevent glare)
    False = LED ON   (clear road → full illumination)

Week 1: Linear mapping via frame dimensions.
Week 3: Replace _linear_map() with calibration JSON lookup.
"""

import numpy as np
from typing import List, Tuple, Optional
import json
import os

# ── Grid / frame constants ────────────────────────────────────────────────────
GRID_ROWS: int = 16
GRID_COLS: int = 16
FRAME_WIDTH: int = 1920
FRAME_HEIGHT: int = 1080

# Padding in LED units added around every bounding box to prevent edge-glare.
# A value of 1 means one extra LED cell on each side.
MARGIN: int = 1

BBox = Tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max)  — pixel coords
GridCell = Tuple[int, int]          # (row, col)  — 0-indexed


# ── Core coordinate mapping ───────────────────────────────────────────────────

def pixel_to_grid(
    x: float,
    y: float,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
) -> GridCell:
    """Map a single pixel coordinate (x, y) to an LED grid cell (row, col).

    Uses the linear formula from the project spec:
        Grid_col = floor( x * TotalColumns / FrameWidth  )
        Grid_row = floor( y * TotalRows    / FrameHeight )

    Args:
        x:            Horizontal pixel position (0 … frame_width-1).
        y:            Vertical pixel position   (0 … frame_height-1).
        frame_width:  Source frame width  in pixels (default 1920).
        frame_height: Source frame height in pixels (default 1080).
        grid_rows:    Number of LED rows    (default 8).
        grid_cols:    Number of LED columns (default 8).

    Returns:
        (row, col) — both 0-indexed and clamped to valid grid range.

    Edge-case handling:
        • Coordinates exactly equal to frame_width / frame_height (e.g. from
          rounding) would produce an out-of-range index without clamping.
        • np.clip ensures the result always lands inside [0, grid_size - 1].
    """
    col = int(x * grid_cols / frame_width)
    row = int(y * grid_rows / frame_height)

    col = int(np.clip(col, 0, grid_cols - 1))
    row = int(np.clip(row, 0, grid_rows - 1))

    return (row, col)


def bbox_to_grid_range(
    bbox: BBox,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    margin: int = MARGIN,
) -> Tuple[int, int, int, int]:
    """Convert a bounding box to the inclusive LED grid cell range it covers.

    A car spanning x=[400, 1200] on a 1920-wide frame touches *multiple* LED
    columns.  This function returns (row_min, col_min, row_max, col_max) —
    the rectangular block of LED cells that the bbox overlaps, expanded by
    `margin` cells on each side.

    Args:
        bbox:   (x_min, y_min, x_max, y_max) in pixel coordinates.
        margin: Extra LED cells of padding around the mapped region (default 1).

    Returns:
        (row_min, col_min, row_max, col_max) — inclusive, clamped to grid.
    """
    x_min, y_min, x_max, y_max = bbox

    # Map each corner of the bounding box to grid space.
    row_min, col_min = pixel_to_grid(x_min, y_min, frame_width, frame_height, grid_rows, grid_cols)
    row_max, col_max = pixel_to_grid(x_max, y_max, frame_width, frame_height, grid_rows, grid_cols)

    # Ensure min ≤ max (defensive against degenerate boxes where x_min > x_max).
    row_min, row_max = min(row_min, row_max), max(row_min, row_max)
    col_min, col_max = min(col_min, col_max), max(col_min, col_max)

    # Apply ±margin padding and clamp to valid grid indices.
    row_min = int(np.clip(row_min - margin, 0, grid_rows - 1))
    col_min = int(np.clip(col_min - margin, 0, grid_cols - 1))
    row_max = int(np.clip(row_max + margin, 0, grid_rows - 1))
    col_max = int(np.clip(col_max + margin, 0, grid_cols - 1))

    return (row_min, col_min, row_max, col_max)


# ── Mask generation ───────────────────────────────────────────────────────────

def generate_mask(
    bboxes: List[BBox],
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    margin: int = MARGIN,
    calibration_path: Optional[str] = None,
) -> np.ndarray:
    """Generate an 8×8 boolean LED mask from a list of bounding boxes.

    True  → LED OFF  (object present — dim this zone)
    False → LED ON   (clear — full brightness)

    Handles the multi-zone spanning case by iterating over every cell inside
    the mapped grid range for each bounding box.

    Args:
        bboxes:           List of (x_min, y_min, x_max, y_max) pixel coords.
        frame_width:      Source frame width  (pixels).
        frame_height:     Source frame height (pixels).
        grid_rows:        LED grid rows.
        grid_cols:        LED grid cols.
        margin:           Padding cells around each bounding box.
        calibration_path: (Week 3) Path to calibration.json.  When provided,
                          non-linear zone mapping replaces the linear formula.

    Returns:
        np.ndarray of shape (grid_rows, grid_cols) and dtype bool.
        True = dim, False = bright.
    """
    # Start fully ON — all LEDs lit.
    mask = np.zeros((grid_rows, grid_cols), dtype=bool)

    if not bboxes:
        return mask   # No objects → all LEDs bright, no change needed.

    # Week 3 hook: load calibration mapping if available.
    zone_map = _load_calibration(calibration_path) if calibration_path else None

    for bbox in bboxes:
        if zone_map is not None:
            cells = _calibrated_cells(bbox, zone_map)
        else:
            row_min, col_min, row_max, col_max = bbox_to_grid_range(
                bbox, frame_width, frame_height, grid_rows, grid_cols, margin
            )
            # Mark every cell in the rectangular block as OFF.
            cells = [
                (r, c)
                for r in range(row_min, row_max + 1)
                for c in range(col_min, col_max + 1)
            ]

        for r, c in cells:
            if 0 <= r < grid_rows and 0 <= c < grid_cols:
                mask[r, c] = True

    return mask


# ── Week 3 calibration hook (stub) ───────────────────────────────────────────

def _load_calibration(path: str) -> Optional[dict]:
    """Load calibration.json produced by focal_calibration.py.

    Returns None if the file is missing or malformed so the system
    degrades gracefully to linear mapping.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _calibrated_cells(bbox: BBox, zone_map: dict) -> List[GridCell]:
    """(Week 3 stub) Map bbox → cells using non-linear calibration data.

    Replace this body when calibration.json schema is finalised.
    Falls back to linear mapping on any error.
    """
    # TODO (Week 3): implement lookup into zone_map["grid_zones"] or similar.
    row_min, col_min, row_max, col_max = bbox_to_grid_range(bbox)
    return [
        (r, c)
        for r in range(row_min, row_max + 1)
        for c in range(col_min, col_max + 1)
    ]


# ── Utility ───────────────────────────────────────────────────────────────────

def mask_summary(mask: np.ndarray) -> str:
    """Return a compact ASCII representation of the mask for debugging."""
    lines = []
    for row in mask:
        lines.append(" ".join("■" if cell else "□" for cell in row))
    off_count = int(mask.sum())
    on_count = mask.size - off_count
    lines.append(f"  ■ OFF={off_count}  □ ON={on_count}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke-test with two bounding boxes on a 1920×1080 frame.
    test_bboxes: List[BBox] = [
        (200,  300,  600,  700),   # Car on the left — spans ~cols 0-2, rows 2-5
        (1400, 200, 1800,  600),   # Car on the right — spans ~cols 5-7, rows 1-4
    ]

    mask = generate_mask(test_bboxes)
    print("Generated 8×8 LED mask  (■ = OFF / dim,  □ = ON / bright):\n")
    print(mask_summary(mask))

    print("\nRaw numpy mask (True = OFF):")
    print(mask.astype(int))

    # Edge-case: bounding box exactly at frame borders.
    edge_cases: List[BBox] = [
        (0, 0, 1919, 1079),         # Entire frame — should dim all LEDs.
        (1919, 1079, 1920, 1080),   # 1-pixel box at bottom-right corner.
        (960, 540, 960, 540),       # Zero-area box at frame centre.
    ]

    print("\nEdge-case tests:")
    for bb in edge_cases:
        m = generate_mask([bb])
        print(f"  bbox={bb}  →  {int(m.sum())} LEDs dimmed")
