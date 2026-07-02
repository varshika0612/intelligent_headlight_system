"""
led_display.py — Simulated 8×8 LED Matrix (OpenCV)
====================================================
Renders the boolean mask produced by mask_gen.py as a visual LED grid.

    True  → LED OFF → dark grey  (#2a2a2a)
    False → LED ON  → bright yellow-white (#FFE066)

Hackathon fallback if physical hardware isn't available.
"""

import numpy as np
from typing import Optional, List, Tuple

# Try to import OpenCV; fall back gracefully for CI / unit-test environments.
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[LEDDisplay] OpenCV not found — render() will return a numpy array only.")

# ── Colour constants (BGR for OpenCV) ─────────────────────────────────────────
LED_ON_COLOR:  Tuple[int, int, int] = (102, 224, 255)   # Yellow-white  #FFE066 → BGR
LED_OFF_COLOR: Tuple[int, int, int] = (42,  42,  42)    # Dark grey     #2a2a2a → BGR
BORDER_COLOR:  Tuple[int, int, int] = (20,  20,  20)
BACKGROUND:    Tuple[int, int, int] = (15,  15,  15)

# ── Layout ────────────────────────────────────────────────────────────────────
CELL_SIZE:   int = 60    # pixels per LED cell
CELL_MARGIN: int = 6     # gap between cells
CORNER_R:    int = 8     # corner radius for rounded cells


class LEDDisplay:
    """Renders an 8×8 boolean mask as a simulated LED panel."""

    def __init__(
        self,
        grid_rows: int = 16,
        grid_cols: int = 16,
        cell_size: int = CELL_SIZE,
        margin:    int = CELL_MARGIN,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_size = cell_size
        self.margin    = margin

        # Pre-compute canvas dimensions.
        self.canvas_h = grid_rows * (cell_size + margin) + margin
        self.canvas_w = grid_cols * (cell_size + margin) + margin

    # ── Public API ────────────────────────────────────────────────────────────

    def render(self, mask: np.ndarray) -> np.ndarray:
        """Convert a boolean mask into an LED panel image.

        Args:
            mask: np.ndarray of shape (grid_rows, grid_cols), dtype bool.
                  True → OFF,  False → ON.

        Returns:
            BGR image of shape (canvas_h, canvas_w, 3), dtype uint8.
        """
        canvas = np.full((self.canvas_h, self.canvas_w, 3), BACKGROUND, dtype=np.uint8)

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                color = LED_OFF_COLOR if mask[row, col] else LED_ON_COLOR
                self._draw_led(canvas, row, col, color)

        return canvas

    def render_with_overlay(
        self,
        mask:  np.ndarray,
        tracks: Optional[List[dict]] = None,
    ) -> np.ndarray:
        """Render LED panel with optional track-ID overlay text per cell.

        Useful for debugging: shows which track ID caused each LED to dim.
        """
        canvas = self.render(mask)

        if tracks and _CV2_AVAILABLE:
            for track in tracks:
                from mask_gen import bbox_to_grid_range
                r_min, c_min, r_max, c_max = bbox_to_grid_range(track["bbox"])
                label = f"{track['id']}"
                # Place text at the centre of the top-left cell of the range.
                cx, cy = self._cell_centre(r_min, c_min)
                cv2.putText(
                    canvas, label,
                    (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

        return canvas

    def show(self, mask: np.ndarray, window_name: str = "LED Matrix") -> None:
        """Convenience wrapper: render and display in an OpenCV window."""
        if not _CV2_AVAILABLE:
            print("[LEDDisplay] OpenCV unavailable — cannot show window.")
            return
        canvas = self.render(mask)
        cv2.imshow(window_name, canvas)

    # ── Internal drawing helpers ──────────────────────────────────────────────

    def _cell_top_left(self, row: int, col: int) -> Tuple[int, int]:
        x = self.margin + col * (self.cell_size + self.margin)
        y = self.margin + row * (self.cell_size + self.margin)
        return (x, y)

    def _cell_centre(self, row: int, col: int) -> Tuple[int, int]:
        x, y = self._cell_top_left(row, col)
        return (x + self.cell_size // 2, y + self.cell_size // 2)

    def _draw_led(
        self,
        canvas: np.ndarray,
        row:    int,
        col:    int,
        color:  Tuple[int, int, int],
    ) -> None:
        """Draw one LED cell onto the canvas using numpy slicing.

        Pure numpy — no OpenCV dependency — so it always works regardless
        of how cv2 was installed (headless, partial, venv vs system).
        """
        x, y = self._cell_top_left(row, col)
        x2 = x + self.cell_size
        y2 = y + self.cell_size

        # Fill the LED cell.
        canvas[y:y2, x:x2] = color

        # Inner glow for ON cells — a brighter concentric square.
        if color == LED_ON_COLOR:
            inset = 6
            glow = (
                min(255, color[0] + 40),
                min(255, color[1] + 40),
                min(255, color[2] + 40),
            )
            canvas[y + inset : y2 - inset, x + inset : x2 - inset] = glow


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _CV2_AVAILABLE:
        print("OpenCV not available — skipping visual test.")
    else:
        import cv2
        display = LEDDisplay()

        # Mask: first two columns dimmed (object on the left).
        mask = np.zeros((8, 8), dtype=bool)
        mask[:, :2] = True   # Left two columns OFF.

        canvas = display.render(mask)
        cv2.imshow("LED Matrix Test", canvas)
        print("Press any key in the OpenCV window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
