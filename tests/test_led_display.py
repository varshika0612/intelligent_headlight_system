"""
tests/test_led_display.py — Unit Tests for led_display.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from led_display import LEDDisplay, LED_ON_COLOR, LED_OFF_COLOR


class TestLEDDisplay:

    def setup_method(self):
        self.display = LEDDisplay()

    def test_render_shape(self):
        """Rendered image must have the right canvas dimensions."""
        mask   = np.zeros((8, 8), dtype=bool)
        canvas = self.display.render(mask)
        assert canvas.shape == (self.display.canvas_h, self.display.canvas_w, 3)
        assert canvas.dtype == np.uint8

    def test_all_on_produces_bright_pixels(self):
        """All-False mask → no dark cells — canvas average should be bright."""
        mask   = np.zeros((8, 8), dtype=bool)
        canvas = self.display.render(mask)
        assert canvas.mean() > 60    # brighter than a mostly-dark image

    def test_all_off_produces_dark_pixels(self):
        """All-True mask → all LEDs dim — canvas average should be dark."""
        mask   = np.ones((8, 8), dtype=bool)
        canvas = self.display.render(mask)
        assert canvas.mean() < 60    # darker than a mostly-bright image

    def test_partial_mask_brightness_between(self):
        """Partial mask brightness should sit between all-on and all-off."""
        mask_on  = np.zeros((8, 8), dtype=bool)
        mask_off = np.ones((8, 8), dtype=bool)
        mask_half        = np.zeros((8, 8), dtype=bool)
        mask_half[:4, :] = True

        bright = self.display.render(mask_on).mean()
        dark   = self.display.render(mask_off).mean()
        mid    = self.display.render(mask_half).mean()

        assert dark < mid < bright

    def test_cell_top_left_calculation(self):
        """Cell (0,0) should start at margin offset."""
        x, y = self.display._cell_top_left(0, 0)
        assert x == self.display.margin
        assert y == self.display.margin
