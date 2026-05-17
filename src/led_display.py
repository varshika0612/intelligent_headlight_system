import cv2
import numpy as np

GRID_ROWS = 8
GRID_COLS = 8
LED_SIZE = 60       # pixel size of each LED cell
PADDING = 10

ON_COLOR  = (0, 220, 255)    # bright yellow-white
OFF_COLOR = (30, 30, 30)     # dark

class LEDDisplay:
    def __init__(self):
        h = GRID_ROWS * LED_SIZE + 2 * PADDING
        w = GRID_COLS * LED_SIZE + 2 * PADDING
        self.canvas_shape = (h, w, 3)

    def render(self, mask: np.ndarray) -> np.ndarray:
        """
        mask: 8x8 boolean numpy array
              True  = LED OFF (blocking glare)
              False = LED ON  (illuminating road)
        """
        canvas = np.zeros(self.canvas_shape, dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                color = OFF_COLOR if mask[r, c] else ON_COLOR
                x1 = PADDING + c * LED_SIZE + 3
                y1 = PADDING + r * LED_SIZE + 3
                x2 = x1 + LED_SIZE - 6
                y2 = y1 + LED_SIZE - 6
                cv2.rectangle(canvas, (x1,y1), (x2,y2), color, -1)
                # add a subtle circle highlight for realism
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if not mask[r, c]:
                    cv2.circle(canvas, (cx, cy), 8, (255,255,255), -1)

        return canvas

# # Quick standalone test
# if __name__ == "__main__":
#     display = LEDDisplay()
#     # test: dim top-right quadrant
#     mask = np.zeros((8, 8), dtype=bool)
#     mask[0:4, 4:8] = True
#     img = display.render(mask)
#     cv2.imshow("LED Matrix", img)
#     cv2.waitKey(0)