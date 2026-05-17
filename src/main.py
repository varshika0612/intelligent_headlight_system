import cv2
import time
import numpy as np
from detector import Detector
from tracker import SimpleTracker
from mask_gen import MaskGenerator
from led_display import LEDDisplay

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

detector  = Detector()
tracker   = SimpleTracker()
mask_gen  = MaskGenerator(frame_width=w, frame_height=h)
display   = LEDDisplay()

while True:
    ret, frame = cap.read()
    if not ret: break

    t = time.time()
    detections    = detector.detect(frame)
    tracked       = tracker.update(detections, t)
    mask          = mask_gen.generate(tracked)
    led_img       = display.render(mask)

    # draw bounding boxes on camera frame
    for obj in tracked:
        x1,y1,x2,y2 = obj.bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,100), 2)
        cv2.putText(frame, f"ID:{obj.track_id} {obj.class_name}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 1)

    # show both windows side by side
    frame_resized = cv2.resize(frame, (480, 380))
    led_resized   = cv2.resize(led_img, (480, 380))
    combined = np.hstack([frame_resized, led_resized])
    cv2.imshow("Camera | LED Matrix", combined)

    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()