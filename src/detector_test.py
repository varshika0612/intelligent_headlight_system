import cv2
from detector import Detector

det = Detector()
cap = cv2.VideoCapture(0)   # 0 = webcam

while True:
    ret, frame = cap.read()
    if not ret: break

    detections = det.detect(frame)
    for d in detections:
        x1,y1,x2,y2 = d.bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Test", frame)
    for d in detections:
        print(d.distance, d.class_name)
    if cv2.waitKey(1) == ord('q'): break