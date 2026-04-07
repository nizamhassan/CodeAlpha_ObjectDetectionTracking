
from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Sort()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    # Collect detections
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    # Apply tracking
    if len(detections) > 0:
        tracks = tracker.update(detections)
    else:
        tracks = []

    # Draw results
    for t in tracks:
        x1, y1, x2, y2, track_id = map(int, t)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()