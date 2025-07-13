from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained YOLO model
model = YOLO(r"runs\detect\detected 43\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run detection
    results = model(frame)
    boxes = results[0].boxes
    labels = results[0].names

    elephant_detected = False

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        name = labels[cls_id]

        if name.lower() == "elephant" and conf >= 0.3:
            elephant_detected = True

            # Bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Draw bounding box and confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show detection message
            cv2.putText(frame, "✅ Animal detected in the farm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 3)

            break  # Exit after first valid elephant

    # ✅ Always show the webcam frame (with or without detection)
    cv2.imshow("Elephant Detection", frame)

    # Break on ESC or 'q' key
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()