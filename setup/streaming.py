import torch
import cv2
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Start Webcam Stream and Detection
cap = cv2.VideoCapture(0)  # Open the default camera (index 0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)

    # Perform object detection
    results = model(pil_image)

    # Display detected objects
    detected_frame = results.render()[0]
    cv2.imshow('Object Detection', cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
