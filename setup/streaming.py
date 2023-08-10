import torch
import pandas
import cv2
from PIL import Image
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Start Webcam Stream and Detection
cap = cv2.VideoCapture(0)  # Open the default camera (index 0)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)

    # Perform object detection
    results = model(pil_image)
    
    # Print the confidence scores of the objects
    print(results.pandas().xyxy[0])

    # Display detected objects
    detected_frame = results.render()[0]
    cv2.imshow('Object Detection', cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR))
    
    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time

    # Print the FPS
    print(f"FPS: {fps:.2f}", end="\r")  # Use carriage return to overwrite the line
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
