import torch
import pandas as pd
import cv2  # OpenCV library for working with videos

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open the video file using OpenCV
video_path = '/home/aman/Video-database-analytics/setup/videos/input.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Open a log file for writing
log_file = open('detection_log.txt', 'w')

# Loop through each frame of the video
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame
    
    if not ret:
        break  # Break the loop when no frames are left
    
    # Convert the frame to a format that YOLOv5 can process (BGR to RGB)
    rgb_frame = frame[..., ::-1]
    
    # Perform object detection on the current frame
    results = model(rgb_frame)
    
    # Access and write detection results for the current frame to the log file
    detections = results.pandas().xyxy[0]
    log_file.write(f"Frame {frame_number}:\n")
    log_file.write(str(detections) + '\n')
    
    frame_number += 1

    # Display the frame with detected objects (optional)
    # Note: This will display the video with detections in real-time.
    # You might need to close the window to proceed with the loop.
    cv2.imshow('YOLOv5 Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the log file
log_file.close()

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
