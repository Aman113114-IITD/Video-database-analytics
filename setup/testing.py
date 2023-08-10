import torch
import cv2
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load YOLOv5 model and move it to GPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.to(device)

# Open the video capture
input_video_path = 'input.mp4'  # Change to your input video path
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec and create VideoWriter object
output_video_path = 'output_video.mp4'  # Change to your output video path
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

nm=0
while cap.isOpened():
    nm+=1
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)
    rendered_frame = results.render()[0]

    # Convert the rendered frame to BGR format
    rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

    # Write the frame with bounding boxes to the output video
    out.write(rendered_frame)

    # Display the frame
    cv2.imshow('Video Analytics',rendered_frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()

print("Nunmber of frames is ",nm)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)
