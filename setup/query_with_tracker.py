import torch
import cv2
import matplotlib.pyplot as plt
import time
# import queue
import argparse
from typing import List, Optional, Union
import numpy as np
import norfair
from norfair import Detection, Paths, Tracker, Video
from collections import deque

# functions for tarcking objects in video
def center(points):
    return [np.mean(np.array(points), axis=0)]

# functions for tarcking objects in video
def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections


# start time of program
start_time = time.time()

# thresholds for object tracing
DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000
Yolov5_model = 'yolov5s'

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load YOLOv5 model and move it to GPU
model = torch.hub.load('ultralytics/yolov5', Yolov5_model)
model.to(device)

# Open the video capture
input_video_path = 'input2.mp4'  # Change to your input video path

# Get video properties
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(5))

# Define the codec and create VideoWriter object
# output_video_path = 'output_video.mp4'  # Change to your output video path
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# log_file = open('detection_log.txt', 'w')



# Function to answer query for finding truck in a given video
def istruck(start,end,chunk_size,step_size,object,threshold) :

    distance_function = "euclidean"
    distance_threshold = (
        DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    # final answer
    ans=[]

    # Open the Video Capture
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(5))

    if not cap.isOpened():
        print("Error opening video file.")
        return
    
    # indices range to be processed based on starting and ending time passed in function
    start_frame = fps*start
    end_frame = fps*end

    # Array to store output per frame with confidence score
    output= deque()

    # process video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    valid_frames = 0
    while cap.isOpened() and start_frame < end_frame:

        start_frame+=1

        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)

        # do object tracking
        detections = yolo_detections_to_norfair_detections(
            results, track_points="centroid"
        )
        # print(detections)
        tracked_objects = tracker.update(detections=detections)

        # detections = results.pandas().xyxy[0]
        # confidence = 0
        objects=[]

        for obj in tracked_objects:
            if (obj.label==object) :
                objects.append(obj.id)
                
        # for _,objects in detections.iterrows() :
        #     if (objects['name']==object) :
        #         confidence=max(confidence,objects['confidence'])
        
        # confidence that object is present in frame
        output.append(objects)

        frame_count += 1

        if (len(objects)>0) :
            valid_frames+=1

        # if (confidence>=threshold) :
        #     valid_frames+=1

        if (frame_count == chunk_size*fps) :
            if (valid_frames>0) :
                do=set()
                for olist in output :
                    for obj in olist :
                        do.add(obj)
                ans.append((start_frame/fps-chunk_size,start_frame/fps,len(do),valid_frames))
                do.clear()
            
            for i in range(step_size*fps) :
                frame_count-=1
                if (len(output.popleft())>0) :
                    valid_frames-=1

    cap.release()
    return ans
        



# Loop through each frame of the video
# frame_number = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

    # Perform object detection on the frame
    # results = model(frame)
    # rendered_frame = results.render()[0]

    # Convert the rendered frame to BGR format
    # rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

    # Write the frame with bounding boxes to the output video
    # out.write(rendered_frame)

    # Access and write detection results for the current frame to the log file
    # detections = results.pandas().xyxy[0]
    # log_file.write(f"Frame {frame_number}:\n")
    # log_file.write(str(detections) + '\n')
    
    # frame_number += 1


    # Display the frame
    # cv2.imshow('Video Analytics',rendered_frame)

    # Exit when 'q' key is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Close the log file
# log_file.close()

# Release video capture and writer, and close all windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# input1=int(input("start time in sec: "))
# input2=int(input("end time in sec: "))
# input3=int(input("interval size in sec: "))
# input4=int(input("interval step size in sec: "))
# input5=input("object to be detected: ")
# input6=float(input("threshold for confidence value: "))

# print(istruck(input1,input2,input3,input4,input5,input6))
print(istruck(5,30,5,1,2,0.8))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)
