import torch
import cv2
import matplotlib.pyplot as plt
import time
import argparse
from typing import List, Optional, Union
import numpy as np
import norfair
from norfair import Detection, Paths, Tracker, Video
from collections import deque


# thresholds for object tracing
DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000
HISTOGRAM_THRESHOLD = 1


## YOLO CLASS
class YOLO:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("ultralytics/yolov5", model_name, device=device)
    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_height: int = 1080,
        image_width: int = 1080,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=(image_height,image_width))
        return detections

# functions for tarcking objects in video
def center(points):
    return [np.mean(np.array(points), axis=0)]
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

# Function to answer query for finding truck in a given video
def query(start,end,chunk_size,step_size,object,argum) : 

    # detecting the model to be used
    model = YOLO(args.model_name, device=args.device)

    # final answer
    ans=[]

    for input_video_path in argum.files:

        # Initialize tracker
        distance_function = "iou" if argum.track_points == "bbox" else "euclidean"
        distance_threshold = (
            DISTANCE_THRESHOLD_BBOX
            if argum.track_points == "bbox"
            else DISTANCE_THRESHOLD_CENTROID
        )
        tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
        )

        # Open the Video Capture
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(5))
        if not cap.isOpened():
            print("Error opening video file.")
            return
        
        # Indices range to be processed based on starting and ending time passed in function
        start_frame = fps*start
        end_frame = fps*end

        # Array to store output per frame with confidence score
        output= set()
        frame_count = 0
        valid_frames = 0

        # Process video from start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev_hist = None
        while cap.isOpened() and start_frame < end_frame:

            # Read next frame
            start_frame+=1
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the histogram of the current frame and Normalize it
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist = hist / hist.sum()

            # Process current frame
            if prev_hist is None or cv2.compareHist(hist, prev_hist, cv2.HISTCMP_INTERSECT) < HISTOGRAM_THRESHOLD:
                print("The frame is taken !! ",start_frame)
                prev_hist = hist

                # Perform object detection on the frame
                results = model(
                    frame,
                    conf_threshold=argum.conf_threshold,
                    iou_threshold=argum.iou_threshold,
                    image_height=argum.img_height,
                    image_width=argum.img_width,
                    classes=argum.classes,
                )

                # Do object tracking
                detections = yolo_detections_to_norfair_detections(
                    results, track_points=argum.track_points
                )
                tracked_objects = tracker.update(detections=detections)
                if (len(tracked_objects)>=0) :
                    valid_frames+=1
                for obj in tracked_objects:
                    if (obj.label==object) :
                        output.add(obj.id)          
            else:
                print("The frame is not taken !! ",start_frame)

            # If complete window is processed store results and reset for next window
            if (frame_count == chunk_size*fps) :
                ans.append((start_frame/fps-chunk_size,start_frame/fps,len(output),valid_frames))
                output.clear()
                frame_count=0
                valid_frames=0
                tracker = Tracker(
                    distance_function=distance_function,
                    distance_threshold=distance_threshold,
                )
        cap.release()
    return ans

##parser arguements
parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--model-name", type=str, default="yolov5x", help="YOLOv5 model name")
parser.add_argument("--img-height", type=int, default="1080", help="YOLOv5 inference height (pixels)")
parser.add_argument("--img-width", type=int, default="1080", help="YOLOv5 inference width (pixels)")
parser.add_argument("--conf-threshold",type=float,default="0.25",help="YOLOv5 object confidence threshold",)
parser.add_argument("--iou-threshold", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument("--classes",nargs="+",type=int,help="Filter by class: --classes 0, or --classes 0 2 3",)
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track-points",type=str,default="centroid",help="Track points: 'centroid' or 'bbox'",)
args = parser.parse_args()

# start time of program
start_time = time.time()
print(query(0,30,5,5,2,args))
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)
