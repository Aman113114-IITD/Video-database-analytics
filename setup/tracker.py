import argparse
from typing import List, Optional, Union
import numpy as np
import torch
import norfair
from norfair import Detection, Paths, Tracker, Video
import time
import cv2

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000


class YOLO:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load model
        self.model = torch.hub.load("ultralytics/yolov5", model_name, device=device)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


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



parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument(
    "--model-name", type=str, default="yolov5s", help="YOLOv5 model name"
)
parser.add_argument(
    "--img-size", type=int, default="720", help="YOLOv5 inference size (pixels)"
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default="0.25",
    help="YOLOv5 object confidence threshold",
)
parser.add_argument(
    "--iou-threshold", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS"
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,
    help="Filter by class: --classes 0, or --classes 0 2 3",
)
parser.add_argument(
    "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track-points",
    type=str,
    default="centroid",
    help="Track points: 'centroid' or 'bbox'",
)
args = parser.parse_args()

model = YOLO(args.model_name, device=args.device)

curr_time = time.time()

for input_path in args.files:
    video = Video(input_path=input_path)

    distance_function = "iou" if args.track_points == "bbox" else "euclidean"
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if args.track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    total_tracking_time = 0

    total_model_time = 0

    # Initialize variables to store the previous frame and its histogram
    prev_frame = None
    prev_hist = None
    threshold = 0.8

    for frame in video:
        # Calculate the histogram of the current frame
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize the histogram
        if prev_hist is not None:
          print(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_INTERSECT))
        # Compare histograms and apply the model only if frames differ significantly
        if prev_hist is None or cv2.compareHist(hist, prev_hist, cv2.HISTCMP_INTERSECT) < threshold:
            print("Frame is taken")
            curr_time1 = time.time()
            yolo_detections = model(
                frame,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                image_size=args.img_size,
                classes=args.classes,
            )
            # print(yolo_detections)
            curr_time2 = time.time()
            total_model_time += (curr_time2-curr_time1)
            detections = yolo_detections_to_norfair_detections(
                yolo_detections, track_points=args.track_points
            )
            # print(detections)
            tracked_objects = tracker.update(detections=detections)
            # print(tracked_objects)
            # for obj in tracked_objects:
            #     print(f"Tracked Object ID: {obj.id}, Class: {obj.label}")
            curr_time3 = time.time()
            total_tracking_time += (curr_time3-curr_time2)
            # if args.track_points == "centroid":
                # norfair.draw_points(frame, detections)
                # norfair.draw_tracked_objects(frame, tracked_objects)
            # elif args.track_points == "bbox":
                # norfair.draw_boxes(frame, detections)
                # norfair.draw_tracked_boxes(frame, tracked_objects)
            # video.write(frame)
        else:
            print("Frame is not taken")

        # Update the previous frame and its histogram
        prev_frame = frame.copy()
        prev_hist = hist
        
end_time = time.time()

print(f"Total time taken by model is {total_model_time}")
print(f"Total time taken by tracking is {total_tracking_time}")
print(f"Total time taken is {end_time-curr_time} seconds")