import torch
import numpy as np

def compute_average_bbox(det):
    xmin, ymin, xmax, ymax = det['xmin'], det['ymin'], det['xmax'], det['ymax']
    return (xmin + xmax) / 2, (ymin + ymax) / 2

def comparator(model1, model2, epsilon, epsilon2, image):
    model1_appl = torch.hub.load('ultralytics/yolov5', model1)
    model2_appl = torch.hub.load('ultralytics/yolov5', model2)
    result1 = model1_appl(image)
    result2 = model2_appl(image)
    detections1 = result1.pandas().xyxy[0]
    detections2 = result2.pandas().xyxy[0]
    
    num_objects1 = len(detections1)
    num_objects2 = len(detections2)
    
    if num_objects1 != num_objects2:
        return False
    else:
        object_dict = {}
        
        for _, det1 in detections1.iterrows():
            object_name = det1['name']
            confidence_score = det1['confidence']
            avg_bbox = compute_average_bbox(det1)
            
            if object_name in object_dict:
                object_dict[object_name].append((avg_bbox, confidence_score))
            else:
                object_dict[object_name] = [(avg_bbox, confidence_score)]
        
        for _, det2 in detections2.iterrows():
            object_name = det2['name']
            confidence_score = det2['confidence']
            avg_bbox = compute_average_bbox(det2)
            
            if object_name not in object_dict:
                return False
            
            matched = False
            for avg_bbox_ref, conf_score_ref in object_dict[object_name]:
                if np.linalg.norm(np.array(avg_bbox_ref) - np.array(avg_bbox)) < epsilon2 and \
                   abs(conf_score_ref - confidence_score) <= epsilon:
                    matched = True
                    break
            
            if not matched:
                return False
        
        return True

# Example usage
image_path = 'sample.jpg'  # Replace with the path to your image
model1_name = 'yolov5s'
model2_name = 'yolov5m'
epsilon = 0.1
epsilon2 = 5.0

result = comparator(model1_name, model2_name, epsilon, epsilon2, image_path)
if result:
    print("The models are similar.")
else:
    print("The models are not similar.")
