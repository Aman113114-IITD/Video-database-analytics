import torch

def comparator(model1, model2, epsilon, image):
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
        confidence_dict = {}
        for _, det1 in detections1.iterrows():
            object_name = det1['name']
            confidence_score = det1['confidence']
            confidence_dict[object_name] = confidence_score
            
        for _, det2 in detections2.iterrows():
            object_name = det2['name']
            confidence_score = det2['confidence']
            
            if confidence_dict.get(object_name) is None:
                return False
            elif abs(confidence_score - confidence_dict[object_name]) > epsilon:
                return False
        
        return True

# Example usage
image_path = 'dogcat.jpeg'  # Replace with the path to your image
model1_name = 'yolov5s'
model2_name = 'yolov5m'
epsilon = 0.1

result = comparator(model1_name, model2_name, epsilon, image_path)
if result:
    print("The models are similar.")
else:
    print("The models are not similar.")
