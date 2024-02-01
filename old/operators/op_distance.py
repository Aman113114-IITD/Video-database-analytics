import math

def calculate_centroid(obj):
    x1 = obj.past_detections[-1].points[0][0]
    y1 = obj.past_detections[-1].points[0][1]
    x2 = obj.past_detections[-1].points[1][0]
    y2 = obj.past_detections[-1].points[1][1]
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    return centroid_x, centroid_y

def distance(obj1,obj2):
    c1,c2 = calculate_centroid(obj1)
    c3,c4 = calculate_centroid(obj2)
    # Check for distance between centroids
    centroid_distance = math.sqrt((c1 - c3) ** 2 + (c2 - c4) ** 2)
    return centroid_distance
