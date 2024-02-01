#Input -> Takes the tracked object
#Output -> Returns a boolean value true or false
def intersect(obj1,obj2):

    x11 = obj1.past_detections[-1].points[0][0]
    y11 = obj1.past_detections[-1].points[0][1]
    x12 = obj1.past_detections[-1].points[1][0]
    y12 = obj1.past_detections[-1].points[1][1]

    x21 = obj1.past_detections[-1].points[0][0]
    y21 = obj1.past_detections[-1].points[0][1]
    x22 = obj1.past_detections[-1].points[1][0]
    y22 = obj1.past_detections[-1].points[1][1]

    # Check for overlap in both x and y directions
    overlap_x = max(0, min(x12, x22) - max(x11, x21))
    overlap_y = max(0, min(y12, y22) - max(y11, y21))

    # Check for overlap and centroid distance
    if overlap_x > 0 and overlap_y > 0:
        return True
    else:
        return False