# If bounding box of B is completely lying covers by A then true else false;
def covers(A,B)->bool :
    if ((B.past_detections[-1].points[0][0]>=A.past_detections[-1].points[0][0]) and (B.past_detections[-1].points[0][1]>=A.past_detections[-1].points[0][1]) and (B.past_detections[-1].points[1][0]<=A.past_detections[-1].points[1][0]) and (B.past_detections[-1].points[1][1]<=A.past_detections[-1].points[1][1])) :
        return True
    else :
        return False