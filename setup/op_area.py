# Takes tracked object as input
# Output the area in pixel^2 
def area(obj):
	x1 = obj.past_detections[-1].points[0][0]
	y1 = obj.past_detections[-1].points[0][1]
	x2 = obj.past_detections[-1].points[1][0]
	y2 = obj.past_detections[-1].points[1][1]
	length = abs(x2-x1)
	width = abs(y2-y1)
	return length*width