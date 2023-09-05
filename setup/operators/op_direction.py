import math

# Direction is expressed as tuple with signed values, where (a,b) means a units in positive x direction and b units in positive y direction
def current_direction(object)->tuple[float,float] :
    nof = len(object.past_detections)
    centroids=[]
    for i in range(nof-2,nof) :
        a=object.past_detections[i].points[0][0]+object.past_detections[i].points[1][0]
        a=a/2
        b=object.past_detections[i].points[0][1]+object.past_detections[i].points[1][1]
        b=b/2
        centroids.append((a,b))
    direction=(0,0)
    for i in range(1) :
        direction=((centroids[i+1][0]-centroids[i][0]),(centroids[i+1][1]-centroids[i][1]))
    return direction

def average_direction(object)->tuple[float,float] :
    nof = len(object.past_detections)
    centroids=[]
    for i in [0,-1] :
        a=object.past_detections[i].points[0][0]+object.past_detections[i].points[1][0]
        a=a/2
        b=object.past_detections[i].points[0][1]+object.past_detections[i].points[1][1]
        b=b/2
        centroids.append((a,b))
    direction=(0,0)
    for i in range(1) :
        direction=((centroids[i+1][0]-centroids[i][0]),(centroids[i+1][1]-centroids[i][1]))
    return direction