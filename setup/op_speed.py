import math

# Speed is basically calculated as relative to other objects in units of pixels moved per frames taken
def average_speed(object)->float :
    nof = len(object.past_detections)
    centroids=[]
    for i in range(nof) :
        a=object.past_detections[i].points[0][0]+object.past_detections[i].points[1][0]
        a=a/2
        b=object.past_detections[i].points[0][1]+object.past_detections[i].points[1][1]
        b=b/2
        centroids.append((a,b))
    speed=0
    for i in range(nof-1) :
        speed+=math.sqrt(pow(centroids[i+1][0]-centroids[i][0],2)+pow(centroids[i+1][1]-centroids[i][1],2))
    speed=speed/(nof-1)
    return speed

def current_speed(object)->float :
    nof = len(object.past_detections)
    centroids=[]
    for i in range(nof-2,nof) :
        a=object.past_detections[i].points[0][0]+object.past_detections[i].points[1][0]
        a=a/2
        b=object.past_detections[i].points[0][1]+object.past_detections[i].points[1][1]
        b=b/2
        centroids.append((a,b))
    speed=0
    for i in range(1) :
        speed+=math.sqrt(pow(centroids[i+1][0]-centroids[i][0],2)+pow(centroids[i+1][1]-centroids[i][1],2))
    return speed