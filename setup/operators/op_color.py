import numpy as np
# Define predefined colors
colors = {
	"RED": np.array([255, 0, 0]),
	"GREEN": np.array([0, 255, 0]),
	"BLUE": np.array([0, 0, 255]),
	"YELLOW": np.array([255, 255, 0]),
	"BLACK": np.array([0, 0, 0]),
	"WHITE": np.array([255, 255, 255])
}

def closest_color(dominant_color):
	min_distance = float('inf')
	closest_color = None
	for color_name, color_value in colors.items():
		distance = np.linalg.norm(dominant_color - color_value)
		if distance < min_distance:
			min_distance = distance
			closest_color = color_name
	return closest_color

#Input -> frame and tracked object
#Output -> Color among the the colors defined in dictionary
def color(image,obj):
	x1 = obj.past_detections[-1].points[0][0]
	y1 = obj.past_detections[-1].points[0][1]
	x2 = obj.past_detections[-1].points[1][0]
	y2 = obj.past_detections[-1].points[1][1]
	roi = image[int(y1):int(y2), int(x1):int(x2)]  # Region of interest
	dominant_color = np.mean(roi, axis=(0, 1))
	closest_color_of_obj = closest_color(dominant_color)
	return closest_color_of_obj