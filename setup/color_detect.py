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

def extract_color_from_bbox(image, x1,y1,x2,y2):
    roi = image[int(y1):int(y2), int(x1):int(x2)]  # Region of interest
    dominant_color = np.mean(roi, axis=(0, 1))
    return dominant_color