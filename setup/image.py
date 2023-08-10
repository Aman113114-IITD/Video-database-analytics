import torch
import pandas
import matplotlib.pyplot as plt
# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = 'dogcat.jpeg'  # or file, Path, PIL, OpenCV, numpy, list
results = model(img)
print(results.pandas().xyxy[0])
# print(results)
# fig, ax = plt.subplots(figsize=(16, 12))
# ax.imshow(results.render()[0])
# plt.savefig("dogcat_labelled.jpeg")