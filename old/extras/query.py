import torch
import cv2
import matplotlib.pyplot as plt
import time
import queue

start_time = time.time()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load YOLOv5 model and move it to GPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.to(device)

# Open the video capture
input_video_path = 'input2.mp4'  # Change to your input video path

# Get video properties
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(5))

# Define the codec and create VideoWriter object
# output_video_path = 'output_video.mp4'  # Change to your output video path
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# log_file = open('detection_log.txt', 'w')



# Function to answer query for finding truck in a given video
def istruck(start,end,chunk_size,step_size,object,threshold) :

    # final answer
    ans=[]

    # Open the Video Capture
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(5))

    if not cap.isOpened():
        print("Error opening video file.")
        return
    
    # indices range to be processed based on starting and ending time passed in function
    start_frame = fps*start
    end_frame = fps*end

    # Array to store output per frame with confidence score
    output= queue.Queue()

    # process video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    valid_frames = 0
    while cap.isOpened() and start_frame < end_frame:

        start_frame+=1

        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)
        detections = results.pandas().xyxy[0]
        confidence = 0 
        for _,objects in detections.iterrows() :
            if (objects['name']==object) :
                confidence=max(confidence,objects['confidence'])
        
        # confidence that object is present in frame
        output.put(confidence)
        frame_count += 1

        if (confidence>=threshold) :
            valid_frames+=1

        if (frame_count == chunk_size*fps) :
            if (valid_frames>0) :
                ans.append((start_frame/fps-chunk_size,valid_frames))
            
            for i in range(step_size*fps) :
                frame_count-=1
                if (output.get()>=threshold) :
                    valid_frames-=1

    cap.release()
    return ans
        



# Loop through each frame of the video
# frame_number = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

    # Perform object detection on the frame
    # results = model(frame)
    # rendered_frame = results.render()[0]

    # Convert the rendered frame to BGR format
    # rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

    # Write the frame with bounding boxes to the output video
    # out.write(rendered_frame)

    # Access and write detection results for the current frame to the log file
    # detections = results.pandas().xyxy[0]
    # log_file.write(f"Frame {frame_number}:\n")
    # log_file.write(str(detections) + '\n')
    
    # frame_number += 1


    # Display the frame
    # cv2.imshow('Video Analytics',rendered_frame)

    # Exit when 'q' key is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Close the log file
# log_file.close()

# Release video capture and writer, and close all windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()

input1=int(input("start time in sec: "))
input2=int(input("end time in sec: "))
input3=int(input("interval size in sec: "))
input4=int(input("interval step size in sec: "))
input5=input("object to be detected: ")
input6=float(input("threshold for confidence value: "))

print(istruck(input1,input2,input3,input4,input5,input6))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)
