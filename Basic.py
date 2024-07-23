import torch
import cv2
import numpy as np
import time

# Mouse callback function to capture mouse movements
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)  # Print the (x, y) coordinates of the mouse

# Set up the OpenCV window and mouse callback
cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

# Load the YOLOv5 model from the Ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize video capture from a file
cap = cv2.VideoCapture('mumbaitraffic.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

count = 0
start_time = time.time()
total_inference_time = 0.0

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # Check if frame was read correctly
        print("End of video or unable to read the frame.")
        break
    count += 1

    if count % 3 != 0:  # Process every 3rd frame to reduce workload
        continue

    # Resize the frame for the model
    frame = cv2.resize(frame, (1020, 500))
    
    # Measure the inference time for the model
    inference_start_time = time.time()
    results = model(frame)  # Perform detection
    inference_end_time = time.time()
    
    # Accumulate total inference time
    inference_time = inference_end_time - inference_start_time
    total_inference_time += inference_time
    
    # Draw bounding boxes and labels on the frame
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        label = row['name']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
        cv2.putText(frame, str(label), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)  # Draw label 
    
    # Display the frame with detections
    cv2.imshow("ROI", frame)
    
    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Calculate total processing statistics
end_time = time.time()
total_time = end_time - start_time
total_frames = count
fps = total_frames / total_time
# Calculate average inference time per frame
avg_inference_time = total_inference_time / total_frames if total_frames > 0 else 0

# Print performance metrics
print("Average inference time:", avg_inference_time, "seconds")
print("Total frames:", total_frames)
print("Total time:", total_time, "seconds")
print("Average FPS:", fps)

# Release resources
cap.release()
cv2.destroyAllWindows()