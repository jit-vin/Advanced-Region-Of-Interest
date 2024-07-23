import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Callback function to handle mouse events to display mouse coordinates
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)  # Print the coordinates of the mouse pointer

# Create a named window for displaying the video stream
cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)  # Set mouse callback for the window

# Load a pre-trained YOLOv10s model
model = YOLO("yolov10s.pt")

# Initialize counters
count = 0
total_frames = 0
total_inference_time = 0

# Open the video file
cap = cv2.VideoCapture('mumbaitraffic.mp4')

# Define the polygon area for detection as a series of vertices
area = [(792, 296), (713, 353), (869, 401), (988, 351), (972, 313), (792, 296)]

# Start timing for performance measurement
start_time = time.time()
while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no frames left

    count += 1
    total_frames += 1
    
    # Process every third frame to reduce workload
    if count % 3 != 0:
        continue

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (1020, 500))
    
    # Measure inference time for the model
    start_inference_time = time.time()
    results = model(frame)  # Run YOLO model inference on the frame
    inference_time = time.time() - start_inference_time
    total_inference_time += inference_time  # Accumulate inference time
    
    # Iterate over detection results
    for result in results:
        for row in result.boxes.data.tolist():  # Access detected boxes in results
            # Extract detection box coordinates, confidence, and class
            x1, y1, x2, y2, conf, cls = map(int, row[:6])  
            label = result.names[cls]  # Get class label
            
            # Calculate center coordinates of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Check if the center point is within the defined polygon area
            is_in_area = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            
            if is_in_area >= 0:
                # Draw bounding box and label only if within ROI
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                cv2.putText(frame, str(label), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)  # Label in Blue
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)  # Center point in Blue

    # Draw the ROI polygon on the frame
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)  # Red polygon
    cv2.imshow("ROI", frame)  # Display the frame in the window
    
    # Exit the loop on pressing the ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Calculate and print statistics at the end of processing
elapsed_time = time.time() - start_time
average_inference_time = total_inference_time / total_frames if total_frames > 0 else 0
average_fps = total_frames / elapsed_time if elapsed_time > 0 else 0

# Output performance metrics
print("Average Inference Time:", average_inference_time, "seconds")
print("Total frames processed:", total_frames)
print("Total processing time:", elapsed_time, "seconds")
print("Average FPS:", average_fps)