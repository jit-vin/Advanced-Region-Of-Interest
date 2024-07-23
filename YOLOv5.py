import torch
import cv2
import numpy as np
import time

# Callback function to handle mouse move events and print the coordinates
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]  # Store mouse coordinates
        print(colorsBGR)     # Print the coordinates

# Create a named window for display and set a mouse callback
cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

# Load the YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

count = 0
total_frames = 0
total_inference_time = 0
# Open the video file for processing
cap = cv2.VideoCapture('mumbaitraffic.mp4')

# Define the area of interest (ROI) as a polygon
area = [(792, 296), (713, 353), (869, 401), (988, 351), (972, 313), (792, 296)]

# Start timing the processing
start_time = time.time()

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # Check if frame is valid
        break
    
    count += 1
    total_frames += 1

    # Process every 3rd frame to optimize performance
    if count % 3 != 0:
        continue
    
    # Resize the frame for consistent processing
    frame = cv2.resize(frame, (1020, 500))
    
    # Measure inference time
    start_inference_time = time.time()
    results = model(frame)  # Perform inference using YOLOv5
    inference_time = time.time() - start_inference_time
    total_inference_time += inference_time  # Accumulate inference time
    
    # Iterate over detection results
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = str(row['name'])  # Get the detected object's label
        
        # Calculate center of the bounding box
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        
        # Check if the detected point is inside the defined area (ROI)
        roi_test_result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)

        # If the point is within the area, draw the bounding box and label
        if roi_test_result >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, d, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)  # Draw label
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)  # Draw center point

    # Draw the area of interest polygon
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)       
    cv2.imshow("ROI", frame)  # Display the frame
    
    # Exit if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Calculate elapsed time and performance metrics
elapsed_time = time.time() - start_time
average_inference_time = total_inference_time / total_frames
average_fps = total_frames / elapsed_time

# Print the processing results
print("Average Inference Time:", average_inference_time, "seconds")
print("Total frames processed:", total_frames)
print("Total processing time:", elapsed_time, "seconds")
print("Average FPS:", average_fps)