#!/usr/bin/env python
import cv2

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera (change to 1, 2, etc. if you have multiple cameras)
cap.set(cv2.CAP_PROP_FPS, 10)

# Set webcam resolution (adjust as needed)
width = 680# Adjust this to your camera's resolution
height =480   # Adjust this to your camera's resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Create a window to display the webcam feed
cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was read successfully, display it
    if ret:
    	cv2.circle(frame, (320,240), 3, (0, 0, 0), -1)
    	cv2.imshow("Webcam Feed", frame)
        

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

