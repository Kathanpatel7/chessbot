#!/usr/bin/env python

import cv2
import numpy as np
import csv


# Initialize the webcam (usually the default camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (adjust as needed)
width = 720 # Adjust this to your camera's resolution
height = 720   # Adjust this to your camera's resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)




# Wait for the Enter key to be pressed
print("Press Enter to capture an image...")
input()  # Wait for Enter key

# Capture a frame
ret, frame = cap.read()

if ret:
    # Display the captured image
    cv2.imshow("Input Image", frame)

    # Save the captured image to a file
    cv2.imwrite("captured_image.jpg", frame)
    print("Captured image saved as captured_image.jpg")
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define the number of rows and columns in the chessboard
    rows = 7  # Number of interior corners in a row - 1
    cols = 7  # Number of interior corners in a column - 1

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # Calculate the mean coordinates of each square and continue processing if corners are found
    if ret and corners is not None:
        centers = [corner[0] for corner in corners]  # Extract the centers of the corners

        for center in centers:
            center_x = int(center[0])
            center_y = int(center[1])
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

        # Calculate the mean of all detected centers
        mean_x = int(np.mean([x for x, _ in centers]))
        mean_y = int(np.mean([y for _, y in centers]))
        mean_center = (mean_x, mean_y)

                # Draw the mean center in green
        cv2.circle(frame, mean_center, 5, (0, 255, 0), -1)

        print("Mean Center:", mean_center)

        # Display the image with detected corners and mean center
        cv2.imshow('Chessboard Corners and Mean Center', frame)
        
        # Calculate the distance between corner 1 and corner 2 (for example)
        corner1 = corners[23][0]
        corner2 = corners[24][0]
        distance_pixels = np.linalg.norm(corner1 - corner2)
        
        # Define the size of each square and the number of squares in each row and column
        square_size = distance_pixels
        num_squares = 8

        # Calculate the size of the chessboard pattern
        pattern_size = square_size * num_squares

        # Define the user's choice for the center location (x, y)
        user_center_x =  mean_center[0]  # Change this to your desired x-coordinate
        user_center_y =  mean_center[1]  # Change this to your desired y-coordinate
        # Calculate the starting position to center the pattern around the user's choice
        start_x = int(user_center_x - pattern_size // 2)
        start_y = int(user_center_y - pattern_size // 2)
        e4_coordinates = []
        
        for i in range(8):
        	for j in range(7, -1, -1):
        		e4_x = start_x + i * square_size + square_size / 2
        		e4_y = start_y + j * square_size + square_size / 2
        		
        
        # Print the coordinates of the 'e4' square
        print("Coordinates of 'e4' square:")
        print("X:", e4_x)
        print("Y:", e4_y)

	
        # Define the colors for the red lines and blue dots (in BGR format)
        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)

        # Define chessboard labels
        columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        rows = [str(i) for i in range(1, 9)]

        # Draw the red lines to create the chessboard pattern
        for i in range(num_squares + 1):
            # Draw horizontal lines
            y = int(start_y + i * square_size)
            cv2.line(frame, (start_x, y), (int(start_x + pattern_size), y), red_color, 2)

            # Draw vertical lines
            x = int(start_x + i * square_size)
            cv2.line(frame, (x, start_y), (x, int(start_y + pattern_size)), red_color, 2)

        # Draw blue dots in the center of each square and label them
        for i in range(num_squares):
            for j in range(num_squares):
                center_x = int(start_x + (i + 0.5) * square_size)
                center_y = int(start_y + (j + 0.5) * square_size)
                cv2.circle(frame, (center_x, center_y), 5, blue_color, -1)  # -1 for filled circle

                # Calculate and display the chess coordinate labels
                chess_coordinate = columns[i] + rows[7 - j]
                #print(f"{chess_coordinate} = ({center_x},{center_y})")
                # Save the 'e4' square coordinates to a CSV file
                e4_coordinates.append((chess_coordinate,center_x,center_y))
                with open('/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv', mode='w', newline='') as file:
                	writer = csv.writer(file)
                	
                	writer.writerow(['box', 'X_coordinate', 'Y_coordinate'])
                	for chess_coordinate,center_x,center_y in e4_coordinates:
                		writer.writerow([chess_coordinate,center_x,center_y])
                
                cv2.putText(frame, chess_coordinate, (int(center_x) - 10, int(center_y) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the image with the detected corners and mean center
        print("Displaying the image...")
        cv2.imshow('Chessboard Corners and Mean Center', frame)
        # Keep the image window open
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Release the webcam
    cap.release()

else:
    print("Failed to capture a frame from the camera. Make sure the camera is connected and the index is correct.")

