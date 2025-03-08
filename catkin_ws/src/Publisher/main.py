#!/usr/bin/env python

from __future__ import print_function
from six.moves import input
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import Constraints
import geometry_msgs.msg
import shape_msgs.msg
import time 
import csv
import numpy as np
import imutils
import pandas as pd
import math
import cv2
import time
import chess
import chess.engine
from pynput import keyboard

# Path to your Stockfish binary (update this with your Stockfish installation path)
stockfish_path = '/usr/games/stockfish'

# Initialize the Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
previous_moves = []  # Declare the list outside of the function
##Declaration of Constants
height_Tcp = 0.62017
X_fov = 46.2659
Y_fov = 39.37
centre_point_x = 300
centre_point_y = 300
slide_offset = 0 
#calculations 

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

def name_to_coordinates(search_value, mean_center):
    # Extracting Pixels from CSV
    # Specify the path to your CSV file
    csv_file = "/home/kathan/catkin_ws/src/Publisher/Final_coordinates_hard.csv"  # Replace with your CSV file path

    # The value to search for in the first column


    # Open and read the CSV file
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        
        found = False  # Flag to check if the value is found
        
        for row in reader:
            if row and row[0] == search_value:
                found = True
                print(f"Found {search_value} in the first column of the CSV file.")
                print(f"X coordinate =  {row[1]}, Y coordinate = {row[2]}")
                break
        if not found:
            print("The value 'e4' was not found in the CSV file.")

    e4_x = mean_center[0]  # float(row[1])  # input pixels
    e4_y = mean_center[1]  # float(row[2])

    centre_point_x = 300
    centre_point_y = 300

    offset_x = (e4_x - centre_point_x)
    offset_y = (e4_y - centre_point_y)

    angle_x = (46.2659 * offset_x) / 600
    angle_y = (39.37 * offset_y) / 600

    theta_x = math.radians(angle_x)
    theta_y = math.radians(angle_y)

    dist_offset_x = math.tan(theta_x) * height_Tcp
    dist_offset_y = math.tan(theta_y) * height_Tcp

    # Define the point (x, y)
    x = dist_offset_x  + float(row[1]) #+ 0.03   # Replace with your desired coordinates
    y = dist_offset_y + 0.05 + float(row[2])  # Replace with your desired coordinates

    # Define the angle in degrees for clockwise rotation
    angle_degrees = -45

    # Convert the angle to radians
    angle_radians = math.radians(angle_degrees)

    # Create a rotation matrix
    rotation_matrix = np.array([[math.cos(angle_radians), -math.sin(angle_radians)],
                            [math.sin(angle_radians), math.cos(angle_radians)]])


    # Perform the clockwise rotation
    rotated_point = np.dot(rotation_matrix, np.array([x, y]))

    # Extract the new coordinates
    new_x, new_y = rotated_point

    new_x = -new_x + (-0.390284113345598)
    new_y = new_y + 0.5966461492210844
    # Print the rotated coordinates
    print(f"Original Point: ({x}, {y})")
    print(f"Rotated Point: ({new_x}, {new_y}")

    return new_x, new_y
def camera_calibration():
    # Initialize the webcam (usually the default camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 90)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    # for image 1
    height, width, channels = frame.shape
    size = min(height, width)
    y1 = (height - size) // 2
    y2 = y1 + size
    x1 = (width - size) // 2
    x2 = x1 + size
    frame = frame[y1:y2, x1:x2]



    # Wait for the Enter key to be pressed
    print("Press Enter to capture an image...")
    input()  # Wait for Enter key

    # Capture a frame

    #cv2.resize(frame, (800, 600))

    if ret:
        # Display the captured image
        # cv2.imshow("Input Image", frame)

        # Save the captured image to a file
        cv2.imwrite("/home/kathan/catkin_ws/src/Publisher/captured_image.jpg", frame)
        print("Captured image saved as captured_image.jpg")
          
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define the number of rows and columns in the chessboard
    rows = 7  # Number of interior corners in a row - 1
    cols = 7  # Number of interior corners in a column - 1
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
    
    if ret:
        cv2.drawChessboardCorners(frame, (rows, cols), corners, ret)
        cv2.imshow('Chessboard Corners', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Chessboard corners not found.")
    
    # Calculate the mean coordinates of each square and continue processing if corners are found
    if ret is not None:
        centers = []
        for row in range(rows):
            for col in range(cols):
                center_x = int(corners[row * cols + col][0][0])
                center_y = int(corners[row * cols + col][0][1])
                centers.append((center_x, center_y))
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                corner_number = row * cols + col + 1
                cv2.putText(frame, str(corner_number), (center_x + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # Calculate the mean of all detected centers
        mean_x = sum(x for x, _ in centers) / len(centers)
        mean_y = sum(y for _, y in centers) / len(centers)
        mean_center = (int(mean_x), int(mean_y))
    
        # Draw the mean center in green
        cv2.circle(frame, mean_center, 5, (0, 255, 0), -1)
    
        print("Mean Center:", mean_center)
        
    
        # Display the image with detected corners and mean center
    
        # Calculate the distance between corner 1 and corner 2 (for example)
        corner1 = corners[23][0]  # Access the 24th detected corner
        corner2 = corners[24][0]  # Access the 25th detected corner
        distance_pixels = np.linalg.norm(corner1 - corner2)
    
        # Define the size of each square and the number of squares in each row and column
        square_size = distance_pixels
        num_squares = 8
    
        # Calculate the size of the chessboard pattern
        pattern_size = square_size * num_squares
    
        # Define the user's choice for the center location (x, y)
        user_center_x = mean_center[0]  # Change this to your desired x-coordinate
        user_center_y = mean_center[1]  # Change this to your desired y-coordinate
    
        # Calculate the starting position to center the pattern around the user's choice
        start_x = int(user_center_x - pattern_size // 2)
        start_y = int(user_center_y - pattern_size // 2)
    
        # Define the colors for the red lines and blue dots (in BGR format)
        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)
    
        # Define chessboard labels
        columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        rows = [str(i) for i in range(8, 0, -1)]
    
        # Draw the red lines to create the chessboard pattern
        for i in range(num_squares + 1):
            # Draw horizontal lines
            y = int(start_y + i * square_size)
            cv2.line(frame, (start_x, y), (int(start_x + pattern_size), y), red_color, 2)
    
            # Draw vertical lines
            x = int(start_x + i * square_size)
            cv2.line(frame, (x, start_y), (x, int(start_y + pattern_size)), red_color, 2)
    
    
    
    # Create an empty NumPy array to store x and y coordinates along with EorF
    coordinate_array = np.empty((0, 3), int)
    
    # Create an empty list to store the chessboard labels
    chess_labels = []
    
    # Draw blue dots in the center of each square and label them
    for j in range(num_squares):
        for i in range(num_squares):
            center_x = int(start_x + (i + 0.5) * square_size)
            center_y = int(start_y + (j + 0.5) * square_size)
            cv2.circle(frame, (center_x, center_y), 5, blue_color, -1)  # -1 for a filled circle
    
            # Calculate and display the chess coordinate labels
            chess_coordinate = columns[i] + rows[j]
            cv2.putText(frame, chess_coordinate, (int(center_x) - 10, int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
            # Set EorF as 1 for specific rows, and 0 for others
            if (chess_coordinate[0] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) and (chess_coordinate[1] in ['1', '2', '7', '8']):
                eorf = 1
            else:
                eorf = 0
    
            # Append x, y, and EorF to the NumPy array
            coordinate_array = np.vstack((coordinate_array, [center_x, center_y, eorf]))
           
            # Append the chessboard label to the list
            chess_labels.append(chess_coordinate)
    
     # Open a CSV file for writing
    with open("/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Square', 'x', 'y', 'EorF'])  # Write the header row

        for j in range(num_squares):
            for i in range(num_squares):
                center_x = int(start_x + (i + 0.5) * square_size)
                center_y = int(start_y + (j + 0.5) * square_size)
                cv2.circle(frame, (center_x, center_y), 5, blue_color, -1)  # -1 for a filled circle

                # Calculate and display the chess coordinate labels
                chess_coordinate = columns[i] + rows[j]
                cv2.putText(frame, chess_coordinate, (int(center_x) - 10, int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Set EorF as 4 for rows a1-h1 and a2-h2, and as 2 for rows a7-h7 and a8-h8
                if (chess_coordinate[0] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) and (chess_coordinate[1] in ['1', '2']):
                    eorf = 4
                elif (chess_coordinate[0] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) and (chess_coordinate[1] in ['7', '8']):
                    eorf = 2
                else:
                    eorf = 0

                # Append x, y, and EorF to the NumPy array
                coordinate_array = np.vstack((coordinate_array, [center_x, center_y, eorf]))

                # Append the chessboard label and EorF to the CSV file
                csvwriter.writerow([chess_coordinate, center_x, center_y, eorf])



    print("CSV file 'chessboard_coordinates.csv' created.")
    return mean_center
    
# Function to change 'EorF' value for a square by square name
def game_restart():

    # Load the image from file
    frame = cv2.imread("/home/kathan/catkin_ws/src/Publisher/captured_image.jpg")

    # Check if the image is loaded successfully
    if frame is None:
        print("Error: Unable to load the image from 'captured_image.jpg'")
        return None



   
    # Capture a frame

    #cv2.resize(frame, (800, 600))


          
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define the number of rows and columns in the chessboard
    rows = 7  # Number of interior corners in a row - 1
    cols = 7  # Number of interior corners in a column - 1
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
    
    if ret:
        cv2.drawChessboardCorners(frame, (rows, cols), corners, ret)
        cv2.imshow('Chessboard Corners', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Chessboard corners not found.")
    
    # Calculate the mean coordinates of each square and continue processing if corners are found
    if ret is not None:
        centers = []
        for row in range(rows):
            for col in range(cols):
                center_x = int(corners[row * cols + col][0][0])
                center_y = int(corners[row * cols + col][0][1])
                centers.append((center_x, center_y))
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                corner_number = row * cols + col + 1
                cv2.putText(frame, str(corner_number), (center_x + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # Calculate the mean of all detected centers
        mean_x = sum(x for x, _ in centers) / len(centers)
        mean_y = sum(y for _, y in centers) / len(centers)
        mean_center = (int(mean_x), int(mean_y))
    
        # Draw the mean center in green
        cv2.circle(frame, mean_center, 5, (0, 255, 0), -1)
    
        print("Mean Center:", mean_center)
        
    
        # Display the image with detected corners and mean center
    
        # Calculate the distance between corner 1 and corner 2 (for example)
        corner1 = corners[23][0]  # Access the 24th detected corner
        corner2 = corners[24][0]  # Access the 25th detected corner
        distance_pixels = np.linalg.norm(corner1 - corner2)
    
        # Define the size of each square and the number of squares in each row and column
        square_size = distance_pixels
        num_squares = 8
    
        # Calculate the size of the chessboard pattern
        pattern_size = square_size * num_squares
    
        # Define the user's choice for the center location (x, y)
        user_center_x = mean_center[0]  # Change this to your desired x-coordinate
        user_center_y = mean_center[1]  # Change this to your desired y-coordinate
    
        # Calculate the starting position to center the pattern around the user's choice
        start_x = int(user_center_x - pattern_size // 2)
        start_y = int(user_center_y - pattern_size // 2)
    
        # Define the colors for the red lines and blue dots (in BGR format)
        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)
    
        # Define chessboard labels
        columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        rows = [str(i) for i in range(8, 0, -1)]
    
        # Draw the red lines to create the chessboard pattern
        for i in range(num_squares + 1):
            # Draw horizontal lines
            y = int(start_y + i * square_size)
            cv2.line(frame, (start_x, y), (int(start_x + pattern_size), y), red_color, 2)
    
            # Draw vertical lines
            x = int(start_x + i * square_size)
            cv2.line(frame, (x, start_y), (x, int(start_y + pattern_size)), red_color, 2)
    
    
    
    # Create an empty NumPy array to store x and y coordinates along with EorF
    coordinate_array = np.empty((0, 3), int)
    
    # Create an empty list to store the chessboard labels
    chess_labels = []
    
    # Draw blue dots in the center of each square and label them
    for j in range(num_squares):
        for i in range(num_squares):
            center_x = int(start_x + (i + 0.5) * square_size)
            center_y = int(start_y + (j + 0.5) * square_size)
            cv2.circle(frame, (center_x, center_y), 5, blue_color, -1)  # -1 for a filled circle
    
            # Calculate and display the chess coordinate labels
            chess_coordinate = columns[i] + rows[j]
            cv2.putText(frame, chess_coordinate, (int(center_x) - 10, int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
            # Set EorF as 1 for specific rows, and 0 for others
            if (chess_coordinate[0] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) and (chess_coordinate[1] in ['1', '2', '7', '8']):
                eorf = 1
            else:
                eorf = 0
    
            # Append x, y, and EorF to the NumPy array
            coordinate_array = np.vstack((coordinate_array, [center_x, center_y, eorf]))
           
            # Append the chessboard label to the list
            chess_labels.append(chess_coordinate)
    
     # Open a CSV file for writing
    with open("/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Square', 'x', 'y', 'EorF'])  # Write the header row

        for j in range(num_squares):
            for i in range(num_squares):
                center_x = int(start_x + (i + 0.5) * square_size)
                center_y = int(start_y + (j + 0.5) * square_size)
                cv2.circle(frame, (center_x, center_y), 5, blue_color, -1)  # -1 for a filled circle

                # Calculate and display the chess coordinate labels
                chess_coordinate = columns[i] + rows[j]
                cv2.putText(frame, chess_coordinate, (int(center_x) - 10, int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Set EorF as 4 for rows a1-h1 and a2-h2, and as 2 for rows a7-h7 and a8-h8
                if (chess_coordinate[0] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) and (chess_coordinate[1] in ['1', '2']):
                    eorf = 4
                elif (chess_coordinate[0] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) and (chess_coordinate[1] in ['7', '8']):
                    eorf = 2
                else:
                    eorf = 0

                # Append x, y, and EorF to the NumPy array
                coordinate_array = np.vstack((coordinate_array, [center_x, center_y, eorf]))

                # Append the chessboard label and EorF to the CSV file
                csvwriter.writerow([chess_coordinate, center_x, center_y, eorf])



    print("CSV file 'chessboard_coordinates.csv' created.")
    return mean_center
    
def change_eorf_value(square_name):
    # Load the chessboard coordinates from the CSV file
    chessboard_coordinates = pd.read_csv("/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv")
    
    # Find the index of the specified square_name
    index = chessboard_coordinates.index[chessboard_coordinates['Square'] == square_name].tolist()

    if not index:
        print(f"Square '{square_name}' not found in the chessboard coordinates.")
        return

    # Get the current EorF value
    current_eorf = chessboard_coordinates.loc[index, 'EorF'].values[0]

    # Toggle the EorF value (1 to 0 or 0 to 1)
    new_eorf = 1 if current_eorf == 0 else 0

    # Update the DataFrame with the new EorF value
    chessboard_coordinates.loc[index, 'EorF'] = new_eorf

    # Save the changes back to the CSV file
    chessboard_coordinates.to_csv("/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv", index=False)

    print("changes kari nakhiyaa")
def move_detection(img1, img2):
 
    # Load the two images
    img1 = cv2.resize(img1, (800, 600))  # Resize both images to have the same dimensions
    img2 = cv2.resize(img2, (800, 600))
    # for image 1
    height, width, channels = img1.shape
    size = min(height, width)
    y1 = (height - size) // 2
    y2 = y1 + size
    x1 = (width - size) // 2
    x2 = x1 + size
    img1 = img1[y1:y2, x1:x2]
    #for image 2
    height, width, channels = img2.shape
    size = min(height, width)
    y1 = (height - size) // 2
    y2 = y1 + size
    x1 = (width - size) // 2
    x2 = x1 + size
    img2 = img2[y1:y2, x1:x2]
    coordinates_csv_path = "/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv"
    #Histogram distortion correction 
    # Grayscale
    #gray1 = cv2.equalizeHist(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    #gray2 = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    

    # Find the difference between the two images using absdiff
    # Calculate the Mean Squared Error (MSE)
    mse = ((gray1 - gray2) ** 2).mean()

    diff = cv2.absdiff(gray1, gray2)
    cv2.imshow("Difference (img1, img2)", diff)

    # Apply threshold
    matrix, thresh = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh1", thresh)

    # Apply morphological opening operation to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    finalimage = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Black and white Final", finalimage)

    # Load the chessboard coordinates from the CSV file
    chessboard_coordinates = pd.read_csv(coordinates_csv_path)

    # Loop through the coordinates and draw red points on the final image
    for _, row in chessboard_coordinates.iterrows():
        x = int(row['x'])  # Assuming 'x' is the column name for the x-coordinate in your CSV
        y = int(row['y'])  # Assuming 'y' is the column name for the y-coordinate in your CSV
        cv2.circle(finalimage, (x, y), 3, (255, 0, 0), -1)  # Draw a red circle at the coordinates

    # Show or save the final image with the drawn points
    cv2.imshow("Final Image with Points", finalimage)
    
    # Show final images with differences
    x = np.zeros((600, 10, 3), np.uint8)  # Adjust dimensions to match the resized image
    result = np.hstack((img1, x, img2))
    cv2.imshow("Differences", result)

    # Define a function to calculate the white region (foreground) around a point
    def calculate_white_region(image, x, y, radius=6):
        roi = image[y - radius:y + radius, x - radius:x + radius]
        white_region = np.count_nonzero(roi == 255)
        return white_region

    # Initialize variables to keep track of the points with the largest white regions
    largest_region_points = [None, None]
    largest_regions = [0, 0]
    
    # Load the chessboard coordinates from the CSV file
    chessboard_coordinates = pd.read_csv("/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv")
    coordinate_array = chessboard_coordinates[['x', 'y']].to_numpy()
    chess_labels = chessboard_coordinates['Square'].to_numpy()  # Use 'Square' as the column name
    eorf_values = chessboard_coordinates['EorF'].to_numpy()  # Use 'EorF' as the column name
    # Loop through the coordinates to find the two points with the largest white regions
    
    for _, row in chessboard_coordinates.iterrows():
        x = int(row['x'])  # Assuming 'x' is the column name for the x-coordinate in your CSV
        y = int(row['y'])  # Assuming 'y' is the column name for the y-coordinate in your CSV

        white_region = calculate_white_region(finalimage, x, y)

        # Check if this point has a larger white region than the current largest regions
        if white_region > largest_regions[0]:
            largest_regions[1] = largest_regions[0]
            largest_regions[0] = white_region
            largest_region_points[1] = largest_region_points[0]
            largest_region_points[0] = (x, y)
        elif white_region > largest_regions[1]:
            largest_regions[1] = white_region
            largest_region_points[1] = (x, y)

    # The two points with the largest white regions are stored in largest_region_points
    point1 = largest_region_points[0]
    point2 = largest_region_points[1]
    
    # Find the corresponding square names from the CSV file for these two points
    square_name1 = chessboard_coordinates[
        (chessboard_coordinates['x'] == point1[0]) & (chessboard_coordinates['y'] == point1[1])
    ]['Square'].values[0]

    square_name2 = chessboard_coordinates[
        (chessboard_coordinates['x'] == point2[0]) & (chessboard_coordinates['y'] == point2[1])
    ]['Square'].values[0]

    # Display or use the square names as needed
    print("Point 1 Square Name:", square_name1)
    print("Point 2 Square Name:", square_name2)
    output_string = "error_move"
    # Check conditions and swap values in CSV file accordingly
    eorf_square_name1 = chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'].values[0]
    eorf_square_name2 = chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'].values[0]

    if eorf_square_name1 == 0:  
        # Swap values in CSV file
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'] = eorf_square_name2
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'] = 0

        # Swap square names
        square_name1, square_name2 = square_name2, square_name1
        output_string = square_name1+square_name2
        print(f"Swapped values for {square_name1}{square_name2}")

    elif eorf_square_name2 == 0:
        # Swap values in CSV file
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'] = 0
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'] = eorf_square_name1
        output_string = square_name1+square_name2
        print(f"Swapped values for {square_name1}{square_name2}")

    elif eorf_square_name1 == 4 and eorf_square_name2 == 2:
        # Swap values in CSV file
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'] = 0
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'] = 4
        output_string = square_name1+square_name2
        print(f"Swapped values for {square_name1}{square_name2}")
   
    elif eorf_square_name1 == 2 and eorf_square_name2 == 4:
        # Swap values in CSV file
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'] = 4
        chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'] = 0
        output_string = square_name2+square_name1
        print(f"Swapped values for {square_name2}{square_name1}")    

    # Save the modified CSV file
    chessboard_coordinates.to_csv(coordinates_csv_path, index=False)
    





    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the output string
    return output_string

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

    
class MoveGroupPythonInterfaceTutorial(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        
        
        
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = -0.74735
        joint_goal[1] = -1.58633
        joint_goal[2] = -1.722465
        joint_goal[3] = -1.415462
        joint_goal[4] = 1.565735
        joint_goal[5] = 0.042412 # 1/6 of a turn
        
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self,T_x,T_y,T_z):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x =-0.9242373753356713
        pose_goal.orientation.y = -0.3816768858930473
        pose_goal.orientation.z = 0.0014685228283635545
        pose_goal.orientation.w = 0.010289424100389678
        pose_goal.position.x = T_x # -0.26546338816271076
        pose_goal.position.y = T_y # 0.5942884108626451
        pose_goal.position.z = T_z # 0.03640
        
        
        
       

        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)
        
    def go_to_pose_goal_exp(self, T_x, T_y, T_z):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        orientation_constraints=True

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:

        # Create a pose goal
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = -0.9242373753356713
        pose_goal.orientation.y = -0.3816768858930473
        pose_goal.orientation.z = 0.0014685228283635545
        pose_goal.orientation.w = 0.010289424100389678
        pose_goal.position.x = T_x
        pose_goal.position.y = T_y
        pose_goal.position.z = T_z

        # Set pose target
        move_group.set_pose_target(pose_goal)

        if orientation_constraints:
            orientation_constraint = moveit_msgs.msg.OrientationConstraint()
            # Set frame_id and link_name before printing
            orientation_constraint.header.frame_id = "base_link"
            orientation_constraint.link_name = move_group.get_end_effector_link()
            # Print frame_id and link_name
            print("this is the constraint frame_id =", orientation_constraint.header.frame_id)
            print("this is the constraint link_name =", orientation_constraint.link_name)
            print("this is the constraint link_name =", orientation_constraint.link_name)



            # Set fixed orientation values directly
            orientation_constraint.orientation.x = -0.9242373753356713
            orientation_constraint.orientation.y = -0.3816768858930473
            orientation_constraint.orientation.z = 0.0014685228283635545
            orientation_constraint.orientation.w = 0.010289424100389678

            orientation_constraint.absolute_x_axis_tolerance = 0.2
            orientation_constraint.absolute_y_axis_tolerance = 0.2
            orientation_constraint.absolute_z_axis_tolerance = 0.2
            orientation_constraint.weight = 1.0
            
            orientation_constraints = Constraints()
            orientation_constraints.orientation_constraints.append(orientation_constraint)
            '''
            #joint constraints 
            joint_constraints = moveit_msgs.msg.JointConstraint()
            joint_constraints.joint_name = "shoulder_lift_joint"  # Replace with the actual joint name
            joint_constraints.min_position = -2.478368  # Replace with the minimum allowed joint position
            joint_constraints.max_position = -1.53589  # Replace with the maximum allowed joint position
            joint_constraints.weight = 1.0  # Adjust the weight as needed
            '''


            
            
            move_group.set_path_constraints(orientation_constraints)
  

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)
        

        

def csv_reader(search_value):
# Specify the path to your CSV file
    csv_file = "/home/kathan/catkin_ws/src/Publisher/Final_coordinates_hard.csv"  # Replace with your CSV file path

    # The value to search for in the first column
   

    # Open and read the CSV file
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)

        found = False  # Flag to check if the value is found

        for row in reader:
            if row and row[0] == search_value:
                found = True
                print(f"Found {search_value} in the first column of the CSV file.")
                print(f"X coordinate =  {row[1]}, Y coordinate = {row[2]}")
                break

        if not found:
            print("The value 'e4' was not found in the CSV file.")
    return row
def home_position():
    tutorial = MoveGroupPythonInterfaceTutorial()
    tutorial.go_to_joint_state()
def pick_and_drop(best_move,mean_center):
    tutorial = MoveGroupPythonInterfaceTutorial()
    best_move_str = str(best_move)
    print("ahiya pohchi gayoo")
    pick_location = best_move_str[:2]
    drop_location = best_move_str[2:4]
    # picking action
    #tutorial.make_workspace()
    new_x , new_y = name_to_coordinates(pick_location, mean_center)
    
    tutorial.go_to_pose_goal(new_x, new_y,0.12640)
    tutorial.go_to_pose_goal(new_x, new_y,0.0620)
    time.sleep(0.25)
    tutorial.go_to_pose_goal(new_x, new_y,0.15640)
    
    #droping action
    new_x , new_y = name_to_coordinates(drop_location, mean_center)
    tutorial.go_to_pose_goal(new_x, new_y,0.15640)
    tutorial.go_to_pose_goal(new_x, new_y,0.061)
    time.sleep(0.25)
    tutorial.go_to_pose_goal(new_x, new_y,0.15640)
    tutorial.go_to_joint_state()
    
def capturing(best_move,mean_center):
    best_move_str = str(best_move)
    
    
    drop_location = best_move_str[2:4]
    # picking action
    new_x , new_y = name_to_coordinates(drop_location, mean_center)
    #tutorial.make_workspace()
    tutorial.go_to_pose_goal(new_x, new_y,0.15640)
    tutorial.go_to_pose_goal(new_x, new_y,0.0620)
    time.sleep(0.5)
    tutorial.go_to_pose_goal(new_x, new_y,0.16640)
    
    #droping action
    new_x , new_y = name_to_coordinates(drop_location, mean_center)
    global slide_offset
    slide_offset = slide_offset + 0.03
    tutorial.go_to_pose_goal(-0.47304922963651314-slide_offset,  0.22612090722999234 +slide_offset ,0.16640)
    tutorial.go_to_pose_goal(-0.47304922963651314-slide_offset,  0.22612090722999234+slide_offset ,0.0620)
    tutorial.go_to_pose_goal(-0.47304922963651314-slide_offset,  0.22612090722999234+slide_offset ,0.44423314184507895)
    
    #tutorial.go_to_joint_state()
    #tutorial.go_to_pose_goal(-0.5055464363723627, 0.501745225892255 ,0.37323686254311)
    
    #tutorial.go_to_pose_goal(new_x, new_y,0.03640)
    #tutorial.go_to_pose_goal(new_x, new_y,0.12640)

def t23(best_move,mean_center):
    drop_location = str(best_move)
    
    

    # picking action
    new_x , new_y = name_to_coordinates(drop_location, mean_center)
    #tutorial.make_workspace()
    tutorial.go_to_pose_goal(new_x, new_y,0.15640)
    tutorial.go_to_pose_goal(new_x, new_y,0.0620)
    time.sleep(0.5)
    tutorial.go_to_pose_goal(new_x, new_y,0.16640)
    


    tutorial.go_to_pose_goal(-0.6561850313182633,  0.3460530492507995 ,0.44423314184507895)
    print("                                                                                ")
    print("                                                                                ")
    print("                                                                                ")
    print("                  # You cannot Defeat the Maker's of the Gameee! ")
    print("                                                                                ")
    print("                                                                                ")
    print("                                                                                ")
    
    #tutorial.go_to_joint_state()
    #tutorial.go_to_pose_goal(-0.5055464363723627, 0.501745225892255 ,0.37323686254311)
    
    #tutorial.go_to_pose_goal(new_x, new_y,0.03640)
    #tutorial.go_to_pose_goal(new_x, new_y,0.12640)
    

def on_key_release(key):
    if key == keyboard.Key.esc:
        return False  # Stop listener
    if key == keyboard.KeyCode.from_char('r'):
        return True  # Set the game_over flag to True 
def capture_frame():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Camera {camera_index} cannot be opened.")
        sys.exit(1)

    # Set camera parameters for consistent image quality
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Turn off auto exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, 0.5)  # Set exposure to a specific value (adjust as needed)
    cap.set(cv2.CAP_PROP_GAIN, 0.5)  # Set gain to a specific value (adjust as needed)

    # Capture a frame

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture a frame from the camera.")
        sys.exit(1)

    # Release the camera
    cap.release()

    return frame

def playing_best_move(center_coordinates, user_move_uci=None):
    # Path to your Stockfish binary (update this with your Stockfish installation path)
    stockfish_path = '/usr/games/stockfish'
    #cap = cv2.VideoCapture(0)  # 0 represents the default camera, change it if you have multiple cameras
    # Create a variable to store the first image
    image_1 = None
    image_2 = None
    # Flag to track which image is being captured
    #capturing_image1 = True
    print("Now set chess pieces on the board and then press enter!")
    #ret, frame = cap.read()
    #time.sleep(1)
    image_1 = capture_frame()
    cv2.imshow("Image_1 Game Starts", image_1)
    #time.sleep(0.5)
    #key = cv2.waitKey(1000)
    #cap.release()
    # input("========= Camera calibration successful. Press `Enter` after playing your move ...")

    # Initialize the Stockfish engine
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = chess.Board()
        # Set the desired Elo rating to adjust difficulty
        desired_elo = 1350 # 2850 You can set your desired rating here
        # Set the Elo rating option for Stockfish
        engine.configure({"UCI_Elo": desired_elo})
        game_over = False  # Flag to track game over

        # Set up the keyboard listener
        listener = keyboard.Listener(on_release=on_key_release)
        listener.start()

        while not board.is_game_over() and not game_over:
            print(board)
            
            calibration_choice = input("Enter your move manually (e.g., 'e2e4') by pressing M or continue with automation detection by pressing ENTER")

            if calibration_choice.lower() == 'm':
                user_move_uci = input("Enter your move here: ")
            elif calibration_choice.lower() == '23':  # Assuming you want to check if 'calibration_choice' is equal to '23'
                current_fen = board.fen()
                parts = current_fen.split(' ')
                position = parts[0]
                black_king_square = position.find('k')
                file_index = black_king_square % 8
                rank_index = 7 - black_king_square // 8
                file_name = chr(ord('a') + file_index)
                rank_number = rank_index + 1
                black_king_location = f"{file_name}{rank_number}"
                print("kings position = ",black_king_location )
                t23(black_king_location,center_coordinates)
                # Additional logic for handling '23'
            elif calibration_choice.lower() == 'h':  # Assuming you want to check if 'calibration_choice' is equal to '23'
                home_position()
                continue 	
                
            else:
                print("OK, your wish :)")
                image_2 = capture_frame()
                user_move_uci = move_detection(image_1, image_2)
                cv2.destroyAllWindows()
            
            '''
            if not ret:
                print("Error: Failed to capture a frame from the camera.")
                # You may want to add further error-handling logic here if needed.
            else:
            	#time.sleep(0.5)
            	#cap = cv2.VideoCapture(0)

                #time.sleep(2)
            	#time.sleep(3)
            	#ret, frame2 = cap.read()
            	#time.sleep(0.5)
            	
            	#time.sleep(0.5)
            	#cap.release()
		#key = cv2.waitKey(100)

            '''
            

            print(f"this is the move you just played {user_move_uci}")
            ## ### ##
            coordinates_csv_path = "/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv"
            chessboard_coordinates = pd.read_csv("/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv")
            # Validate and make the user's move
            try:
                user_move = chess.Move.from_uci(user_move_uci)
                if user_move in board.legal_moves:
                
                    board.push(user_move)
                    change_karva = user_move_uci
                    square_name1 = change_karva[:2]
                    square_name2 = change_karva[2:4]
                    chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'] = 0
                    chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'] = 4
                else:
                    print("Invalid move. Try again.")
                    image_1 = capture_frame()
                    #cap = cv2.VideoCapture(0)
                    

                    #ret, frame1 = cap.read()
                    #time.sleep(0.5)

                    #cv2.imshow("Image_1 ", image_1)
                    #time.sleep(0.5)
                    #key = cv2.waitKey(100)
                    #cap.release()
                    continue
            except ValueError:
                print("Invalid move format. Try again.")
                continue

            print(f"You played: {user_move_uci}")

            # Get the best move from Stockfish in response to the user's move
            result = engine.play(board, chess.engine.Limit(time=0.1))
            print(f"Stockfish suggests: {result.move}")

            # Make Stockfish's move
            initial_square = result.move.from_square
            target_square = result.move.to_square
            print(f"this are the initial square:: {initial_square}")
            change_karva = result.move.uci()

            ## ### ##

            if board.piece_at(target_square) is None and board.piece_at(initial_square) is not None:
                print(f"Best move is not a capture: {result.move.uci()}")
                pick_and_drop(result.move, center_coordinates)
                square_name1 = change_karva[:2]
                square_name2 = change_karva[2:4]
                # Check conditions and swap values in CSV file accordingly
                eorf_square_name1 = chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'].values[0]
                eorf_square_name2 = chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'].values[0]
                chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'] = 0
                chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'] = 2
                # Save the modified CSV file
                chessboard_coordinates.to_csv(coordinates_csv_path, index=False)
				
		        
            else:
                print(f"Best move is a capture: {result.move.uci()}")
                capturing(result.move, center_coordinates)
                pick_and_drop(result.move, center_coordinates)
                
                # Check conditions and swap values in CSV file accordingly
                eorf_square_name1 = chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'].values[0]
                eorf_square_name2 = chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'].values[0]
                
                chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name1, 'EorF'] = 0
                chessboard_coordinates.loc[chessboard_coordinates['Square'] == square_name2, 'EorF'] = 2
                # Save the modified CSV file
                chessboard_coordinates.to_csv(coordinates_csv_path, index=False)
				
                
            image_1 = capture_frame()
            print("saambaar cafeee!!!")
            board.push(result.move)


            

        # Stop the keyboard listener
        listener.stop()
        listener.join()

        print("Game over")
        print(board.result())

    '''	
            # Get the user's move
            #user_move_uci = input("Enter your move (in UCI notation, e.g., 'e2e4'): ")
            
            # Validate and make the user's move
            try:
                user_move = chess.Move.from_uci(user_move_uci)
                if user_move in board.legal_moves:
                    board.push(user_move)
                else:
                    print("Invalid move. Try again.")
                    continue
            except ValueError:
                print("Invalid move format. Try again.")
                continue
            
            print(f"You played: {user_move_uci}")
            
            # Get the best move from Stockfish in response to the user's move
            result = engine.play(board, chess.engine.Limit(time=0.1))
            
            print(f"Stockfish suggests: {result.move}")
            
            # Make Stockfish's move
            board.push(result.move)

    print("Game over")
    print(board.result())
    return result.move
    '''

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


if __name__ == "__main__":
    try:
        print("")
        print("----------------------------------------------------------")
        print("Welcome to the exciting world of robotics, right before you stands the UR10 robot, a master chess player. Are you ready to challenge the best and play a game of chess with it?")
        input("============ Press `Enter` to start with home position ...")
        tutorial = MoveGroupPythonInterfaceTutorial()
        print("Going to home position --------------")
        tutorial.go_to_joint_state()
        print("----------------------------------------------------------")
        # Ask the user if they want to run camera calibration
        calibration_choice = input("Do you want to run camera calibration? (y/n): ")
        if calibration_choice.lower() == 'y':
            print("============ Press `Enter` to begin camera calibration ...")
            center_coordinates = camera_calibration()
        elif calibration_choice.lower() == 'n':
            # Handle the case when the user chooses to skip camera calibration
            print("OKK your Wish:).")
            center_coordinates = game_restart()
            
        else:
            print("OKK your Wish:).")

	# Continue with the rest of your code using center_coordinates as needed

        '''
        cap = cv2.VideoCapture(0)  # 0 represents the default camera, change it if you have multiple cameras
        # Create a variable to store the first image
        image_1 = None
        # Flag to track which image is being captured
        capturing_image1 = True
        print("Now set chess pieces on the board and then press enter!")
        ret, frame = cap.read()
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1)
        image_1 = frame.copy()

        '''
        input("============ Press `Enter` to begin the game ...")
        while True:
            #user_input = input("========= Camera calibration successful. Press `Enter` after playing your move ...")
            #image_2 = frame.copy()
            #move_detection(image_1, image_2, 'chessboard_coordinates.csv')
            #center_coordinates = [302, 312]
            
            playing_best_move(center_coordinates)
            #next_move = best_move(user_input)
            #print(f"next best move is {next_move}")
            #pick_and_drop(next_move, center_coordinates)
            #exit()
    except KeyboardInterrupt:
        print("Exiting the program.")



