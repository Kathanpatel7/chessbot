
#!/usr/bin/env python
import csv
# Define chessboard labels
columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
rows = [str(i) for i in range(1, 9)]
e4_coordinates = []
square_size = 0.049
for i in range(8):
	for j in range(7, -1, -1):
		center_x = float(-0.1715 + (i ) * square_size)
		center_y = float(-0.1715 + (j ) * square_size)
		#cv2.circle(frame, (center_x, center_y), 5, blue_color, -1)  # -1 for filled circle
		# Calculate and display the chess coordinate labels
		chess_coordinate = columns[i] + rows[7 - j]
		#print(f"{chess_coordinate} = ({center_x},{center_y})")
		# Save the 'e4' square coordinates to a CSV file
		e4_coordinates.append((chess_coordinate,center_x,center_y))
		print(e4_coordinates)
		with open('/home/kathan/catkin_ws/src/Publisher/Final_coordinates_hard.csv', mode='w', newline='') as file:
                	writer = csv.writer(file)
                	
                	writer.writerow(['box', 'X_coordinate', 'Y_coordinate'])
                	for chess_coordinate,center_x,center_y in e4_coordinates:
                		writer.writerow([chess_coordinate,center_x,center_y])

'''

import cv2
import numpy as np
import csv

# Load the image
image = cv2.imread(r"/home/kathan/msoom/empty.jpg")  # Replace with the path to your image

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the number of rows and columns in the chessboard
rows = 7  # Number of interior corners in a row - 1
cols = 7  # Number of interior corners in a column - 1

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

if ret:
    cv2.drawChessboardCorners(image, (rows, cols), corners, ret)
    cv2.imshow('Chessboard Corners', image)
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
            cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)
            corner_number = row * cols + col + 1
            cv2.putText(image, str(corner_number), (center_x + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate the mean of all detected centers
    mean_x = sum(x for x, _ in centers) / len(centers)
    mean_y = sum(y for _, y in centers) / len(centers)
    mean_center = (int(mean_x), int(mean_y))

    # Draw the mean center in green
    cv2.circle(image, mean_center, 5, (0, 255, 0), -1)

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
        cv2.line(image, (start_x, y), (int(start_x + pattern_size), y), red_color, 2)

        # Draw vertical lines
        x = int(start_x + i * square_size)
        cv2.line(image, (x, start_y), (x, int(start_y + pattern_size)), red_color, 2)



# Create an empty NumPy array to store x and y coordinates along with EorF
coordinate_array = np.empty((0, 3), int)

# Create an empty list to store the chessboard labels
chess_labels = []

# Draw blue dots in the center of each square and label them
for j in range(num_squares):
    for i in range(num_squares):
        center_x = int(start_x + (i + 0.5) * square_size)
        center_y = int(start_y + (j + 0.5) * square_size)
        cv2.circle(image, (center_x, center_y), 5, blue_color, -1)  # -1 for a filled circle

        # Calculate and display the chess coordinate labels
        chess_coordinate = columns[i] + rows[j]
        cv2.putText(image, chess_coordinate, (int(center_x) - 10, int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
with open('chessboard_coordinates.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Square", "x", "y", "EorF"])  # Write header row

    for label, coordinates in zip(chess_labels, coordinate_array):
        csvwriter.writerow([label, coordinates[0], coordinates[1], coordinates[2]])

# Print the array with x, y, and EorF coordinates along with their chessboard labels
for label, coordinates in zip(chess_labels, coordinate_array):
    print(f"{label}: x={coordinates[0]}, y={coordinates[1]}, EorF={coordinates[2]}")

print("CSV file 'chessboard_coordinates.csv' created.")
'''



'''







        #center_coordinates = camera_calibration()
        # Initialize the camera capture
        # Commented out block of code using triple single quotes
        # cap = cv2.VideoCapture(0)  # 0 represents the default camera, change it if you have multiple cameras
        # Create a variable to store the first image
        # image_1 = None
        # Flag to track which image is being captured
        # capturing_image1 = True
        # print("Now set chess pieces on the board and then press enter!")
        # ret, frame = cap.read()
        # cv2.imshow("Camera Feed", frame)
        # key = cv2.waitKey(1)
        # image_1 = frame.copy()
        
        
        
                    #next_move = best_move(user_input)
            #print(f"next best move is {next_move}")
            #pick_and_drop(next_move, center_coordinates)
            #exit()

        
'''
'''
import cv2
import numpy as np
import pandas as pd

def move_detection(image_1, image_2, coordinates_csv_path):
    # Load two images
    image1 = image_1
    image2 = image_2

    image_diff = cv2.absdiff(image1, image2)
    image_diff_gray = cv2.cvtColor(image_diff, cv2.COLOR_BGR2GRAY)

    matrix, thresold = cv2.threshold(image_diff_gray, 22, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(thresold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) >= 2:
        required_contoures_mid_point = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                mid_x = x + w // 2
                mid_y = y + h // 2
                required_contoures_mid_point.append((mid_x, mid_y))

        # Store mid points in separate variables
        mid_point1 = required_contoures_mid_point[0]
        mid_point2 = required_contoures_mid_point[1]

    # Load the chessboard coordinates from the CSV file
    chessboard_coordinates = pd.read_csv(coordinates_csv_path)
    coordinate_array = chessboard_coordinates[['x', 'y']].to_numpy()
    chess_labels = chessboard_coordinates['Square'].to_numpy()
    eorf_values = chessboard_coordinates['EorF'].to_numpy()

    # Calculate the nearest square to Mid-Point 1
    def find_nearest_square(mid_point, coordinates, labels):
        min_distance = float('inf')
        nearest_square = None
        for square, (square_x, square_y) in zip(labels, coordinates):
            distance = np.sqrt((mid_point[0] - square_x) ** 2 + (mid_point[1] - square_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_square = square
        return nearest_square

    # Call the function to find the nearest square for Mid-Point 1
    nearest_square_mid_point1 = find_nearest_square(mid_point1, coordinate_array, chess_labels)

    # Call the function to find the nearest square for Mid-Point 2
    nearest_square_mid_point2 = find_nearest_square(mid_point2, coordinate_array, chess_labels)

    # Get the 'EorF' values of the two nearest squares
    eorf_value_mid_point1 = eorf_values[chess_labels == nearest_square_mid_point1][0]
    eorf_value_mid_point2 = eorf_values[chess_labels == nearest_square_mid_point2][0]

    # Create a list of (square, EorF) tuples for the two mid-points
    mid_point1_info = (nearest_square_mid_point1, eorf_value_mid_point1)
    mid_point2_info = (nearest_square_mid_point2, eorf_value_mid_point2)

    # Sort the mid-points based on their EorF values, with '1' first
    sorted_mid_points = sorted([mid_point1_info, mid_point2_info], key=lambda x: x[1], reverse=True)

    # Create a string by concatenating square names with '1' EorF value first, and then '0'
    output_string = "".join(square for square, _ in sorted_mid_points)

    # Load the chessboard coordinates from the CSV file
    csv_file_path = "/home/kathan/catkin_ws/src/Publisher/chessboard_coordinates.csv"

    # Function to change 'EorF' value for a square by square name
    def change_eorf_value(square_name):
        index = chessboard_coordinates[chessboard_coordinates['Square'] == square_name].index
        current_eorf = chessboard_coordinates.loc[index, 'EorF'].values[0]
        new_eorf = 1 if current_eorf == 0 else 0
        chessboard_coordinates.loc[index, 'EorF'] = new_eorf

    # Assuming you have the two square names, change their 'EorF' values
    square_name1 = nearest_square_mid_point1
    square_name2 = nearest_square_mid_point2

    change_eorf_value(square_name1)
    change_eorf_value(square_name2)

    # Save the updated DataFrame back to the CSV file
    chessboard_coordinates.to_csv(csv_file_path, index=False)

    return output_string

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera, change it if you have multiple cameras

# Create a variable to store the first image
image_1 = None

# Flag to track which image is being captured
capturing_image1 = True


print("Now set chess pieces on the board and then press enter!")



while True:
    ret, frame = cap.read()
    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1)


    if key == 13:  # 13 is the ASCII code for 'Enter' key
        if capturing_image1:
            image_1 = frame.copy()
            print("Start your game and do not move the chess board.")
        else:
            image_2 = frame.copy()
            print("Captured image and stored in image 2")

            result = move_detection(image_1, image_2, 'chessboard_coordinates.csv')
            print("Move Detection Result:", result)

            image_1 = image_2
            image_2 = None

        capturing_image1 = False

    if key == ord('w'):
        break

cap.release()
cv2.destroyAllWindows()
'''

