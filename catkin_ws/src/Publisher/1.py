#!/usr/bin/env python

import numpy as np
import math

e4_x = 0
e4_y = 0


centre_point_x = 400
centre_point_y = 300

offset_x = -(e4_x - centre_point_x)
offset_y = -( e4_y - centre_point_y)

angle_x = (46.2659*offset_x)/800
angle_y = (39.37*offset_y)/600

theta_x = math.radians(angle_x) 
theta_y = math.radians(angle_y)

dist_offset_x = math.tan(theta_x)* 0.623
dist_offset_y = math.tan(theta_y)* 0.623


# Define the point (x, y)
x =  dist_offset_x +0.022555625357921327 # Replace with your desired coordinates
y =  dist_offset_y + 0.7126010744372056 -0.053 # Replace with your desired coordinates

# Define the angle in degrees for clockwise rotation
angle_degrees = -55

# Convert the angle to radians
angle_radians = math.radians(angle_degrees)

# Create a rotation matrix
rotation_matrix = np.array([[math.cos(angle_radians), -math.sin(angle_radians)],
                            [math.sin(angle_radians), math.cos(angle_radians)]])

# Perform the clockwise rotation
rotated_point = np.dot(rotation_matrix, np.array([x, y]))

# Extract the new coordinates
new_y, new_x = rotated_point
new_x = - new_x  
# Print the rotated coordinates
print(f"Original Point: ({x}, {y})")
print(f"Rotated Point: ({new_x}, {new_y})")
