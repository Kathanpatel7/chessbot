#!/usr/bin/env python
import numpy as np
import math

# Define the point (x, y)
x =  -0.28458588695387443 # Replace with your desired coordinates
y = 0.5016803305105838  # Replace with your desired coordinates

# Define the angle in degrees for clockwise rotation
angle_degrees = 55

# Convert the angle to radians
angle_radians = math.radians(angle_degrees)

# Create a rotation matrix
rotation_matrix = np.array([[math.cos(angle_radians), -math.sin(angle_radians)],
                            [math.sin(angle_radians), math.cos(angle_radians)]])

# Perform the clockwise rotation
rotated_point = np.dot(rotation_matrix, np.array([x, y]))

# Extract the new coordinates
new_x, new_y = rotated_point

# Print the rotated coordinates
print(f"Original Point: ({x}, {y})")
print(f"Rotated Point: ({new_x}, {new_y})")

