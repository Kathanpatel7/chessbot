#!/usr/bin/env python

import rospy
import sys
import moveit_commander
import numpy as np
import math

# Initialize ROS node
rospy.init_node('ur10e_moveit_example', anonymous=True)

# Initialize MoveIt Commander
moveit_commander.roscpp_initialize(sys.argv)

# Create a RobotCommander object
robot = moveit_commander.RobotCommander()

# Create a MoveGroupCommander for the end effector group
group_name = "manipulator"  # Replace with your specific group name
group = moveit_commander.MoveGroupCommander(group_name)

# Get the current pose of the end effector
current_pose = group.get_current_pose()

# Extract X, Y, and Z coordinates
x_coord = current_pose.pose.position.x
y_coord = current_pose.pose.position.y
z_coord = current_pose.pose.position.z

# Print the current pose and separated coordinates
print("Current End Effector Pose:")
print(current_pose)
print(f"X Coordinate: {x_coord}")
print(f"Y Coordinate: {y_coord}")
print(f"Z Coordinate: {z_coord}")



# Define the point (x, y)
x =  x_coord # Replace with your desired coordinates
y = y_coord # Replace with your desired coordinates

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
new_y, new_x = rotated_point
new_y = -new_y
# Print the rotated coordinates
print(f"Original Point: ({x}, {y})")
print(f"Rotated Point: ({new_x}, {new_y})")

