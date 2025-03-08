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
import geometry_msgs.msg
import time 
import csv

height_Tcp = 0.62017
####################################Extracting Pixels from CSV

# Specify the path to your CSV file
csv_file = "/home/kathan/catkin_ws/src/Publisher/Final_coordinates_hard.csv"  # Replace with your CSV file path

# The value to search for in the first column
search_value = "a8"

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
################################
#Pixel to Coordinates

import numpy as np
import math

e4_x = 401 # float(row[1])  #input pixels
e4_y = 300 #float(row[2])


centre_point_x = 400
centre_point_y = 300

offset_x = (e4_x - centre_point_x)
offset_y = ( e4_y - centre_point_y)

angle_x = (46.2659*offset_x)/800
angle_y = (39.37*offset_y)/600

theta_x = math.radians(angle_x) 
theta_y = math.radians(angle_y)

dist_offset_x = math.tan(theta_x)* height_Tcp
dist_offset_y = math.tan(theta_y)* height_Tcp


# Define the point (x, y)
x =  dist_offset_x  - 0.011 + float(row[1]) # Replace with your desired coordinates
y =  dist_offset_y + 0.048 + float(row[2]) # Replace with your desired coordinates

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
print(f"Rotated Point: ({new_x}, {new_y})")


try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))



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
        pose_goal.orientation.x = 0.9243235476448
        pose_goal.orientation.y = 0.3815552312064111
        pose_goal.orientation.z = 0.004676821544333007
        pose_goal.orientation.w = 0.004439836544472645
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
        
def main():
    try:
        print("")
        print("----------------------------------------------------------")
        print("Welcome to the MoveIt MoveGroup Python Interface Tutorial")
        print("----------------------------------------------------------")
        print("Press Ctrl-D to exit at any time")
        print("")
        input(
            "============ Press `Enter` to begin the tutorial by setting up the moveit_commander ..."
        )
        tutorial = MoveGroupPythonInterfaceTutorial()

        input(
            "============ Press `Enter` to execute a movement using a joint state goal ..."
        )
        tutorial.go_to_joint_state()
        
        input(f"============ Press `Enter` to execute a movement using a pose goal {search_value}...")
        #tutorial.go_to_pose_goal(new_x, new_y,0.1640)
        
        #tutorial.go_to_pose_goal(new_x, new_y,0.0620)
        
        #input("============ Press `Enter` to execute a movement using a pose goal home position ...")
        
        '''

        input("============ Press `Enter` to execute a movement using a pose goal e4...")
        
        pose_goal1 = geometry_msgs.msg.Pose()
        pose_goal1.orientation.x = 0.9243235476448
        pose_goal1.orientation.y = 0.3815552312064111
        pose_goal1.orientation.z = 0.004676821544333007
        pose_goal1.orientation.w = 0.004439836544472645
        pose_goal1.position.x =-0.26546338816271076
        pose_goal1.position.y =  0.5942884108626451
        pose_goal1.position.z = 0.03640
        
        tutorial.go_to_pose_goal(pose_goal1)
        
        #input("============ Press `Enter` to execute a movement using a pose goal e4...")
        
        pose_goal11 = geometry_msgs.msg.Pose()
        pose_goal11.orientation.x = 0.9243235476448
        pose_goal11.orientation.y = 0.3815552312064111
        pose_goal11.orientation.z = 0.004676821544333007
        pose_goal11.orientation.w = 0.004439836544472645
        pose_goal11.position.x =-0.26546338816271076
        pose_goal11.position.y =  0.5942884108626451
        pose_goal11.position.z = 0.14040
        
        tutorial.go_to_pose_goal(pose_goal11)
        
        #input("============ Press `Enter` to execute a movement using a pose goal e8...")
        
        pose_goal12 = geometry_msgs.msg.Pose()
        pose_goal12.orientation.x = 0.9243235476448
        pose_goal12.orientation.y = 0.3815552312064111
        pose_goal12.orientation.z = 0.004676821544333007
        pose_goal12.orientation.w = 0.004439836544472645
        pose_goal12.position.x =-0.3686916000527256
        pose_goal12.position.y =  0.7417135759129223
        pose_goal12.position.z = 0.14040


        tutorial.go_to_pose_goal(pose_goal12)
        #input("============ Press `Enter` to execute a movement using a pose goal e8...")
        
        pose_goal10 = geometry_msgs.msg.Pose()
        pose_goal10.orientation.x = 0.9243235476448
        pose_goal10.orientation.y = 0.3815552312064111
        pose_goal10.orientation.z = 0.004676821544333007
        pose_goal10.orientation.w = 0.004439836544472645
        pose_goal10.position.x =-0.3686916000527256
        pose_goal10.position.y =  0.7417135759129223
        pose_goal10.position.z = 0.038


        tutorial.go_to_pose_goal(pose_goal10)
        
        time.sleep(0.35)
        '''
        #tutorial.go_to_joint_state()
    except rospy.ROSInterruptException:
    	return
    except KeyboardInterrupt:
    	return
    	
if __name__ == "__main__":
    main()


