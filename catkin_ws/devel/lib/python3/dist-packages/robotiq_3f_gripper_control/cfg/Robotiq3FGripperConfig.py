## *********************************************************
##
## File autogenerated for the robotiq_3f_gripper_control package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'name': 'Default', 'type': '', 'state': True, 'cstate': 'true', 'id': 0, 'parent': 0, 'parameters': [{'name': 'ind_control_fingers', 'type': 'bool', 'default': False, 'level': 0, 'description': 'Set individual control of fingers', 'min': False, 'max': True, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'bool', 'cconsttype': 'const bool'}, {'name': 'ind_control_scissor', 'type': 'bool', 'default': False, 'level': 0, 'description': 'Set individual control of scissor', 'min': False, 'max': True, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'bool', 'cconsttype': 'const bool'}, {'name': 'mode', 'type': 'int', 'default': 0, 'level': 0, 'description': 'The grasping mode', 'min': -2147483648, 'max': 2147483647, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': "{'enum': [{'name': 'Basic', 'type': 'int', 'value': 0, 'srcline': 37, 'srcfile': '/home/kathan/catkin_ws/src/robotiq_1/robotiq_3f_gripper_control/cfg/Robotiq3FGripper.cfg', 'description': 'Basic mode', 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'Pinch', 'type': 'int', 'value': 1, 'srcline': 38, 'srcfile': '/home/kathan/catkin_ws/src/robotiq_1/robotiq_3f_gripper_control/cfg/Robotiq3FGripper.cfg', 'description': 'Pinch mode', 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'Wide', 'type': 'int', 'value': 2, 'srcline': 39, 'srcfile': '/home/kathan/catkin_ws/src/robotiq_1/robotiq_3f_gripper_control/cfg/Robotiq3FGripper.cfg', 'description': 'Wide mode', 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'Scissor', 'type': 'int', 'value': 3, 'srcline': 40, 'srcfile': '/home/kathan/catkin_ws/src/robotiq_1/robotiq_3f_gripper_control/cfg/Robotiq3FGripper.cfg', 'description': 'Scissor mode', 'ctype': 'int', 'cconsttype': 'const int'}], 'enum_description': 'An enum to set the grasp operation mode'}", 'ctype': 'int', 'cconsttype': 'const int'}, {'name': 'velocity', 'type': 'double', 'default': 66.0, 'level': 0, 'description': 'Set velocity for fingers', 'min': 22.0, 'max': 110.0, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force', 'type': 'double', 'default': 30.0, 'level': 0, 'description': 'Set force for fingers', 'min': 15.0, 'max': 60.0, 'srcline': 291, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}], 'groups': [], 'srcline': 246, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'class': 'DEFAULT', 'parentclass': '', 'parentname': 'Default', 'field': 'default', 'upper': 'DEFAULT', 'lower': 'groups'}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

Robotiq3FGripper_Basic = 0
Robotiq3FGripper_Pinch = 1
Robotiq3FGripper_Wide = 2
Robotiq3FGripper_Scissor = 3
