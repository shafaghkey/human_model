# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import stat
import sys

# find the import for catkin's python package - either from source space or from an installed underlay
if os.path.exists(os.path.join('/opt/ros/noetic/share/catkin/cmake', 'catkinConfig.cmake.in')):
    sys.path.insert(0, os.path.join('/opt/ros/noetic/share/catkin/cmake', '..', 'python'))
try:
    from catkin.environment_cache import generate_environment_script
except ImportError:
    # search for catkin package in all workspaces and prepend to path
    for workspace in '/home/shkey/catkins/meam520_ws/devel_isolated/python_orocos_kdl;/home/shkey/catkins/meam520_ws/devel_isolated/panda_simulator_examples;/home/shkey/catkins/meam520_ws/devel_isolated/panda_simulator;/home/shkey/catkins/meam520_ws/devel_isolated/panda_sim_moveit;/home/shkey/catkins/meam520_ws/devel_isolated/panda_gazebo;/home/shkey/catkins/meam520_ws/devel_isolated/panda_sim_custom_action_server;/home/shkey/catkins/meam520_ws/devel_isolated/panda_sim_controllers;/home/shkey/catkins/meam520_ws/devel_isolated/franka_tools;/home/shkey/catkins/meam520_ws/devel_isolated/franka_moveit;/home/shkey/catkins/meam520_ws/devel_isolated/panda_moveit_config;/home/shkey/catkins/meam520_ws/devel_isolated/panda_hardware_interface;/home/shkey/catkins/meam520_ws/devel_isolated/orocos_kinematics_dynamics;/home/shkey/catkins/meam520_ws/devel_isolated/meam520_labs;/home/shkey/catkins/meam520_ws/devel_isolated/franka_visualization;/home/shkey/catkins/meam520_ws/devel_isolated/franka_ros_interface;/home/shkey/catkins/meam520_ws/devel_isolated/franka_ros_controllers;/home/shkey/catkins/meam520_ws/devel_isolated/franka_ros;/home/shkey/catkins/meam520_ws/devel_isolated/franka_panda_description;/home/shkey/catkins/meam520_ws/devel_isolated/franka_interface;/home/shkey/catkins/meam520_ws/devel_isolated/franka_gazebo;/home/shkey/catkins/meam520_ws/devel_isolated/franka_example_controllers;/home/shkey/catkins/meam520_ws/devel_isolated/franka_control;/home/shkey/catkins/meam520_ws/devel_isolated/franka_hw;/home/shkey/catkins/meam520_ws/devel_isolated/franka_core_msgs;/home/shkey/catkins/meam520_ws/devel_isolated/franka_msgs;/home/shkey/catkins/meam520_ws/devel_isolated/franka_gripper;/home/shkey/catkins/meam520_ws/devel_isolated/franka_description;/opt/ros/noetic'.split(';'):
        python_path = os.path.join(workspace, 'lib/python3/dist-packages')
        if os.path.isdir(os.path.join(python_path, 'catkin')):
            sys.path.insert(0, python_path)
            break
    from catkin.environment_cache import generate_environment_script

code = generate_environment_script('/home/shkey/catkins/human_model_ws/src/human_model/build/devel/env.sh')

output_filename = '/home/shkey/catkins/human_model_ws/src/human_model/build/catkin_generated/setup_cached.sh'
with open(output_filename, 'w') as f:
    # print('Generate script for cached setup "%s"' % output_filename)
    f.write('\n'.join(code))

mode = os.stat(output_filename).st_mode
os.chmod(output_filename, mode | stat.S_IXUSR)
