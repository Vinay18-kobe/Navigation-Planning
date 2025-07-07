
# Planning and Control for Mobile Robot Navigation using TurtleBot3 in ROS
Description

This project implements an integrated navigation system for a TurtleBot3 differential-drive robot using the Robot Operating System (ROS) Noetic and Gazebo simulation environment

A brief description of what this project does and who it's for


## Requirements
To run this project successfully, ensure the following software and packages are installed:
Operating System
Ubuntu 20.04 LTS
ROS
ROS Noetic
Robot Simulation
TurtleBot3 Packages
turtlebot3
turtlebot3_gazebo
turtlebot3_simulations
turtlebot3_navigation
turtlebot3_slam
turtlebot3_description
## Set up

1.Terminal 1: Start roscore

Keep this terminal running.

2.Terminal 2: Launch Gazebo with TurtleBot3

roslaunch auto_world auto_navigation.launch

3.We need to give the intial pose(2d pose Estimation)

4.After that we need to give (2d Nav Goal)

5.By assigning these commands need to wait till the bot turtlebot reaches the goal.
## Author

Kasari Vinay Kumar

Technise Hoschule Deggendorf

vinay.kasari@stud.th-deg.de
## Licence

This project is for academic and educational purposes.
