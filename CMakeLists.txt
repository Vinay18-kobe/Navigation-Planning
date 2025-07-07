cmake_minimum_required(VERSION 3.0.2)
project(auto_world)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  gazebo_ros
  xacro
  tf
  tf2_ros
  nav_msgs
  geometry_msgs
)

## Declare catkin package
catkin_package()

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install Python scripts (controllers and planners)
catkin_install_python(PROGRAMS
  scripts/navigator.py
  scripts/potential_fields.py
  scripts/kinematic_controller.py
  scripts/a_star_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Install world files
install(DIRECTORY world/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/world
)

## Install map files
install(DIRECTORY maps/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/maps
)

## Install RViz config (optional)
install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

