cmake_minimum_required(VERSION 3.0.2)
project(align_depth_to_color)

# # Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# # Find catkin macros and libraries
# # if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
# # is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(

  # INCLUDE_DIRS include
  # LIBRARIES align_depth_to_color
  # CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs
  # DEPENDS system_lib
  CATKIN_DEPENDS ${PACKAGE_DEPENDENCIES}
)

include_directories(

  # include
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS
  src/align_depth_to_color.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
