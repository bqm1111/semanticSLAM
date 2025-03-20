cmake_minimum_required(VERSION 2.8)
project(semantic_voxblox)
add_definitions(-std=c++14 -Wall -Wextra)

find_package(catkin_simple REQUIRED)

catkin_simple()

file(GLOB SOURCES src/color.cpp
  src/csv_iterator.cpp
  src/ros_params.cpp
  src/rosbag_data_provider.cpp
  src/semantic_integrator_base.cpp
  src/semantic_tsdf_integrator_factory.cpp
  src/semantic_tsdf_integrator_fast.cpp
  src/semantic_tsdf_integrator_merged.cpp
  src/semantic_tsdf_server.cpp
  src/semantic_voxblox_node.cpp)
file(GLOB HEADERS
  include/semantic_voxblox/kimera_semantics_ros/depth_map_to_pointcloud.h
  include/semantic_voxblox/kimera_semantics_ros/ros_params.h
  include/semantic_voxblox/kimera_semantics_ros/rosbag_data_provider.h
  include/semantic_voxblox/kimera_semantics_ros/semantic_tsdf_server.h
  include/semantic_voxblox/color.h
  include/semantic_voxblox/common.h
  include/semantic_voxblox/csv_iterator.h
  include/semantic_voxblox/macros.h
  include/semantic_voxblox/semantic_integrator_base.h
  include/semantic_voxblox/semantic_tsdf_integrator_factory.h
  include/semantic_voxblox/semantic_tsdf_integrator_fast.h
  include/semantic_voxblox/semantic_tsdf_integrator_merged.h
  include/semantic_voxblox/semantic_voxel.h)

cs_add_library(semantic_voxblox_lib ${SOURCES} ${HEADER})

cs_add_executable(${PROJECT_NAME}
  src/semantic_voxblox_node.cpp
)

target_link_libraries(${PROJECT_NAME} semantic_voxblox_lib ${PCL_LIBRARIES})
cs_install()
cs_export()
