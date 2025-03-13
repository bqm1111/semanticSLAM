cmake_minimum_required(VERSION 2.8)
project(semantic_voxblox)
add_definitions(-std=c++14 -Wall -Wextra)

find_package(catkin_simple REQUIRED)

catkin_simple()

file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/*.h include/semantic_voxblox/*.h include/semantic_voxblox/kimera_semantic_voxblox/*.h)

cs_add_library(semantic_voxblox_lib ${SOURCES} ${HEADER})

cs_add_executable(${PROJECT_NAME}
  src/semantic_voxblox_ros.cpp
)                                            
target_link_libraries(${PROJECT_NAME} semantic_voxblox_lib ${PCL_LIBRARIES})
cs_install()
cs_export()
