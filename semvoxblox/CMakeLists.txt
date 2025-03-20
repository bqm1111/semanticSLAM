cmake_minimum_required(VERSION 2.8)
project(semvoxblox)
add_definitions(-std=c++14 -Wall -Wextra)

find_package(catkin_simple REQUIRED)

catkin_simple()

file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/*.h include/semvoxblox/*.h)

cs_add_library(semvoxblox_lib ${SOURCES} ${HEADER})

cs_add_executable(${PROJECT_NAME}
  src/semvoxblox_node.cpp
)
target_link_libraries(${PROJECT_NAME} semvoxblox_lib ${PCL_LIBRARIES})
cs_install()
cs_export()
