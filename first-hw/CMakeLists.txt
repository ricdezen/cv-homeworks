cmake_minimum_required(VERSION 3.19)
project(cv_homework_1)

set(CMAKE_CXX_STANDARD 14)

# Needed in my case.
set(OpenCV_DIR C:/tools/opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab2 lab2.cpp)
target_link_libraries(lab2 ${OpenCV_LIBS})

add_executable(lab3 lab3.cpp filter.cpp)
target_link_libraries(lab3 ${OpenCV_LIBS})