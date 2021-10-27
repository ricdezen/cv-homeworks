cmake_minimum_required(VERSION 3.19)
project(cv_homework_3)

set(CMAKE_CXX_STANDARD 14)

# Needed in my case.
# set(OpenCV_DIR D:/clib/opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab5 lab5.cpp panoramic.cpp)
target_link_libraries(lab5 ${OpenCV_LIBS})