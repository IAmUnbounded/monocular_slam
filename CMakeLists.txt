cmake_minimum_required(VERSION 2.8)
project(aerial)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCES iwill.cpp)
add_executable(test ${SOURCES})
find_package( OpenCV 3 REQUIRED PATHS "/usr/local/include/OpenCV" )
include_directories( ${OpenCV_INCLUDE_DIRS})

target_link_libraries( test ${OpenCV_LIBS} )

