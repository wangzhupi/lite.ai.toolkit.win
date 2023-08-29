cmake_minimum_required(VERSION 3.25)
project(aiDenoise)

set(CMAKE_CXX_STANDARD 17)

# set onnxruntime
link_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/include)

# set opencv
set(OpenCV_DIR E:\\win-4.7.0-dll\\x64\\vc16\\lib)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lite main.cpp)


# 添加子目录
add_subdirectory(utils)
add_subdirectory(lite)

target_link_libraries(lite utils)
target_link_libraries(lite cutout)
target_link_libraries(lite lightenhance)
target_link_libraries(lite onnxruntime.dll)
target_link_libraries(lite ${OpenCV_LIBS})
