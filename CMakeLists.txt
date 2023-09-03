cmake_minimum_required(VERSION 3.25)
project(aiDenoise)

set(CMAKE_CXX_STANDARD 17)

if (WIN32)
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
    target_link_libraries(lite photo2cartoon)
    target_link_libraries(lite onnxruntime.dll)
    target_link_libraries(lite ${OpenCV_LIBS})

elseif (UNIX)
    # find opencv
    find_package(OpenCV 4.7.0 REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    # find onnx
    include_directories(/home/wangzijian/Desktop/onnxruntime-linux-x64-gpu-1.15.0/include)

    add_executable(lite main.cpp)
    add_subdirectory(utils)
    add_subdirectory(lite)

    target_link_libraries(lite utils)
    target_link_libraries(lite cutout)
    target_link_libraries(lite lightenhance)
    target_link_libraries(lite photo2cartoon)
    target_link_libraries(lite ${OpenCV_LIBS})
    target_link_libraries(test /home/wangzijian/Desktop/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime.so)
endif ()