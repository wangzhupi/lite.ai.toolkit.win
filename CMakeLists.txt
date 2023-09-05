cmake_minimum_required(VERSION 3.22)
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
    # TODO
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
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    # find onnx
    # linux
    # include_directories(/home/wangzijian/Desktop/onnxruntime-linux-x64-gpu-1.14.1/include)

    # virtual machine
     include_directories(/home/wangzijian/Desktop/onnxruntime-linux-x64-gpu-1.15.0/include)


    # 这里原来叫做lite与文件夹中的lite重名
    # 改为lite_test debug成功
    add_executable(lite_test main.cpp)
    add_subdirectory(utils)
    add_subdirectory(lite)

    target_link_libraries(lite_test utils)
    target_link_libraries(lite_test cutout)
    target_link_libraries(lite_test lightenhance)
    target_link_libraries(lite_test photo2cartoon)
    target_link_libraries(lite_test headposeestimation)
    target_link_libraries(lite_test ${OpenCV_LIBS})

    # linux
    # target_link_libraries(lite_test /home/wangzijian/Desktop/onnxruntime-linux-x64-gpu-1.14.1/lib/libonnxruntime.so)

#     virtual machine
     target_link_libraries(lite_test /home/wangzijian/Desktop/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime.so)
endif ()