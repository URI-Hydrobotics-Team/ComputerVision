cmake_minimum_required(VERSION 3.10)
project(Detection LANGUAGES CXX CUDA)

# Find OpenCV and TensorRT
find_package(OpenCV REQUIRED)
# set(TENSORRT_INCLUDE_DIRS
#     /usr/include/x86_64-linux-gnu
# )

# List all the TensorRT libraries you need.
set(TENSORRT_LIBRARIES
    path to libnvinfer.so file
    path to libnvinfer_plugin.so file
    path to libnvonnxparser.so file
)

# change it to the correct AVOE directory
file(GLOB_RECURSE EXTERNAL_SOURCES CONFIGURE_DEPENDS ../../../AVOE/core/*.cpp ../../../AVOE/lib/*.cpp)

list(FILTER EXTERNAL_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

add_executable(Detection main.cpp detection.cpp helper_functions.cu ${EXTERNAL_SOURCES})

target_include_directories( Detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} "../../../AVOE" "../../../AVOE/core" "../../../AVOE/lib" ${OpenCV_INCLUDE_DIRS} ${GSTREAMER_INCLUDE_DIRS} ${GSTREAMER_APP_INCLUDE_DIRS} ${GSTREAMER_VIDEO_INCLUDE_DIRS} ${GSTREAMER_SERVER_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS})

target_link_libraries( Detection ${OpenCV_LIBS} ${GST_APP_LIBRARIES} ${GST_VIDEO_LIBRARIES} ${GST_SERVER_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})