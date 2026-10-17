cmake_minimum_required(VERSION 3.10)

# add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
# add_link_options(-fsanitize=address,undefined)

# Find OpenCV and TensorRT
find_package(OpenCV REQUIRED)

# change it to the correct AVOE directory
file(GLOB_RECURSE EXTERNAL_SOURCES CONFIGURE_DEPENDS ../../../AVOE/core/*.cpp ../../../AVOE/lib/*.cpp)

list(FILTER EXTERNAL_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

add_executable(Detection main.cpp cpu_cv.cpp ${EXTERNAL_SOURCES})

target_include_directories( Detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} "../../../AVOE" "../../../AVOE/core" "../../../AVOE/lib" ${OpenCV_INCLUDE_DIRS} ${GSTREAMER_INCLUDE_DIRS} ${GSTREAMER_APP_INCLUDE_DIRS} ${GSTREAMER_VIDEO_INCLUDE_DIRS} ${GSTREAMER_SERVER_INCLUDE_DIRS})

target_link_libraries( Detection ${OpenCV_LIBS} ${GST_APP_LIBRARIES} ${GST_VIDEO_LIBRARIES} ${GST_SERVER_LIBRARIES})