cmake_minimum_required(VERSION 3.12)

project(camera_feed)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0) 
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GST_SERVER REQUIRED gstreamer-rtsp-server-1.0)

include_directories( ${GLIB_INCLUDE_DIRS} ${GSTREAMER_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS} ${GST_VIDEO_INCLUDE_DIRS} ${GST_SERVER_INCLUDE_DIRS} )

add_executable(camera_feed main.cpp)
target_link_libraries( camera_feed ${GLIB_LIBRARIES} ${GSTREAMER_LIBRARIES} ${GST_APP_LIBRARIES} ${GST_VIDEO_LIBRARIES} ${GST_SERVER_LIBRARIES} )