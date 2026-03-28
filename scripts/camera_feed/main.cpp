#include <iostream>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>


int main() {
   // Initialize the gstreamer object
    gst_init(nullptr, nullptr);

    std::cout << "Gstreamer initialized\n";

    GstRTSPServer* server;
    GMainLoop* loop;
    GstRTSPMediaFactory* factory;
    GstRTSPMountPoints* mounts;

    // Create a new server
    server = gst_rtsp_server_new();
    std::cout << "GstRTSPServer object created\n";

    // Create a loop that the server will run in
    loop = g_main_loop_new(NULL, false);
    factory = gst_rtsp_media_factory_new();

    /* 
        current camera already compress frame into H.264 format already 
        So no need to convert again

        This will get frames from camera (already in H.264) -> specify the height and width -> parse the H.264 stream -> put it in a queue to create a buffer
        -> encode H.264 data into RTP packets
    */
    gst_rtsp_media_factory_set_launch(factory, "( videotestsrc pattern=ball ! video/h-264, width=1280, height=720 ! h264parse ! queue ! rtph264pay name=pay0 pt=96 )");

    // Some other possible setting to use
    // gst_rtsp_media_factory_set_launch(factory, "( v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=1280, height=720 ! videoconvert ! x264enc ! h264parse ! queue ! rtph264pay name=pay0 pt=96 )");

    mounts = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_add_factory(mounts, "/cam1", factory);

    // factory = gst_rtsp_media_factory_new();
    // gst_rtsp_media_factory_set_launch(factory, "( udpsrc port=5001 ! queue ! application/x-rtp,encoding-name=H264 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! rtph264pay name=pay0 pt=96 )");
    // gst_rtsp_mount_points_add_factory(mounts, "/test1", factory);

    g_object_unref(mounts);

    gst_rtsp_server_attach(server, NULL);

    std::cout << "Starting the server\n";
    g_main_loop_run(loop);

    return 0;
}