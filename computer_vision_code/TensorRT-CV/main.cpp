#include "detection.h"
#include <opencv2/calib3d.hpp>
#include <sys/socket.h>
#include <netinet/in.h> 
#include <arpa/inet.h> 
#include "../AVOE/core/io.h"

// MODIFY THESE DATA TO MATCH BASIC MODE RECEIVER SET UP
constexpr bool basic = true;
constexpr const char* dest_ip = "127.0.0.1";
constexpr uint16_t dest_port = 8000;
constexpr uint16_t local_port = 8080;
#define PORT_CORE_INPUT 8101
#define IP_CORE "10.42.0.69"

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING){
            std::cout << msg << std::endl;
        }
    }
};

// Gloabl points for object real world dimensions
// camera stuff
// camera matrix
// REQUIRED CAMERA CALIBRATION TO OBTAIN THIS DATA
cv::Mat cameraMatrix = (
    cv::Mat_<double>(3, 3) <<
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.00000000e+00
);

// distortion matrix
// REQUIRED CAMERA CALIBRATION TO OBTAIN THIS DATA
cv::Mat distort = (
    cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0
);

// hashmap to make object dimension adding easier
// For naive distance object
std::unordered_map<std::string, float> naive_area = {
    {"Gate", 5640},
    {"Slalom", 2160},
    {"Octagon", 11802.56}
};

// Gate
std::vector<cv::Point3f> gate_corners = {cv::Point3f(-120/2.0f, 47/2.0f, 0), cv::Point3f(120/2.0f, 47/2.0f, 0), cv::Point3f(120/2.0f, -47/2.0f, 0), cv::Point3f(-120/2.0f, -47/2.0f, 0)};

// need octagon dimensions
std::vector<cv::Point3f> octagon_corners = {};

// Get pixel focal length
float get_focal_length_pixel(float focal_length_mm, float sensor_width, int resolution_width) {
    return focal_length_mm * (resolution_width / sensor_width);
}

// very simple area based distance estimation, could be very inaccurate due to rotation or view angles
// but fast and simple to use
void naive_distance(CV_data &result, float focal_length) {
    float known_area = naive_area.at(std::string(result.object_name));
    float numerator = std::sqrt(known_area) * focal_length;
    float denominator = std::sqrt(result.bbox.area());
    
    result.z = numerator / denominator;
}

void correct_corner(cv::Point2f *points, size_t size) {
    int low = 0;

    for(int i = 1; i < size; i++) {
        if(points[i].x + points[i].y < points[low].x + points[low].y) {
            low = i;
        }
    }

    std::rotate(points, points + low, points + size);

    if(points[1].y > points[3].y) {
        std::reverse(points + 1, points + size);
    }
}

void atan_angle(cv::Point2f *points, size_t size, float cx, float cy) {
    std::sort(points, points + size, [cx, cy](cv::Point2f &point1, cv::Point2f &point2) { 
        float angle1 = std::atan2(point1.y - cy, point1.x - cx);
        float angle2 = std::atan2(point2.y - cy, point2.x - cx);

        float distance1 = ((cx - point1.x) * (cx - point1.x)) + ((cy - point1.y) * (cy - point1.y));
        float distance2 = ((cx - point2.x) * (cx - point2.x)) + ((cy - point2.y) * (cy - point2.y));

        return std::tie(angle1, distance1) < std::tie(angle2, distance2);
    });
}

cv::Point2f get_center(cv::Point2f *points, size_t size, float corner) {
    cv::Point2f center{0.f, 0.f};

    for(int i = 0; i < size; i++) {
        center.x += points[i].x;
        center.y += points[i].y;
    }

    float cx = center.x / corner;
    float cy = center.y / corner;

    return cv::Point2f(cx, cy);
}

// PnP function to do distance estimation
// REQUIRE CAMERA CALIBRATION DATA TO FUNCTION
void PnP_distance(CV_data &result, cv::Mat frame, std::string object_name) {
    cv::Mat rvec;
    cv::Mat tvec;

    // crop bounding box section
    cv::Rect roi = result.bbox;
    cv::Mat cropped = frame(roi);

    // do edge detection
    cv::Mat gray;
    cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    cv::Mat canny;
    cv::Canny(blur, canny, 100, 200, 3, false);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // get the corners of the object
    if(object_name == "Gate") {
        std::vector<cv::Point> all_points;

        for (auto& c : contours)
        {
            if (cv::contourArea(c) > 100)
                all_points.insert(all_points.end(), c.begin(), c.end());
        }
        
        if(all_points.size() == 0) {
            result.z = -1;
            return;
        }
        cv::RotatedRect points = cv::minAreaRect(all_points);

        // get the four corner vertices
        cv::Point2f corners[4];
        points.points(corners);

        // angle rotate points
        cv::Point2f cx_cy = get_center(corners, 4, 4);
        float cx = cx_cy.x, cy = cx_cy.y;

        // sort corners
        atan_angle(corners, 4, cx, cy);
        correct_corner(corners, 4);

        std::vector<cv::Point2f> final_points(corners, corners + 4);

        for(auto &val : final_points) {
            val.x += roi.x;
            val.y += roi.y;
        }

        // for (auto& p : final_points) {
        //     cv::circle(frame, p, 10, cv::Scalar(0, 0, 255), -1);
        // }

        // cv::namedWindow("PnP Debugging", cv::WINDOW_NORMAL);
        // cv::resizeWindow("PnP Debugging", 1200, 800);
        // cv::imshow("PnP Debugging", frame);
        // cv::waitKey(0);

        // find distance
        bool success = cv::solvePnP(gate_corners, final_points, cameraMatrix, distort, rvec, tvec);

        if(success) {
            double distance = tvec.at<double>(2);
            result.z = distance;
            std::cout << "Object: " << result.object_name << " distance: " << result.z << "\n";
        } else {
            result.z = -1;
        }
    } else if(object_name == "Octagon") {
        std::vector<cv::Point2f> approxCurve;
        for(auto c : contours) {
            float ep = 0.02 * cv::arcLength(c, true);
            cv::approxPolyDP(c, approxCurve, ep, true);

            if(approxCurve.size() == 8) {
                break;
            }
        }

        // Did not find any contour of an octagon
        if(approxCurve.size() != 8) {
            result.z = -1;
            return;
        }

        // angle rotate points
        cv::Point2f cx_cy = get_center(approxCurve.data(), approxCurve.size(), 8);
        float cx = cx_cy.x, cy = cx_cy.y;

        atan_angle(approxCurve.data(), approxCurve.size(), cx, cy);
        bool success = cv::solvePnP(octagon_corners, approxCurve, cameraMatrix, distort, rvec, tvec);

        if(success) {
            double distance = tvec.at<double>(2);
            result.z = distance;
            std::cout << "Object: " << result.object_name << " distance: " << result.z << "\n";
        } else {
            result.z = -1;
        }
    }
}

int main(int argc, char* argv[]){

    Logger logger;
    std::vector<std::string> object_classes = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
    detection model(&logger, object_classes, 640, 1, 84, 8400, 32);


    // This line is used to build an engine file
    if(argc == 4 && strncmp(argv[1], "build", 32) == 0) {
        std::string onnx_path = argv[2];
        std::string dest_path = argv[3];
        model.build_engine(onnx_path, dest_path);
        std::cout << "Model build successful, path is: " << dest_path << "\n";
        return EXIT_SUCCESS;
    }

    // Engine path example: "/home/hydro/ComputerVision/models/yolov5_FP32_testing.engine"
    model.load_model("/home/hydro/ComputerVision/models/yolov5_FP32_testing.engine");

    cv::VideoCapture camera;

    // Warm up to initialize GPU memory, allocation, and the model

    // model.inference(frame);
    // std::vector<CV_data> result = model.postprocess("keyboard");

    // PnP_distance(result, frame, "keyboard");
    // naive_distance(result, 595, 3288.7);

    // for(int i = 0; i < 10000; i++){
    //     camera.read(frame);
    //     if(frame.empty()) {
    //         std::cout << "Empty frame\n";
    //         break;
    //     }
    //     model.inference(frame);
    //     std::vector<CV_data> result = model.postprocess();
    // }

    cv::Mat frame;
    camera.open(0);

    // initialize avoe_comm_transmitter object to communicate data from CV to AVOE
    avoe_comm_transmitter CV_to_AVOE("message", "cv", PORT_CORE_INPUT, IP_CORE);
    
    char buffer[1024];
    int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in servaddr, clientaddr; 
    socklen_t len;
    ssize_t bytes;

    if(socket_fd < 0) {
        std::cerr << "Failed to create socket\n";
        std::exit(EXIT_FAILURE);
    }

    // Not basic mode
    if(!basic) {
        memset(&servaddr, 0, sizeof(servaddr)); 
        memset(&clientaddr, 0, sizeof(clientaddr)); 
    
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = INADDR_ANY;
        servaddr.sin_port = htons(local_port);
    
        if(bind(socket_fd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0 ) {
            std::cerr << "Unable to bind server\n";
            std::exit(EXIT_FAILURE);
        }

        len = sizeof(clientaddr);
    } else {
        // transmitter timer set to 300 ms
        CV_to_AVOE.set_timer(300);
    }

    while(true){
        if(basic) {
            // grab frame and make sure frame is not empty
            camera.read(frame);
            if(frame.empty()) {
                std::cerr << "Frame is empty\n";
                std::exit(EXIT_FAILURE);
            }
    
            model.inference(frame);
            // * means return any detected object
            std::vector<CV_data> result = model.postprocess("*");

            // calculate distance for all detected objects
            for(CV_data &detection : result) {
                if(argc == 1 || strncmp(argv[1], "naive", 32) == 0) {
                    naive_distance(detection, 3000);
                } else {
                    std::string obj(detection.object_name);
                    if(obj == "Slalom") {
                        naive_distance(detection, 3000);
                    } else {
                        PnP_distance(detection, frame, obj);
                    }
                }
            }

            // append all object data together
            std::string send;
            for(int i = 0; i < result.size(); i++) {
                // Format: name|confidence|timestamp|pixel x offset|pixel y offset|actual distance in z
                // Seperated by | of different data and seperated by \n for different detected objects
                send += std::string(result[i].object_name) + "|" + std::to_string(result[i].confidence) + "|" + std::to_string(result[i].time) + "|" + std::to_string(result[i].pixel_x_offset) + "|" + std::to_string(result[i].pixel_y_offset) + "|" + std::to_string(result[i].z) + '\n';
            }

            // set message to send
            CV_to_AVOE.set_message(&send[0], send.length());
            // transmit the message
            CV_to_AVOE.refresh();
        } else {

            bytes = recvfrom(socket_fd, buffer, 1023, 0, ( struct sockaddr *) &clientaddr, &len);
            
            if(bytes > 0) {
                buffer[bytes] = '\0';
                // Placeholder for message
                // Assume format is just the object name
                std::string obj(buffer);
                camera.read(frame);
    
                if(frame.empty()) {
                    std::cerr << "Frame is empty\n";
                    std::exit(EXIT_FAILURE);
                }
        
                model.inference(frame);
                std::vector<CV_data> result = model.postprocess(obj);
    
                for(CV_data &detection : result) {
                    if(argc == 1 || strncmp(argv[1], "naive", 32) == 0) {
                        naive_distance(detection, 3000);
                    } else {
                        if(obj == "Slalom") {
                            naive_distance(detection, 3000);
                        } else {
                            PnP_distance(detection, frame, obj);
                        }
                    }
                }
    
                std::string send;
                for(int i = 0; i < result.size(); i++) {
                    // Format: name|confidence|timestamp|pixel x offset|pixel y offset|actual distance in z
                    // Seperated by | of different data and seperated by \n for different detected objects
                    send += std::string(result[i].object_name) + "|" + std::to_string(result[i].confidence) + "|" + std::to_string(result[i].time) + "|" + std::to_string(result[i].pixel_x_offset) + "|" + std::to_string(result[i].pixel_y_offset) + "|" + std::to_string(result[i].z) + '\n';
                }
    
                sendto(socket_fd, send.c_str(), send.length(), MSG_CONFIRM, (const struct sockaddr *) &clientaddr, len); 
    
            } else if(bytes == 0) {
                break;
            } else {
                return 1;
            }
        }
    }

    return 0;
}