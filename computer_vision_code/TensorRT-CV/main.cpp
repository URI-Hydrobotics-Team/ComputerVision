#include "detection.h"
#include <chrono>

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kINFO){
            std::cout << msg << std::endl;
        }
    }
};


int main(){

    Logger logger;
    std::vector<std::string> object_classes = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
    detection model(&logger, object_classes, 640, 1, 84, 8400, 32);


    // This line is used to build an engine file
    // model.build_engine("/home/hydro/ComputerVision/models/yolov5nu.onnx", "/home/hydro/ComputerVision/models/yolov5_FP32_testing.engine");

    model.load_model("/home/hydro/ComputerVision/models/yolov5_FP32_testing.engine");

    cv::VideoCapture camera;
    camera.open(0);

    // infinite while loop 

        // wait for data, pause until something is received from AVOE

        // Model inference

        // crop bounding box section

        // do edge detection

        // use applypolydp to get a shape

        // sort corners

        // find distance (PnP or naive size distance measurement)
        

    

    // Used only for double buffering
    // cv::Mat frames[2];

    // int cur_index = 0;
    // camera.read(frames[cur_index]);
    // model.preprocess_async(cur_index, frames[cur_index]);
    // model.inference_async(cur_index);

    // auto start = std::chrono::high_resolution_clock::now();

    // for(int i = 0; i < 40000; i++){
    //     // std::cout << i << "\n";
    //     cur_index = !cur_index;
    //     camera.read(frames[cur_index]);
    //     model.preprocess_async(cur_index, frames[cur_index]);
    //     model.inference_async(cur_index);
    //     model.postprocess_async(!cur_index);
    // }

    // model.postprocess_async(!cur_index);

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    // std::cout << "Average inference fps is: " << 40000 / duration.count() << "\n";

    // return 0;
}