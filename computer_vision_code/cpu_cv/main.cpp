#include "cpu_cv.h"

int main() {
    std::vector<std::string> object_classes = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
    main_cv_class *model = new default_cv("/ComputerVision/models/testing_model.onnx", object_classes, 640, 1, 84, 8400, 0.20);

    
    for(int i = 0; i < 10000; i++) {
        cv::Mat frame = cv::imread("/ComputerVision/computer_vision_code/TensorRT-CV/test_data/dog_img.jpg");
        if(frame.empty()) {
            std::cout << "failed\n";
            return 0;
        }
        std::vector<detection_result> result = model->postprocess(model->inference(model->preprocess(frame)));
        for(detection_result detected : result) {
            cv::rectangle(frame, detected.box, (0, 255, 0));
            std::cout << "object: " << detected.object_class << ", " << "conf: " << detected.confidence << " offset: " << detected.offset << "\n";
        }

        cv::imshow("test", frame);
        cv::waitKey(100);
    }

    delete model;
    return 0;
}