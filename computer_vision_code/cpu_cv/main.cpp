#include "cpu_cv.h"

int main() {
    std::vector<std::string> object_classes = { "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool" };
    main_cv_class *model = new obb_cv("yolov8n-obb.onnx", object_classes, 1024, 1, 20, 21504, 0.70);

    
    for(int i = 0; i < 10000; i++) {
        cv::Mat frame = cv::imread("boat.jpg");
        if(frame.empty()) {
            std::cout << "failed\n";
            return 0;
        }
        std::vector<detection_result> result = model->postprocess(model->inference(model->preprocess(frame)), false);
        for(detection_result detected : result) {
            cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f((detected.box.x + detected.box.width / 2), (detected.box.y + detected.box.height / 2)), cv::Size2f(detected.box.width, detected.box.height), detected.angle);

            cv::Point2f vertices[4];
            rRect.points(vertices);
            for (int i = 0; i < 4; i++)
                cv::line(frame, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 2);
            std::cout << "object: " << detected.object_class << ", " << "conf: " << detected.confidence << " offset: " << detected.offset << "\n";
        }

        cv::imshow("test", frame);
        cv::waitKey(100);
    }

    delete model;
    return 0;
}