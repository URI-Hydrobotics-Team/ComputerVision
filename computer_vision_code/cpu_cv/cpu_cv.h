#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>

// unified result class
class detection_result {
    public:
        detection_result(cv::Point box, float confidence, std::string object_class, time_t timestamp);
        cv::Point box;
        float confidence;
        std::string object_class;
        time_t timestamp;
};

// main cv class
// use to enforce neccessary functions
class main_cv_class  {
    public:
        virtual cv::Mat preprocess(cv::Mat frame) = 0;
        virtual std::vector<cv::Mat> inference(cv::Mat blob) = 0;
        virtual std::vector<detection_result> postprocess(std::vector<cv::Mat> result, bool nms = false) = 0;
        virtual ~main_cv_class() = default;
    
    protected:
        cv::dnn::Net model;
        std::vector<std::string> object_classes;
};

// default object detection model
class default_cv : public main_cv_class {
    public:
        default_cv(std::string model_path, std::vector<std::string> object_classes, int input_size, int output_channel, int output_dim1, int output_dim2, float conf_threshold);        
        cv::Mat preprocess(cv::Mat frame) override;
        std::vector<cv::Mat> inference(cv::Mat blob) override;
        std::vector<detection_result> postprocess(std::vector<cv::Mat> result, bool nms = false) override;
    private:
        int input_size, output_channel, output_dim1, output_dim2;
        float threshold;
        cv::Mat frame_cpy;
        cv::dnn::Image2BlobParams blob_param;
};