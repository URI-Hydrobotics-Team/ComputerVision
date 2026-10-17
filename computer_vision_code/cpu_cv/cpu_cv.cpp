#include "cpu_cv.h"

detection_result::detection_result(cv::Point box, float confidence, std::string object_class, time_t timestamp) {
    this->box = box;
    this->confidence = confidence;
    this->object_class = object_class;
    this->timestamp = timestamp;
}

default_cv::default_cv(std::string model_path, std::vector<std::string> object_classes, int input_size, int output_channel, int output_dim1, int output_dim2, float conf_threshold) {
    // Load the model
    model = cv::dnn::readNetFromONNX(model_path);

    // image size
    this->input_size = input_size;
    this->output_channel = output_channel;
    this->output_dim1 = output_dim1;
    this->output_dim2 = output_dim2;
    this->object_classes = object_classes;
    threshold = conf_threshold;

    // parameters for preprocessing the frame
    blob_param.datalayout = cv::dnn::DNN_LAYOUT_NCHW;
    blob_param.ddepth = CV_32F;
    blob_param.mean = cv::Scalar();
    blob_param.paddingmode = cv::dnn::DNN_PMODE_LETTERBOX;
    blob_param.scalefactor = 1.0 / 255.0;
    blob_param.size = cv::Size(640, 640);
    blob_param.swapRB = true;
}      

cv::Mat default_cv::preprocess(cv::Mat frame) {
    frame_cpy = frame;
    cv::Mat blob;
	blob = cv::dnn::blobFromImageWithParams(frame, blob_param);

    return blob;
}

std::vector<cv::Mat> default_cv::inference(cv::Mat blob) {
    // Pass the frame to the model
    model.setInput(blob);
    std::vector<cv::Mat> output;
    model.forward(output, model.getUnconnectedOutLayersNames());
    
    return output;
}

std::vector<detection_result> default_cv::postprocess(std::vector<cv::Mat> result, bool nms = false) {
    int dimension = result[0].size[1];
    int detection_nums = result[0].size[2];
    std::vector<detection_result> final_result;
    int center_x = frame_cpy.cols / 2;
    int center_y = frame_cpy.rows / 2;

    if(nms) {
        for(int i = 0; i < dimension; i++) {
            float* cur_row_ptr = result[0].ptr<float>(i);
            float cur_threshold = cur_row_ptr[4];
            if(cur_threshold < threshold)  {
                break;
            }

            cv::Rect bounds = cv::Rect(cv::Point2i(cur_row_ptr[0], cur_row_ptr[1]), cv::Point2i(cur_row_ptr[2], cur_row_ptr[3]));
            cv::Rect final_bounds = blob_param.blobRectToImageRect(bounds, cv::Size(frame_cpy.cols, frame_cpy.rows));

            time_t timestamp;
            std::time(&timestamp);

            final_result.push_back(detection_result(cv::Point((final_bounds.x + final_bounds.width / 2) - center_x, center_y - (final_bounds.y + final_bounds.height / 2)), cur_threshold, object_classes[cur_row_ptr[5]], timestamp));
        }
    } else {
        std::vector<std::string> detected_object;
		std::vector<cv::Rect> bounding_box;
		std::vector<float> confidence;
		std::vector<int> indices;

        // Transpose the output for easier interpretation
        cv::Mat transposed;
        cv::transpose(result[0].reshape(1, dimension), transposed); 

        //Loop through the transposed result to get the data of all the good detections
        cv::Mat access_row;
        for (int i = 0; i < detection_nums; i++) {
            double max_conf;
            cv::Point conf_indx;
            access_row = transposed.row(i);

            //Find the class with the highest probability of a specific detection
            cv::minMaxLoc(access_row.colRange(4, dimension), nullptr, &max_conf, nullptr, &conf_indx);

            //If the probability is greater than or equal to 0.4, append all the data into the vectors 
            if (max_conf >= 0.4) {
                float* row_data = access_row.ptr<float>(0);
                float obj_center_x = row_data[0];
                float obj_center_y = row_data[1];
                float width = row_data[2];
                float height = row_data[3];
                int id = conf_indx.x;

                cv::Rect bounds = cv::Rect(obj_center_x, obj_center_y, width, height);
                cv::Rect final_bounds = blob_param.blobRectToImageRect(bounds, cv::Size(frame_cpy.cols, frame_cpy.rows));

                bounding_box.push_back(final_bounds);
                confidence.push_back(max_conf);
                detected_object.push_back(object_classes[id]);
            }

        }

        //Use NMSBoxes method to filter out the overlapping bounding boxes
        cv::dnn::NMSBoxes(bounding_box, confidence, 0.4, 0.45, indices);

        cv::Rect n;
        // float area = 0;
        std::string object;
        for (int i : indices) {
            n = bounding_box[i];
            // area = bounding_box[i].width * bounding_box[i].height;
            object = detected_object[i];

            time_t timestamp;
            std::time(&timestamp);

            final_result.push_back(detection_result(cv::Point((n.x + n.width / 2) - center_x, center_y - (n.y + n.height / 2)), confidence[i], object, timestamp));
        }
    }

    return final_result;
}