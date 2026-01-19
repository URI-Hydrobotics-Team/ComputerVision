#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include "helper_functions.cuh"
using namespace nvinfer1;


class detection{
    public:
        // constructor, take in the logger object, input and output size
        // model_type is int either 32 or 16 representing FP16 or FP32 models
        detection(ILogger* t, std::vector<std::string> object_classes, int input_size, int output_channel, int output_dim1, int output_dim2, int model_type);

        // function to parse onnx model file
        void convert_onnx(std::string onnx_model_path);

        // function to build engine file 
        // filename: onnx model file path
        void build_engine(std::string onnx_file, std::string filename);

        // function to load the optimized engine file
        void load_model(std::string engine_path);

        // function to read the engine file 
        std::vector<char> readModelFromFile(std::string engine_path);

        // function to do inference on the frame
        void inference(cv::Mat frame);
        // void inference_async(int buffer_index);

        // function to preprocess the frame, converting it to NCHW format for TensorRT to use
        cv::Mat preprocess(cv::Mat frame);
        // void preprocess_async(int buffer_index, cv::Mat frame);

        // function to do postprocess operation on the output tensor
        void postprocess();
        // void postprocess_async(int buffer_index);

        // function to free all GPU memory and objects
        void dec();

    private:
        // Used to capture errors
        ILogger* logger;
        // model path of the onnx model
        std::string model_path;
        // builder that will be used to build the optimized engine model
        IBuilder* builder;
        // Network is the place where it will store the parsed onnx model, which will later be used to construct engine
        INetworkDefinition* network;
        // parser to parse onnx model
        nvonnxparser::IParser* parser;
        // used to configure settings of the engine model that will be built
        // EX: batch size, memeory allocation
        IBuilderConfig* config;
        // used to deserialize the 
        IRuntime* runtime;
        // this is responsible many operations
        // Ex: perform inference, loading tensor
        IExecutionContext *context;
        // This variable contains the model
        ICudaEngine* engine;
        // CUDA stream used to speed things up and also used for model inference
        cudaStream_t stream;
        cudaStream_t stream2;
        // cudaStream_t stream3;
        // cudaStream_t stream4;

        // These are the pointer used to store input and output tensor values if FP32 model is loaded
        float* input_ptr;        
        float* output_ptr;

        // These are the pointer used to store input and output tensor values if FP16 model is loaded
        __half* input_ptr_half;        
        __half* output_ptr_half;

        // This pointer is used if FP16 model is used
        // It is responsible to get float value converted from float16 values
        float* output_ptr_final;

        // model's input size, for example, 1x84x8400 = 1 * 84 * 8400
        int input_size;

        // These are variables used to get process time
        double time;
        int count;
        double time1;
        int count1;
        int data_size;

        // max confidence score of a specific detection
        double max_conf;

        // output tensor atrributes
        int output_channel;
        int output_dim1;
        int output_dim2;

        // index of the max confidence located in the detection matrix
        cv::Point conf_indx;
        std::vector<std::string> object_classes;

        // values of the center of the frame, and the scale variable used to change data back to the values for the actual image
        float x_scale;
        float y_scale;
        float center_x;
        float center_y;

        // OpenCV matrix
        cv::Mat transposed;
        cv::Mat access_row;
        cv::Mat input;
        cv::Mat some;

        // GPU matrix 
        cv::cuda::GpuMat output_mat;
        cv::cuda::GpuMat transposed_output_mat;
        // cv::cuda::GpuMat access_row;


        // experimenting things
        float* boxes;
        float* conf;
        int* classes;
        int* index_count;
        float* IoU;
        float* prob; 

        float model_prob;
        float model_IoU;

        std::vector<int> final_classes;
        std::vector<float> final_conf;
        std::vector<float> final_boxes;
        int num_output;

        cv::Mat cpu_frame;
        cudaStream_t streams[2];
        
};