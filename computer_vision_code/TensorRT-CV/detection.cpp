#include "detection.h"

detection::detection(ILogger* t, std::vector<std::string> object_classes, int input_size, int output_channel, int output_dim1, int output_dim2, int model_type){
    count = 0;
    time = 0.0;
    time1 = 0.0;
    count1 = 0;
    logger = t;
    this->input_size = input_size;
    this->output_channel = output_channel;
    this->output_dim1 = output_dim1;
    this->output_dim2 = output_dim2;
    this->object_classes = object_classes;
    
    // Create the cuda streams
    cudaStreamCreate(&stream);
    cudaStreamCreate(&stream2);

    // if fp16 model is used, do float16 specific operations
    if(model_type == 16){
        // change data size, it is now 2 bytes
        data_size = sizeof(float) / 2;
        // allocate gpu memory for the input tensor to prevent reallocating it everytime during inference
        cudaMalloc((void **)&input_ptr_half, 3 * input_size * input_size * data_size);
        // allocate gpu memory for the output tensor to prevent reallocating it everytime during inference
        cudaMalloc((void **)&output_ptr_half, output_channel * output_dim1 * output_dim2 * data_size);
        // since we will need a new memory to store the float32 value converted from the float16 value later, we will allocate it now
        cudaMalloc((void **)&output_ptr_final, output_channel * output_dim1 * output_dim2 * sizeof(float));

        static cv::cuda::HostMem pinned_host_mem(1 * 3 * 640 * 640 * 2, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
        some = pinned_host_mem.createMatHeader();
    }

    else{
        // data size here is 4 bytes for float32
        data_size = sizeof(float);
        num_output = 0;
        // allocate gpu memory for input and output tensor
        cudaMalloc((void **)&input_ptr, 3 * input_size * input_size * data_size);
        cudaMalloc((void **)&boxes, 30 * 4 * data_size);
        cudaMalloc((void **)&conf, 30 * data_size);
        cudaMalloc((void **)&classes, 30 * sizeof(int));
        cudaMalloc((void **)&index_count, sizeof(int));
        cudaMalloc((void **)&prob, data_size);
        cudaMalloc((void **)&IoU, data_size);

        model_IoU = 0.5;
        model_prob = 0.2;

        cudaMemcpy(prob, &model_prob, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(IoU, &model_IoU, sizeof(float), cudaMemcpyHostToDevice);

        final_classes.resize(30);
        final_boxes.resize(30 * 4);
        final_conf.resize(30);
        cv::cuda::HostMem pinned_host_mem(1 * 3 * 640 * 640 * sizeof(float), cv::cuda::HostMem::AllocType::PAGE_LOCKED);
        some = pinned_host_mem.createMatHeader();
    }
}

void detection::convert_onnx(std::string onnx_model_path){
    // create builder
    builder = createInferBuilder(*logger);
    // create network
    network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
    // create parser 
    parser = nvonnxparser::createParser(*network, *logger);

    // use parser to parse the onnx model
    if(!parser->parseFromFile(onnx_model_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING))){
        std::cout << "Parser failed\n";
    }
    std::cout << "Model parsing successful\n";

    // gets the total output, will slice it later
    ITensor* combined_output = network->getOutput(0);
    std::cout << "Original model output obtained\n";

    // add shuffle layer to transpose the original output
    IShuffleLayer* shuffle_layer = network->addShuffle(*combined_output);
    
    shuffle_layer->setFirstTranspose(Permutation{0, 2, 1});
    ITensor* transposed = shuffle_layer->getOutput(0);
    std::cout << "Model output transposed successfully\n";

    // slice the tensort, start at index 4, which is where class probability begins
    auto confidence = network->addSlice(*transposed, Dims3{0, 0, 4}, Dims3{1, output_dim2, output_dim1 - 4}, Dims3{1, 1, 1});
    ITensor* scores = confidence->getOutput(0);
    std::cout << "All class confidence obtained\n";
    
    // slice layer to get the bounding boxes from the transposed model output
    auto bounding_boxes = network->addSlice(*transposed, Dims3{0, 0, 0}, Dims3{1, output_dim2, 4}, Dims3{1, 1, 1});
    ITensor* box_init = bounding_boxes->getOutput(0);
    std::cout << "All box obtained\n";

    int num = 10;
    Weights num_per_class{DataType::kINT32, &num, 1};
    ITensor* max_output_per_class = network->addConstant(Dims{}, num_per_class)->getOutput(0);

    // add nms layer into the model
    INMSLayer* nms_layer = network->addNMS(*box_init, *scores, *max_output_per_class);
    std::cout << "NMS layer added to model\n";

    // uncomment this if a IoU threshold will be constant throughout and will not need to change later
    // it will be a little faster than dynamic threshold
    // float IoU_threshold = 0.4;
    // Weights IoU{DataType::kFLOAT, &IoU_threshold, 1};
    // ITensor* IoU_tensor = network->addConstant(Dims{}, IoU)->getOutput(0);

    // Add dynamic IoU threshold input node to allow dynamic change threshold without rebuilding engine
    ITensor* IoU_threshold = network->addInput("IoU", DataType::kFLOAT, Dims{});

    // float score_threshold = 0.4;
    // Weights score{DataType::kFLOAT, &score_threshold, 1};
    // ITensor* score_tensor = network->addConstant(Dims{}, score)->getOutput(0);

    // add dynamic confidence threshold for the nms
    ITensor* score_tensor = network->addInput("confidence", DataType::kFLOAT, Dims{});
    
    // set the nms layer attribute
    nms_layer->setTopKBoxLimit(30);
    nms_layer->setInput(3, *IoU_threshold);
    nms_layer->setInput(4, *score_tensor);
    nms_layer->setBoundingBoxFormat(BoundingBoxFormat::kCENTER_SIZES);

    // remove the original output node
    network->unmarkOutput(*combined_output);

    // get the box indicies and count of amount of good box
    ITensor* final_bounding_boxes = nms_layer->getOutput(0);
    ITensor* box_counts = nms_layer->getOutput(1);
    std::cout << "Valid coordinates and counts obtained\n";

    // change it such that the nms output indicies are always 30 elements
    // by filling in values
    ISliceLayer* correct_index_layer = network->addSlice(*final_bounding_boxes, Dims2{0, 0}, Dims2{30, 3}, Dims2{1, 1});
    correct_index_layer->setMode(SampleMode::kFILL);
    ITensor* corrected_index_shape = correct_index_layer->getOutput(0);

    // get class indicies
    ISliceLayer* class_slice_layer = network->addSlice(*corrected_index_shape, Dims2{0, 1}, Dims2{30, 1}, Dims2{1, 1});
    class_slice_layer->setMode(SampleMode::kFILL);
    ITensor* valid_class = class_slice_layer->getOutput(0);

    // get box indicies
    ISliceLayer* index_slice_layer = network->addSlice(*corrected_index_shape, Dims2{0, 2}, Dims2{30, 1}, Dims2{1, 1});
    index_slice_layer->setMode(SampleMode::kFILL);
    ITensor* valid_boxes = index_slice_layer->getOutput(0);

    // shuffle layer to reshape the box indicies to 1D to be used in the gather layer
    IShuffleLayer* reshape_index_layer = network->addShuffle(*valid_boxes);
    reshape_index_layer->setReshapeDimensions(Dims{1, {30}});
    ITensor* valid_index = reshape_index_layer->getOutput(0);

    // make sure all the indicies in box_indicies are at least 0
    int zero = 0;
    Weights zeros{DataType::kINT32, &zero, 1};
    IConstantLayer* zero_constant_layer = network->addConstant(Dims{1, {1}}, zeros);
    ITensor* zero_output = zero_constant_layer->getOutput(0);
    IElementWiseLayer* clap_layer = network->addElementWise(*valid_index, *zero_output, ElementWiseOperation::kMAX);
    ITensor* clamp_output = clap_layer->getOutput(0);

    // get the valid boxes
    IGatherLayer* valid_box_layer = network->addGatherV2(*box_init, *clamp_output, GatherMode::kDEFAULT);
    valid_box_layer->setGatherAxis(1);
    ITensor* valid_box = valid_box_layer->getOutput(0);
    IShuffleLayer* valid_box_final_layer = network->addShuffle(*valid_box);
    valid_box_final_layer->setReshapeDimensions(Dims2{30, 4});
    ITensor* box_final = valid_box_final_layer->getOutput(0);

    auto box = box_final->getDimensions();
    std::cout << "Box dim: " << box.nbDims << "\n";
    for(int i = 0; i < box.nbDims; i++){
        std::cout << box.d[i] << " ";
    }

    std::cout << "\n";

    // get confidence
    IGatherLayer* valid_confidence_layer = network->addGatherV2(*scores, *clamp_output, GatherMode::kDEFAULT);
    valid_confidence_layer->setGatherAxis(1);
    ITensor* valid_confidence = valid_confidence_layer->getOutput(0);
    IReduceLayer* valid_conf_reduce = network->addReduce(*valid_confidence, ReduceOperation::kMAX, 1 << 2, 1);
    ITensor* final_reduce = valid_conf_reduce->getOutput(0);
    IShuffleLayer* valid_conf_final_layer = network->addShuffle(*final_reduce);
    valid_conf_final_layer->setReshapeDimensions(Dims2{30, 1});
    ITensor* conf_final = valid_conf_final_layer->getOutput(0);

    auto conf = conf_final->getDimensions();
    std::cout << "conf dim: " << conf.nbDims << "\n";
    for(int i = 0; i < conf.nbDims; i++){
        std::cout << conf.d[i] << " ";
    }

    std::cout << "\n";

    // set names for the outputs 
    conf_final->setName("confidences");
    box_final->setName("box");
    box_counts->setName("counts");
    valid_class->setName("class");

    // add the new outputs to the model
    network->markOutput(*box_final);
    network->markOutput(*conf_final);
    network->markOutput(*valid_class);
    network->markOutput(*box_counts);

    std::cout << "Model conversion successful\n";
}

void detection::build_engine(std::string onnx_file, std::string filename){
    convert_onnx(onnx_file);
    // create config object, used to set different attributes for the engine building
    config = builder->createBuilderConfig();
    // Allocate 1GB of memory for TensorRT's engine building process
    // 1 << num is a bvitwise operation, essentially meaning 2 to the power of num
    // for example 1U << 30 means 2 ^ 30 bytes
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);
    // set the model to optimize for FP16 model
    // config->setFlag(BuilderFlag::kFP16);

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("IoU", OptProfileSelector::kMAX, Dims{});
    profile->setDimensions("IoU", OptProfileSelector::kOPT, Dims{});
    profile->setDimensions("IoU", OptProfileSelector::kMIN, Dims{});
    profile->setDimensions("confidence", OptProfileSelector::kMAX, Dims{});
    profile->setDimensions("confidence", OptProfileSelector::kOPT, Dims{});
    profile->setDimensions("confidence", OptProfileSelector::kMIN, Dims{});

    if(profile->isValid()) {
        config->addOptimizationProfile(profile);
    }
    else {
        std::cout << "Profile incomplete\n";
    }

    // build the engine model
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

    // no longer need these variables, we delete them
    delete parser;
    delete network;
    delete config;
    delete builder;

    // write the engine model into the filename file in binary format
    std::ofstream disk_writer(filename, std::ios::binary);
    disk_writer.write((const char*)(serializedModel->data()), serializedModel->size());
    disk_writer.close();

    // delete the engine model object
    delete serializedModel;
}

std::vector<char> detection::readModelFromFile(std::string engine_path){
    // read the engine file in binary format
    std::ifstream input(engine_path, std::ios::binary);

    if(!input){
        throw std::runtime_error("Failed to read file");
    }

    // move read position to the end of file
    input.seekg(0, std::ios::end);
    std::streampos file_size;

    // set the file_size to the read position, which is now the end of the file
    file_size = input.tellg();

    // return the read position back to the beginning 
    input.seekg(0, std::ios::beg);

    std::vector<char> model_data(file_size);

    // store the model data into the vector
    input.read(model_data.data(), file_size);

    return model_data;
}

void detection::load_model(std::string engine_path){
    // create a runtime
    runtime = createInferRuntime(*logger);
    
    // get model data 
    std::vector<char> modelData = readModelFromFile(engine_path);

    // deserialize the engine data
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    context = engine->createExecutionContext();
}

cv::Mat detection::preprocess(cv::Mat frame){
    cv::Mat blob;
    // cpu_frame = frame;

    // converts frame into NCHW format
    // TODO: Implement a CUDA kernel to do this instead of using the blobFromImage function
	// cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(input_size, input_size), cv::Scalar(), true, false);

    cv::dnn::Image2bBlobParams blob_param;
    
    blob_param.datalayout = cv::dnn::DNN_LAYOUT_NCHW;
    blob_param.ddepth = CV_32F;
    blob_param.mean = cv::Scalar();
    blob_param.paddingmode = cv::dnn::DNN_PMODE_LETTERBOX;
    blob_param.scalefactor = 1.0 / 255.0;
    blob_param.size = cv::Size(640, 640);
    blob_param.swapRB = true;

    cv::Mat blob = cv::dnn::blobFromImageWithParams(frame, blob_param);

    // std::cout << blob.isContinuous() << "\n";
    return blob;
}

void detection::inference(cv::Mat frame){
    auto start = std::chrono::high_resolution_clock::now();

    x_scale = frame.cols / float(input_size);
    y_scale = frame.rows / float(input_size);
    center_x = frame.cols / 2.0;
    center_y = frame.rows / 2.0;

    // preprocess the frame before inference
    // auto start0 = std::chrono::high_resolution_clock::now();
    input = preprocess(frame);
    // auto stop0 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration0 = stop0 - start0;
    // std::cout << "pre: " << duration0.count() << "\n";

    // auto start1 = std::chrono::high_resolution_clock::now();
    // if the model is a fp16 model, convert frame to FP16 data type
    if(data_size == 2){
        input.convertTo(some, CV_16F);
        cudaMemcpyAsync(input_ptr_half, some.data, some.total() * some.elemSize(), cudaMemcpyHostToDevice, stream);
        context->setTensorAddress("images", input_ptr_half);
        context->setTensorAddress("output0", output_ptr_half);
    }
    else{
        some = input;
        // load frame data into gpu
        cudaMemcpyAsync(input_ptr, some.data, some.total() * some.elemSize(), cudaMemcpyHostToDevice, stream);
        context->setTensorAddress("images", input_ptr);
        context->setTensorAddress("confidence", prob);
        context->setTensorAddress("IoU", IoU);
        context->setTensorAddress("confidences", conf);
        context->setTensorAddress("box", boxes);
        context->setTensorAddress("counts", index_count);
        context->setTensorAddress("class", classes);
    }

    // auto memcpy_stop = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> cpy_duration = memcpy_stop - start1;
    // std::cout << "cpy time: " << cpy_duration.count() << "\n";
    
    // some assertion to make sure things go correctly
    assert(some.data != nullptr);
    assert(3 * input_size * input_size * data_size == some.total() * some.elemSize());

    assert(context != nullptr);
    assert(stream);
    assert(output_ptr != nullptr);

    // perform inference
    context->enqueueV3(stream);
    
    // wait for the stream to complete
    
    // auto stop1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration1 = stop1 - start1;
    // std::cout << "mid: " << duration1.count() << "\n";

    // post process the result
    // auto start2 = std::chrono::high_resolution_clock::now();
    postprocess();
    // auto stop2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration2 = stop2 - start2;
    // std::cout << "post: " << duration2.count() << "\n";

    // auto stop = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = stop - start;

    // count1++;

    // if(count1 > 9){
    //     count++;
    //     time += duration.count();
    //     std::cout << "detection time: " << time / count << "\n";
    // }
}


void detection::postprocess() {
    cudaError err1 = cudaMemcpyAsync(final_classes.data(), classes, 30 * sizeof(int), cudaMemcpyDeviceToHost, stream);

    if(err1 != cudaSuccess) {
        std::cout << "cuda 1 failed";
    }

    cudaError err2 = cudaMemcpyAsync(final_conf.data(), conf, 30 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if(err2 != cudaSuccess) {
        std::cout << "cuda 2 failed";
    }

    cudaError err3 = cudaMemcpyAsync(final_boxes.data(), boxes, 30 * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if(err3 != cudaSuccess) {
        std::cout << "cuda 3 failed";
    }

    cudaError err4 = cudaMemcpyAsync(&num_output, index_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if(err4 != cudaSuccess) {
        std::cout << "cuda 4 failed";
    }

    // std::cout << "count: " << num_output << "\n";
    cudaStreamSynchronize(stream);

    for(int i = 0; i < num_output; i++) {
        float x = final_boxes[i * 4], y = final_boxes[i * 4 + 1], width = final_boxes[i * 4 + 2], height = final_boxes[i * 4 + 3];
        cv::Rect2d bounds = cv::Rect2d((x - (width / 2)) * x_scale, (y - (height / 2)) * y_scale, width * x_scale, height * y_scale);
        
        max_conf = final_conf[i];

        int id = final_classes[i];
        std::string object = object_classes[id];

        std::cout << "Object: " << object << " | " << "X offset: " << (bounds.x + bounds.width / 2) - center_x << " | " << "Y offset: " << center_y - (bounds.y + bounds.height / 2) << "\n";

        // cv::rectangle(cpu_frame, bounds, cv::Scalar(0, 0, 0), 3); // Draw the bounding box
        // std::string info = object + ": ";
        // info += std::to_string(max_conf);
        // cv::putText(cpu_frame, info, cv::Point(bounds.x, bounds.y), cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 255, 255)); // Put text 
        
        // cv::imshow("Pic", cpu_frame);
        // cv::waitKey(10);
    }
}

void detection::dec(){
    cudaStreamDestroy(stream);
    cudaFree(input_ptr);
    cudaFree(output_ptr);
}