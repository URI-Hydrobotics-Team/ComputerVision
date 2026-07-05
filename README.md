# Computer-Vision
Repository to host the team's computer vision model training and testing.

# TensorRT version of computer vision

## How to run
* `git clone` the repo
* `cd` into the `TensorRT-CV` directory
* Select one of the few modes below:
    1. build model ONNX to TensorRT format: `./run.sh build onnx_model_path tensorrt_model_dest_path`
    2. Run the CV program: `./run.sh`

## How it works
1. **Main program**
    * Get arguments from the run.sh script
    * Load model
    * Repeatly run the following actions
        * OpenCV capture frames
        * Model inference on the frame
        * Postprocess the result and return a vector of type CV_data
        * Append all the detected object data into one string of format `name|confidence|timestamp|pixel x offset|pixel y offset|actual distance in z\nname2|confidence2|timestamp2|pixel x offset2|pixel y offset2|actual distance in z2\n...`
        * Send the new string to AVOE

2. **ONNX model conversion to TensorRT format**
    * Parse ONNX model

    * Get the original output node (originally the last node of the model)

    * Add shuffle layer to the model to do transpose operation on the output received from the previous node to (0, 2, 1) dimension. 
    
        **Example:** (1, 84, 8400) to (1, 8400, 84)

    * The last node of model now is a node containing the transposed result

    * Add slice layer to model and slice the transposed output on 3rd dimension of the output from index 4 to the last index to get the class confidence. 

        **Reason:** Because for example the transposed YOLO model will become (1, 8400, 84). 84 includes x, y, w, h, confidence on object 1, confidence on object 2 ... confidence on object n.

    * Add slice layer to the model to slice the transposed output on 3rd dimension of the output from index 0 to 4 to get the bounding boxes

    * Add NMS layer to the model that takes in the bounding boxes, class confidence, and limit it to get a maximum of 10 possible detection per class

    * Add an input (float) to the model for the IoU score that will be used for the NMS layer

    * Add an input (float) to the model for the confidence that will be used for the NMS layer

    * Set NMS layer parameters
        * Output a maximum of 30 detections
        * IoU score set to the input IoU score
        * Confidence score set to the input confidence score
        * Output bounding box format to (x_center, y_center, width, height)
    
    * Unmark the original YOLO model output (making it no longer the model output node anymore).

    * Obtain selected indices and the amount of indices from NMS layer output. They are output index 0 and 1

        **NMS layer returns the following two things:** 
        
        * **SelectedIndices (`final_bounding_boxes` in the code):** tensor of shape **[NumOutputBoxes, 3]**. According to NVIDIA TensorRT documentation, the columns are **(batchIndex, classIndex, boxIndex)**. .These are indices of the detections that is determine by NMS to be valid, which will be used to map back to the original data to get the actual values.

        * **NumOutputBoxes (`box_counts` in the code):** The amount of boxes return by NMS.

    * Add a constant layer that contains 30 dummy values

    * Add concatenate layer to concatenate the dummy layer data into the final_bounding_boxes to prevent negative row dimensions

    * With the newly concatenated values, only maximum of 30 data is needed (adding 30 dummy value could exceed 30), so add a slice layer to get only the first 30 data, the output of this operation is the final data to process, this is the `corrected_index_shape` variable

    * Add slice layer to get all the classes indices (basically it is the class ID for each detection) from `corrected_index_shape`. Which is the entire column 1 of `corrected_index_shape`

    * Add slice layer to get all the box indices from `corrected_index_shape`. Which is the entire column 2 of `corrected_index_shape`

    * Add shuffle layer to reshape the box indices from 2D to 1D to be used in the gather layer

    * Add gather layer with the original box output (the (1, 8400, 4) data for example) and the NMS box indices to get the valid boxes (setGatherAxis(1) is used because we have the indices for boxes, which is in the 8400 and not 1)

    * Add shuffle layer to reshape the output of gather layer from (1, 30, 4) to (30, 4)

    * With all that, the final box output, the (30, 4) is the `box_final`

    * Add another gather layer to extra valid rows from the `scores` variable, which contains all the class probability for each row of the original detection. The output size should be (1, 30, number of classes)

    * Then add a reduce layer to find the maximum probability of each row, so output of this will be shape (1, 30, 1)

    * Add shuffle layer to reshape (1, 30, 1) reduce layer output to (30, 1)

    * Set a name for all the nodes (which is obtain by using the getOutput function on different layers) that contains the values that the model need to output (box, confidence, classes, amount of box). These names will be needed later for model inference

    * Mark all the nodes that contains the needed value as the new output of the model

    * The model should now be successfully converted to TensorRT format

3. **Computer vision pipeline**

    **Preprocess:**
        * Receives a frame
        * Use OpenCV blobFromImageWithParams function to process the frame based on the Image2BlobParams object's attributes, which includes use NCHW output layout, and resize frame to size (640, 640) using letterbox resizing
        * Return the processed frame

    **Inference:** 
        * Load preprocessed frame into GPU
        * Set GPU memory address for their corresponding input or output tensor.
        * Perform inference
        * The model output will be stored in the GPU memories

    **Postprocess:** 
        * Download the model output from GPU to CPU memory
        * Loop through the outputs into a vector of CV_data type and return result

    **Note:** Async version of the above operations are available for double buffering to increase inference speed.
    
