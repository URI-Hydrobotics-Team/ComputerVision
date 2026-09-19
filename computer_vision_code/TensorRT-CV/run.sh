#!/bin/bash

cd make

cmake ..

make

# If pass no argument, it automatically run the program using the naive distance estimation method
# Possible arguments are:
# build: building model, example: ./run.sh build onnx_path dest_path
# naive: run naive distance estimation, example: ./run.sh naive
# Pnp: run PnP distance estimation, example: ./run.sh PnP
# test: run test
    # for run test image: ./run.sh test image model_path
    # (video testing has not implemented yet)
    # for run test video: ./run.sh test video model_path
./Detection "$@"