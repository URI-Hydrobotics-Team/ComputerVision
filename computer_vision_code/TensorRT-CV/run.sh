#!/bin/bash

cd make

cmake ..

make

# If pass no argument, it automatically run the program using the naive distance estimation method
# Possible arguments are:
# build: building model, example: ./run.sh build onnx_path dest_path
# naive: run naive distance estimation, example: ./run.sh naive
# Pnp: run PnP distance estimation, example: ./run.sh PnP
./Detection "$@"