#!/bin/bash

if [ -z "${CUDA_MPS_PIPE_DIRECTORY}" ] ; then
    echo "was expecting CUDA_MPS_PIPE_DIRECTORY to be set" >&2
else
    echo quit | nvidia-cuda-mps-control
    rm -rf "/data/vladf/mps$CUDA_VISIBLE_DEVICES"
fi
