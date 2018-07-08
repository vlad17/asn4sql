#!/bin/bash
# see HELP string below
_LAUNCH_MPS_HELP="

Launch an MPS server tied to a (set of) specific GPU(s).
Expects the CUDA_VISIBLE_DEVICES to be set and this
file should be sourced.

Here is the expected workflow:

$ export CUDA_VISIBLE_DEVICES=3
$ source launch-mps.sh # auto-unsets CUDA_VISIBLE_DEVICES
$ # ... some cuda-dependent thing
$ disable-mps.sh
"

if [ -z "${CUDA_VISIBLE_DEVICES+x}" ] ; then
    echo "was expecting CUDA_VISIBLE_DEVICES to be set$_LAUNCH_MPS_HELP" >&2
elif [[ "${BASH_SOURCE[0]}" = "${0}" ]] ; then
    echo "was expecting the script to be sourced$_LAUNCH_MPS_HELP" >&2
else
    export MPS_DIR="/data/vladf/mps$CUDA_VISIBLE_DEVICES"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR"
    nvidia-cuda-mps-control -d
    unset CUDA_VISIBLE_DEVICES
fi
