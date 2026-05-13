#!/bin/bash

SESSION="kr-vgg16unet-tompei-datasize"
CONDA_ENV="/ediss_data/ediss2/xai-texture/envs/xai-new"
PYTHON="/ediss_data/ediss2/xai-texture/envs/xai-new/bin/python"
SCRIPT_DIR="/ediss_data/ediss2/xai-texture/src/models/kr-xai-vgg16unet-tompei-datasize"

tmux new-session -d -s "$SESSION" \
    "source $(conda info --base)/etc/profile.d/conda.sh && \
     conda activate $CONDA_ENV && \
     cd $SCRIPT_DIR && \
     python unet.py; exec bash"

echo "Started tmux session '$SESSION' running unet.py"
