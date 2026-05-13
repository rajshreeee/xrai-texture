#!/bin/bash

SESSION="rand-vgg11unet-cbis"
CONDA_ENV="/ediss_data/ediss2/xai-texture/envs/xai-new"
SCRIPT_DIR="/ediss_data/ediss2/xai-texture/src/models/rand-xai-vgg11unet-cbis-datasize"

tmux new-session -d -s "$SESSION" \
    "source $(conda info --base)/etc/profile.d/conda.sh && \
     conda activate $CONDA_ENV && \
     cd $SCRIPT_DIR && \
     python unet.py; exec bash"

echo "Started tmux session '$SESSION' running unet.py"