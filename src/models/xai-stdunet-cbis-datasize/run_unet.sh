#!/bin/bash

SESSION="unet2"
CONDA_ENV="/ediss_data/ediss2/xai-texture/envs/xai-new"
SCRIPT_DIR="/ediss_data/ediss2/xai-texture/src/models/xai-stdunet-cbis-datasize"

tmux new-session -d -s "$SESSION" \
    "source $(conda info --base)/etc/profile.d/conda.sh && \
     conda activate $CONDA_ENV && \
     cd $SCRIPT_DIR && \
     python unet.py; exec bash"

echo "Started tmux session '$SESSION' running unet.py"
