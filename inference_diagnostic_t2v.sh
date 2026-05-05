#!/bin/bash

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh

conda activate self_forcing_cuda13
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# note the prompt data set need to be divisible by 8 (for this test, it runs 24()
# Multi-GPU inference with torchrun (use all 8 GPUs)
torchrun --nproc_per_node=8 --standalone inference_diagnostic.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path /data/karthik_data/self_forcing_checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended_diagnostic_24.txt \
    --use_ema