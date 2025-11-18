#!/bin/bash
#
# Training script for ODE initialization on 8x H200 GPUs
# 
# Dataset: 1,537 video samples from mixkit_ode_lmdb
# Total batch size: 16 (2 per GPU x 8 GPUs)
# Expected training time: ~2000 iterations
#

set -e  # Exit on error

# Configuration
NUM_GPUS=8
CONFIG_PATH="configs/wan_causal_ode.yaml"
LOGDIR="/root/karthik/ode_checkpoints"
WANDB_SAVE_DIR="/root/karthik/wandb_logs"

# Activate conda environment
echo "Activating conda environment: self_forcing_cuda13"
eval "$(conda shell.bash hook)"
conda activate self_forcing_cuda13

# Set CUDA environment variables for H200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create directories
mkdir -p $LOGDIR
mkdir -p $WANDB_SAVE_DIR

echo "=============================================="
echo "Starting ODE Training"
echo "=============================================="
echo "Config: $CONFIG_PATH"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size per GPU: 2"
echo "Total batch size: 16"
echo "Log directory: $LOGDIR"
echo "Wandb save directory: $WANDB_SAVE_DIR"
echo "Max iterations: 1000"
echo "Checkpoints saved every: 200 iterations"
echo "=============================================="
echo ""

# Change to SiFRiA directory
cd /root/karthik/SiFRiA

# Run training with torchrun for distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    train.py \
    --config_path $CONFIG_PATH \
    --logdir $LOGDIR \
    --wandb-save-dir $WANDB_SAVE_DIR

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
echo "Checkpoints saved to: $LOGDIR"
echo "Wandb logs saved to: $WANDB_SAVE_DIR"
echo "View results at: https://wandb.ai/lnm-/self_forcing_ode_2.1_1.3b"
echo "=============================================="

