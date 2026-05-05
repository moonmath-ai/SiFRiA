#!/bin/bash
#
# Training script for ODE initialization on 8x H200 GPUs
# 
# Dataset:/prompts/vidprom_filtered_extended_16k.txt
# Total batch size: 64 (8 per GPU x 8 GPUs)


set -e  # Exit on error

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

# Configuration
NUM_GPUS=8
CONFIG_PATH="configs/wan_causal_ode_modified.yaml"
LOGDIR="${ODE_CKPT_DIR}"

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

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
echo "Batch size per GPU: 8"
echo "Total batch size: 64"
echo "Log directory: $LOGDIR"
echo "Wandb save directory: $WANDB_SAVE_DIR"
echo "Max iterations: 3000"
echo "Checkpoints saved every: 250 iterations"
echo "=============================================="
echo ""

# Change to SiFRiA directory
cd "${REPO_DIR}"

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

