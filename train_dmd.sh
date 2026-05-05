#!/bin/bash
#
# Training script for DMD (Distribution Matching Distillation) on 7x H200 GPUs (All available GPUs)
# 
# Starting from pretrained ODE checkpoint:/data/karthik_data/expt_2_1_1_3b_ode/checkpoint_model_002500
# Saving checkpoints to: /data/karthik_data/expt_2_1_1_3b_dmd
# Checkpoint frequency: Every 250 iterations
#

set -e  # Exit on error

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

# Configuration
NUM_GPUS=7
CONFIG_PATH="configs/self_forcing_dmd_modified.yaml"
LOGDIR="${DMD_CKPT_DIR}"

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Print configuration
echo "=============================================="
echo "Starting DMD Training"
echo "=============================================="
echo "Config: ${CONFIG_PATH}"
echo "Number of GPUs: ${NUM_GPUS} (All available GPUs: 0,1,2,3,4,5,6)"
echo "Batch size per GPU: 2"
echo "Gradient accumulation steps: 8"
echo "Effective batch size: 112 (2 per GPU × 7 GPUs × 8 accum steps)"
echo "Guidance scale: 3.0 (teacher critic)"
echo "Gradient clipping: max_grad_norm=1.0 (generator & critic)"
echo "Log directory: ${LOGDIR}"
echo "Wandb save directory: ${WANDB_SAVE_DIR}"
echo "Checkpoint frequency: Every 250 iterations"
echo "Starting from pretrained ODE checkpoint:/data/karthik_data/expt_2_1_1_3b_ode/checkpoint_model_002500"
echo "=============================================="
echo ""

# Create necessary directories
mkdir -p ${LOGDIR}
mkdir -p ${WANDB_SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
# Run distributed training
torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    train.py \
    --config_path ${CONFIG_PATH} \
    --logdir ${LOGDIR} \
    --wandb-save-dir ${WANDB_SAVE_DIR}

echo ""
echo "=============================================="
echo "DMD Training completed!"
echo "Checkpoints saved in: ${LOGDIR}"
echo "=============================================="


