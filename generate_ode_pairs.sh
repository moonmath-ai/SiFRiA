#!/bin/bash
#
# Script to generate ODE pairs for training
# Uses 8x H200 GPUs with distributed processing
#
# Input: vidprom_filtered_extended_16k.txt (16,000 prompts)
# Output: /data/karthik_data/vidprom_ode_pairs
#

set -e  # Exit on error

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

# Configuration
NUM_GPUS=8
CAPTION_PATH="${CAPTION_PATH:-${REPO_DIR}/prompts/vidprom_filtered_extended_16k.txt}"
OUTPUT_FOLDER="${ODE_PAIRS_DIR}"
GUIDANCE_SCALE=6.0

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set PYTHONPATH to include the SiFRiA directory
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

# Create output directory
mkdir -p $OUTPUT_FOLDER

echo "=============================================="
echo "Starting ODE Pair Generation"
echo "=============================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Caption path: $CAPTION_PATH"
echo "Output folder: $OUTPUT_FOLDER"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "Total prompts: 16,000"
echo "Prompts per GPU: ~2,000"
echo "=============================================="
echo ""

# Change to SiFRiA directory
cd "${REPO_DIR}"

# Run ODE pair generation with torchrun for distributed processing
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    scripts/generate_ode_pairs.py \
    --output_folder $OUTPUT_FOLDER \
    --caption_path $CAPTION_PATH \
    --guidance_scale $GUIDANCE_SCALE

echo ""
echo "=============================================="
echo "ODE Pair Generation completed!"
echo "=============================================="
echo "ODE pairs saved to: $OUTPUT_FOLDER"
echo "Total files generated: $(ls -1 $OUTPUT_FOLDER/*.pt 2>/dev/null | wc -l)"
echo "=============================================="

