#!/bin/bash
#
# Generate videos from ODE checkpoints for visual inspection
#

set -e

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

# Configuration
CHECKPOINT_DIR="${ODE_CKPT_DIR}"
CONFIG="configs/wan_causal_ode_modified.yaml"
PROMPT_FILE="${1:-prompts/test_prompts_dset.txt}"
# PROMPT_FILE="${1:-prompts/test_prompts_off_dset.txt}"
OUTPUT_FOLDER="videos/ode_inspection_dset"
# OUTPUT_FOLDER="videos/ode_inspection_off_dset"
NUM_PROMPTS="${2:-1}"
SEED="${3:-0}"

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=============================================="
echo "Generating ODE Videos"
echo "=============================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Config: $CONFIG"
echo "Prompt file: $PROMPT_FILE"
echo "Number of prompts: $NUM_PROMPTS"
echo "Output folder: $OUTPUT_FOLDER"
echo "Seed: $SEED"
echo "=============================================="
echo ""

# Change to SiFRiA directory
cd "${REPO_DIR}"

# Run video generation
python generate_ode_videos.py \
    --config "$CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --output_folder "$OUTPUT_FOLDER" \
    --num_prompts $NUM_PROMPTS \
    --seed $SEED

echo ""
echo "=============================================="
echo "Video generation completed!"
echo "=============================================="
echo "Videos saved to: $OUTPUT_FOLDER"
echo "=============================================="

