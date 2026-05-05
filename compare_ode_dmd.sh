#!/bin/bash

# Script to compare ODE vs DMD checkpoint video generation

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

PROMPT_FILE="${1:-${REPO_DIR}/prompts/test_prompts.txt}"
NUM_FRAMES="${2:-21}"
SEED="${3:-0}"

python compare_ode_dmd.py \
    --ode_config configs/wan_causal_ode.yaml \
    --dmd_config configs/self_forcing_dmd.yaml \
    --ode_checkpoint /data/karthik_data/self_forcing_checkpoints/ode_init.pt \
    --dmd_checkpoint /data/karthik_data/self_forcing_checkpoints/self_forcing_dmd.pt \
    --prompt_file "$PROMPT_FILE" \
    --output_folder videos/ode_vs_dmd \
    --num_output_frames $NUM_FRAMES \
    --seed $SEED \
    --use_ema

echo ""
echo "To view the videos:"
echo "  ODE: videos/ode_vs_dmd/ode_generation.mp4"
echo "  DMD: videos/ode_vs_dmd/dmd_generation.mp4"

