#!/bin/bash

# Script to compare ODE vs DMD checkpoint video generation

PROMPT_FILE="${1:-prompts/MovieGenVideoBench_extended_diagnostic_24.txt}"
NUM_FRAMES="${2:-21}"
SEED="${3:-42}"

python compare_ode_dmd.py \
    --ode_config configs/wan_causal_ode.yaml \
    --dmd_config configs/self_forcing_dmd.yaml \
    --ode_checkpoint checkpoints/ode_init.pt \
    --dmd_checkpoint checkpoints/self_forcing_dmd.pt \
    --prompt_file "$PROMPT_FILE" \
    --output_folder videos/ode_vs_dmd \
    --num_output_frames $NUM_FRAMES \
    --seed $SEED \
    --use_ema

echo ""
echo "To view the videos:"
echo "  ODE: videos/ode_vs_dmd/ode_generation.mp4"
echo "  DMD: videos/ode_vs_dmd/dmd_generation.mp4"

