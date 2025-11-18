#!/bin/bash

# Script to compare ODE checkpoints across training iterations

PROMPT_FILE="${1:-prompts/MovieGenVideoBench_extended_diagnostic_24.txt}"
NUM_FRAMES="${2:-21}"
SEED="${3:-42}"

echo "=============================================="
echo "Comparing ODE Checkpoints Across Iterations"
echo "=============================================="
echo "Prompt file: $PROMPT_FILE"
echo "Number of frames: $NUM_FRAMES"
echo "Seed: $SEED"
echo "Checkpoint directory: /root/karthik/ode_checkpoints"
echo "Output directory: videos/compare_ode"
echo "=============================================="
echo ""

python compare_ode_iter.py \
    --ode_config configs/wan_causal_ode.yaml \
    --checkpoint_dir /root/karthik/ode_checkpoints \
    --prompt_file "$PROMPT_FILE" \
    --output_folder videos/compare_ode \
    --num_output_frames $NUM_FRAMES \
    --seed $SEED

echo ""
echo "=============================================="
echo "Videos saved to: videos/compare_ode/"
echo "To view all videos:"
ls -lh videos/compare_ode/*.mp4 2>/dev/null || echo "  (No videos found)"
echo "=============================================="

