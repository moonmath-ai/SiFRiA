#!/bin/bash

# Simple example script to run batched T2V inference on 4 example prompts
# This demonstrates how to generate 4 videos simultaneously
#
# Usage: ./run_example_batch.sh [batch_size]
#   batch_size: Number of prompts to process simultaneously (default: 1)

# Set default batch size if not provided
BATCH_SIZE=${1:-1}

echo "Running batched T2V inference on 4 example prompts with batch_size=${BATCH_SIZE}..."

python inference_batched.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/example_batch.txt \
    --output_folder videos/example_batch_output \
    --batch_size ${BATCH_SIZE} \
    --use_ema \
    --save_with_index \

echo ""
echo "Done! Check videos/example_batch_output/ for generated videos:"
echo "  - 0-0_ema.mp4 (Lion in savanna)"
echo "  - 1-0_ema.mp4 (Futuristic cityscape)"
echo "  - 2-0_ema.mp4 (Ocean beach)"
echo "  - 3-0_ema.mp4 (Steam locomotive)"

