#!/bin/bash

# Script to generate videos for all DMD checkpoints

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

PROMPT_FILE="${1:-${REPO_DIR}/prompts/gold.txt}"
NUM_FRAMES="${2:-21}"
SEED="${3:-42}"

echo "Generating videos for DMD checkpoints"
echo "Prompt file: $PROMPT_FILE"
echo "Number of frames: $NUM_FRAMES"
echo "Seed: $SEED"
echo ""

conda run -n "${CONDA_ENV}" python generate_dmd_videos.py \
    --checkpoint_dir "${DMD_CKPT_DIR}" \
    --config configs/self_forcing_dmd_modified.yaml \
    --prompt_file "$PROMPT_FILE" \
    --output_folder videos/dmd_gold \
    --num_output_frames $NUM_FRAMES \
    --seed $SEED \
    --use_ema

echo ""
echo "Videos saved in: videos/compare_dmd/"
echo "To view:"
echo "  ls -lh videos/compare_dmd/"

# python generate_dmd_videos.py \
#     --checkpoint_dir /data/karthik_data/expt_2_1_1_3b_dmd_14b_teacher_batch_2_GA_4 \
#     --config configs/self_forcing_dmd_modified.yaml \
#     --prompt_file "/home/karthik/SiFRiA/prompts/test_prompts_off_dset.txt" \
#     --output_folder videos/dmd_t_14_b2_GA4_LRC_1_off_dset \
#     --num_output_frames 21 \
#     --seed 0 \
#     --use_ema