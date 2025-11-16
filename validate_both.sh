#!/bin/bash
# Script to validate and compare both ODE and DMD checkpoints (Single GPU only), multi gpu: issue with encoder/decoder
#
# Usage:
#   ./validate_both.sh                    # Use default batch_size=4
#   ./validate_both.sh 32                 # Use batch_size=32 (tested takes up to 115 gb)

BATCH_SIZE=${1:-4}   # Default to 4 if not provided

python validate_both_checkpoints.py \
    --ode_config configs/ode_validation.yaml \
    --dmd_config configs/dmd_validation.yaml \
    --ode_checkpoint checkpoints/ode_init.pt \
    --dmd_checkpoint checkpoints/self_forcing_dmd.pt \
    --data_path /root/karthik/lmdb/mixkit_ode_lmdb \
    --batch_size $BATCH_SIZE