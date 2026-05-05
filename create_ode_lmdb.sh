#!/bin/bash
#
# Script to create LMDB database from ODE pairs
#
# Input: /data/karthik_data/vidprom_ode_pairs (ODE pair .pt files)
# Output: /data/karthik_data/lmdb/vidprom_ode_lmdb (LMDB shards)
#

set -e  # Exit on error

# Load environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"

# Configuration
DATA_PATH="${ODE_PAIRS_DIR}"
LMDB_PATH="${LMDB_DIR}"
NUM_SHARDS=16

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Set PYTHONPATH
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

echo "=============================================="
echo "Creating LMDB Database from ODE Pairs"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "LMDB path: $LMDB_PATH"
echo "Number of shards: $NUM_SHARDS"
echo "=============================================="
echo ""

# Check if ODE pairs exist
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Error: ODE pairs directory not found: $DATA_PATH"
    exit 1
fi

# Count number of .pt files
NUM_FILES=$(find "$DATA_PATH" -name "*.pt" 2>/dev/null | wc -l)
echo "Found $NUM_FILES ODE pair files"

if [ "$NUM_FILES" -eq 0 ]; then
    echo "❌ Error: No .pt files found in $DATA_PATH"
    exit 1
fi

# Create LMDB directory
mkdir -p "$LMDB_PATH"

# Change to SiFRiA directory
cd "${REPO_DIR}"

# Run LMDB creation
python scripts/create_lmdb_14b_shards.py \
    --data_path "$DATA_PATH" \
    --lmdb_path "$LMDB_PATH" \
    --num_shards $NUM_SHARDS

echo ""
echo "=============================================="
echo "LMDB Creation completed!"
echo "=============================================="
echo "LMDB saved to: $LMDB_PATH"
echo "Number of shards: $NUM_SHARDS"
echo ""
echo "Shard details:"
du -sh "$LMDB_PATH"/shard_* 2>/dev/null || echo "  (run 'du -sh $LMDB_PATH/shard_*' to see sizes)"
echo "=============================================="

