#!/bin/bash
# ==============================================================================
# CHTC training entrypoint for OpenPI
# Called by HTCondor with:  run_train.sh <config_name> <exp_name>
#
# This script:
#   1. Extracts the staged LeRobot dataset
#   2. Computes normalization statistics (if not cached)
#   3. Runs training
#   4. Packages checkpoints for transfer back to /staging
# ==============================================================================

set -euo pipefail

CONFIG_NAME="${1:?Usage: run_train.sh <config_name> <exp_name>}"
EXP_NAME="${2:?Usage: run_train.sh <config_name> <exp_name>}"

echo "============================================"
echo "OpenPI CHTC Training Job"
echo "  Config:     $CONFIG_NAME"
echo "  Experiment: $EXP_NAME"
echo "  Host:       $(hostname)"
echo "  GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Date:       $(date)"
echo "============================================"

# --- GPU diagnostics ---
nvidia-smi || echo "WARNING: nvidia-smi not available"

# --- Setup cache dirs on the scratch space ---
export HF_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OPENPI_DATA_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$OPENPI_DATA_HOME"

# --- Extract the dataset ---
# The submit file transfers collab_dataset.tar.gz into the working dir.
# This tarball should contain the LeRobot dataset at the path that matches
# your config's repo_id (e.g., local/collab/).
DATASET_TAR="collab_dataset.tar.gz"
if [ -f "$DATASET_TAR" ]; then
    echo "Extracting dataset from $DATASET_TAR..."
    # Extract into HF_LEROBOT_HOME so LeRobot can find it
    export HF_LEROBOT_HOME="${_CONDOR_SCRATCH_DIR:-.}/lerobot_data"
    mkdir -p "$HF_LEROBOT_HOME"
    tar -xzf "$DATASET_TAR" -C "$HF_LEROBOT_HOME"
    echo "Dataset extracted to $HF_LEROBOT_HOME"
    ls -la "$HF_LEROBOT_HOME"
    # Free up scratch disk
    rm -f "$DATASET_TAR"
else
    echo "WARNING: $DATASET_TAR not found, assuming dataset is already available"
fi

# --- Compute normalization stats (idempotent, skips if already present) ---
echo "Computing normalization statistics..."
uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

# --- Run training ---
echo "Starting training..."
uv run scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite

# --- Package checkpoints for output transfer ---
echo "Packaging checkpoints..."
CKPT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"
if [ -d "$CKPT_DIR" ]; then
    tar -czf checkpoints_out.tar.gz -C checkpoints "${CONFIG_NAME}/${EXP_NAME}"
    echo "Checkpoints packaged: $(du -sh checkpoints_out.tar.gz | cut -f1)"
else
    echo "WARNING: Checkpoint directory $CKPT_DIR not found"
    # Create an empty tarball so transfer_output_files doesn't fail
    tar -czf checkpoints_out.tar.gz --files-from /dev/null
fi

echo "============================================"
echo "Training complete at $(date)"
echo "============================================"
