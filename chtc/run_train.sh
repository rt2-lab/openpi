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
CKPT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"

# Absolute path to Python in the container's pre-built venv.
# Using absolute paths so we never depend on PATH being set correctly
# by Condor, the NVIDIA entrypoint, or the container ENV.
PYTHON=/.venv/bin/python

# Ensure checkpoints_out.tar.gz always exists so Condor output transfer won't hold.
package_on_exit() {
    local rc=$?
    echo "Packaging outputs (exit code: ${rc})..."
    if [ -d "$CKPT_DIR" ]; then
        tar -czf checkpoints_out.tar.gz -C checkpoints "${CONFIG_NAME}/${EXP_NAME}"
        echo "Checkpoints packaged: $(du -sh checkpoints_out.tar.gz | cut -f1)"
    else
        tar -czf checkpoints_out.tar.gz --files-from /dev/null
        echo "WARNING: no checkpoint dir found; wrote empty archive"
    fi
}
trap package_on_exit EXIT

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

# Seed scratch cache with assets pre-downloaded during Docker build
# (e.g., PaliGemma tokenizer) so jobs don't need GCS access.
if [ -d /opt/openpi-cache ]; then
    cp -rn /opt/openpi-cache/* "$OPENPI_DATA_HOME/" 2>/dev/null || true
fi

# --- Extract the dataset ---
# The submit file transfers collab_dataset.tar.gz into the working dir.
# This tarball should contain the LeRobot dataset at the path that matches
# your config's repo_id (e.g., local/collab/).
DATASET_TAR="collab_dataset.tar.gz"
if [ -f "$DATASET_TAR" ]; then
    echo "Extracting dataset from $DATASET_TAR..."
    export HF_LEROBOT_HOME="${_CONDOR_SCRATCH_DIR:-.}/lerobot_data"
    # Tarball contains "collab/..." but repo_id is "local/collab",
    # so extract into the "local/" subdirectory.
    mkdir -p "$HF_LEROBOT_HOME/local"
    tar -xzf "$DATASET_TAR" -C "$HF_LEROBOT_HOME/local"
    echo "Dataset extracted to $HF_LEROBOT_HOME"
    ls "$HF_LEROBOT_HOME/local/collab"
    rm -f "$DATASET_TAR"
else
    echo "WARNING: $DATASET_TAR not found, assuming dataset is already available"
fi

# --- Compute normalization stats (idempotent, skips if already present) ---
echo "Computing normalization statistics..."
$PYTHON /app/scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

# --- Run training ---
echo "Starting training..."
$PYTHON /app/scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite

echo "============================================"
echo "Training complete at $(date)"
echo "============================================"
