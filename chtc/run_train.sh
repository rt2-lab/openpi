#!/bin/bash
set -euo pipefail

# Usage: run_train.sh <config_name> <exp_name> <netid>
CONFIG_NAME="${1:?Usage: run_train.sh <config_name> <exp_name> <netid>}"
EXP_NAME="${2:?Usage: run_train.sh <config_name> <exp_name> <netid>}"
NETID="${3:?Usage: run_train.sh <config_name> <exp_name> <netid>}"

PYTHON=/.venv/bin/python
CKPT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"
BUNDLE_NAME="checkpoint_bundle.tar"

echo "OpenPI CHTC job: config=${CONFIG_NAME} exp=${EXP_NAME} netid=${NETID}"
nvidia-smi || true

export HF_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OPENPI_DATA_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$OPENPI_DATA_HOME"

# Seed tokenizer/assets cache from image.
if [ -d /opt/openpi-cache ]; then
    cp -rn /opt/openpi-cache/* "$OPENPI_DATA_HOME/" 2>/dev/null || true
fi

# Fail fast if tokenizer cache is missing. This avoids a later opaque GCS 401.
TOKENIZER_PATH="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Missing tokenizer file: $TOKENIZER_PATH"
    exit 2
fi

# Dataset tarball is transferred by HTCondor into scratch.
if [ -f collab_dataset.tar.gz ]; then
    export HF_LEROBOT_HOME="${_CONDOR_SCRATCH_DIR:-.}/lerobot_data"
    mkdir -p "$HF_LEROBOT_HOME/local"
    tar -xzf collab_dataset.tar.gz -C "$HF_LEROBOT_HOME/local"
    rm -f collab_dataset.tar.gz
fi

echo "Computing normalization statistics..."
$PYTHON /app/scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

# Package checkpoints once at end (or during graceful eviction) so HTCondor
# transfers a single known file to /staging.
package_checkpoints() {
    if [ -f "$BUNDLE_NAME" ]; then
        return 0
    fi
    if [ ! -d "$CKPT_DIR" ]; then
        echo "No checkpoint directory found at $CKPT_DIR; skipping bundle."
        return 0
    fi
    echo "Packaging checkpoints into $BUNDLE_NAME ..."
    tar -cf "$BUNDLE_NAME" -C "checkpoints/${CONFIG_NAME}" "${EXP_NAME}"
}

trap 'package_checkpoints' EXIT TERM INT

echo "Starting training..."
mkdir -p "$CKPT_DIR"
$PYTHON /app/scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite
