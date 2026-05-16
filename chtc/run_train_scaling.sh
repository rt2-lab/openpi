#!/bin/bash
set -euo pipefail

# Usage: run_train_scaling.sh <config_name> <exp_name> <netid> <data_pct>
CONFIG_NAME="${1:?Usage: run_train_scaling.sh <config_name> <exp_name> <netid> <data_pct>}"
EXP_NAME="${2:?Usage: run_train_scaling.sh <config_name> <exp_name> <netid> <data_pct>}"
NETID="${3:?Usage: run_train_scaling.sh <config_name> <exp_name> <netid> <data_pct>}"
DATA_PCT="${4:?Usage: run_train_scaling.sh <config_name> <exp_name> <netid> <data_pct>}"

PYTHON=/.venv/bin/python
CKPT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"
BUNDLE_NAME="checkpoint_bundle.tar"

echo "OpenPI CHTC scaling job: config=${CONFIG_NAME} exp=${EXP_NAME} netid=${NETID} data_pct=${DATA_PCT}"
nvidia-smi || true

export HF_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OPENPI_DATA_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export SSL_CERT_FILE=$($PYTHON -c "import certifi; print(certifi.where())")
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$OPENPI_DATA_HOME"

if [ -d /opt/openpi-cache ]; then
    cp -rn /opt/openpi-cache/* "$OPENPI_DATA_HOME/" 2>/dev/null || true
fi

TOKENIZER_PATH="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Missing tokenizer file: $TOKENIZER_PATH"
    exit 2
fi

export HF_LEROBOT_HOME="${_CONDOR_SCRATCH_DIR:-.}/lerobot_data"
mkdir -p "$HF_LEROBOT_HOME"
for tarball in *_dataset.tar.gz; do
    [ -f "$tarball" ] || continue
    echo "Extracting dataset tarball: $tarball"
    tar -xzf "$tarball" -C "$HF_LEROBOT_HOME"
    rm -f "$tarball"
done
echo "Datasets available under $HF_LEROBOT_HOME:"
ls "$HF_LEROBOT_HOME/local/" 2>/dev/null || echo "(none)"

echo "Computing normalization statistics..."
$PYTHON /app/scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

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

NORMAL_COMPLETION=0
on_exit_or_signal() {
    if [ "$NORMAL_COMPLETION" -eq 1 ]; then
        return 0
    fi
    package_checkpoints
}
trap 'on_exit_or_signal' EXIT TERM INT

echo "Starting training with data_pct=${DATA_PCT}%..."
mkdir -p "$CKPT_DIR"
$PYTHON /app/scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite \
    --data.data-pct="$DATA_PCT"

package_checkpoints
NORMAL_COMPLETION=1
trap - EXIT TERM INT
