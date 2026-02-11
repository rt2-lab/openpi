#!/bin/bash
# ==============================================================================
# Pull trained checkpoints from CHTC /staging back to your local machine
# for running inference on your lab GPU.
#
# Usage:
#   ./chtc/pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]
#
# Example:
#   ./chtc/pull_checkpoints.sh lsanchez pi05_collab chtc_train
#   # Then serve locally:
#   uv run scripts/serve_policy.py policy:checkpoint \
#       --policy.config=pi05_collab \
#       --policy.dir=checkpoints/pi05_collab/chtc_train_01/30000
# ==============================================================================

set -euo pipefail

NETID="${1:?Usage: pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]}"
CONFIG_NAME="${2:?Usage: pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]}"
EXP_NAME="${3:?Usage: pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]}"
LOCAL_DIR="${4:-./checkpoints}"
TRANSFER_HOST="transfer.chtc.wisc.edu"

REMOTE_PATH="/staging/${NETID}/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

echo "Downloading checkpoints from CHTC..."
echo "  Remote: ${TRANSFER_HOST}:${REMOTE_PATH}"
echo "  Local:  ${LOCAL_DIR}"

mkdir -p "${LOCAL_DIR}/${CONFIG_NAME}"
scp -r "${NETID}@${TRANSFER_HOST}:${REMOTE_PATH}" "${LOCAL_DIR}/${CONFIG_NAME}/"

echo ""
echo "Checkpoints available at: ${LOCAL_DIR}/"
ls -la "${LOCAL_DIR}/"

echo ""
echo "To run inference locally:"
echo "  uv run scripts/serve_policy.py policy:checkpoint \\"
echo "      --policy.config=${CONFIG_NAME} \\"
echo "      --policy.dir=${LOCAL_DIR}/${CONFIG_NAME}/${EXP_NAME}/<step>"
