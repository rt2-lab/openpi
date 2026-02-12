#!/bin/bash
# ==============================================================================
# Pull trained checkpoints from CHTC /staging back to your local machine
# for running inference on your lab GPU.
#
# Usage:
#   ./chtc/pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]
#
# Example:
#   ./chtc/pull_checkpoints.sh lpxu pi05_collab pi05_4695539
#   # Then serve locally:
#   uv run scripts/serve_policy.py policy:checkpoint \
#       --policy.config=pi05_collab \
#       --policy.dir=checkpoints/pi05_collab/pi05_4695539/299
# ==============================================================================

set -euo pipefail

NETID="${1:?Usage: pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]}"
CONFIG_NAME="${2:?Usage: pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]}"
EXP_NAME="${3:?Usage: pull_checkpoints.sh <netid> <config_name> <exp_name> [local_dir]}"
LOCAL_DIR="${4:-./checkpoints}"
TRANSFER_HOST="transfer.chtc.wisc.edu"

REMOTE_PATH="/staging/groups/hagenow_group/openpi/${EXP_NAME}.tar"

echo "Downloading checkpoints from CHTC..."
echo "  Remote: ${TRANSFER_HOST}:${REMOTE_PATH}"
echo "  Local:  ${LOCAL_DIR}/${CONFIG_NAME}/${EXP_NAME}"

mkdir -p "${LOCAL_DIR}/${CONFIG_NAME}/${EXP_NAME}"
TMP_TAR="$(mktemp -p "${LOCAL_DIR}" "ckpt_XXXX.tar")"

scp "${NETID}@${TRANSFER_HOST}:${REMOTE_PATH}" "${TMP_TAR}"
tar -xf "${TMP_TAR}" -C "${LOCAL_DIR}/${CONFIG_NAME}/${EXP_NAME}" --strip-components=1
rm -f "${TMP_TAR}"

echo ""
echo "Checkpoints available at: ${LOCAL_DIR}/${CONFIG_NAME}/${EXP_NAME}/"
ls -la "${LOCAL_DIR}/${CONFIG_NAME}/${EXP_NAME}/"

echo ""
echo "To run inference locally:"
echo "  uv run scripts/serve_policy.py policy:checkpoint \\"
echo "      --policy.config=${CONFIG_NAME} \\"
echo "      --policy.dir=${LOCAL_DIR}/${CONFIG_NAME}/${EXP_NAME}/<step>"
