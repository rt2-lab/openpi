#!/bin/bash
# ==============================================================================
# Pull trained checkpoints from CHTC /staging back to your local machine
# for running inference on your lab GPU.
#
# Usage:
#   ./chtc/pull_checkpoints.sh <netid> <cluster_id> [local_dir]
#
# Example:
#   ./chtc/pull_checkpoints.sh lsanchez 12345678
#   # Then serve locally:
#   uv run scripts/serve_policy.py policy:checkpoint \
#       --policy.config=pi05_collab \
#       --policy.dir=checkpoints/pi05_collab/chtc_train_01/30000
# ==============================================================================

set -euo pipefail

NETID="${1:?Usage: pull_checkpoints.sh <netid> <cluster_id> [local_dir]}"
CLUSTER_ID="${2:?Usage: pull_checkpoints.sh <netid> <cluster_id> [local_dir]}"
LOCAL_DIR="${3:-./checkpoints}"
TRANSFER_HOST="transfer.chtc.wisc.edu"

REMOTE_PATH="/staging/${NETID}/checkpoints_${CLUSTER_ID}.tar.gz"
LOCAL_TAR="/tmp/checkpoints_${CLUSTER_ID}.tar.gz"

echo "Downloading checkpoints from CHTC..."
echo "  Remote: ${TRANSFER_HOST}:${REMOTE_PATH}"
echo "  Local:  ${LOCAL_DIR}"

scp "${NETID}@${TRANSFER_HOST}:${REMOTE_PATH}" "$LOCAL_TAR"

echo "Extracting to ${LOCAL_DIR}..."
mkdir -p "$LOCAL_DIR"
tar -xzf "$LOCAL_TAR" -C "$LOCAL_DIR"

echo ""
echo "Checkpoints available at: ${LOCAL_DIR}/"
ls -la "${LOCAL_DIR}/"

echo ""
echo "To run inference locally:"
echo "  uv run scripts/serve_policy.py policy:checkpoint \\"
echo "      --policy.config=pi05_collab \\"
echo "      --policy.dir=${LOCAL_DIR}/pi05_collab/chtc_train_01/<step>"

rm -f "$LOCAL_TAR"
