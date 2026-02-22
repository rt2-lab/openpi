#!/bin/bash
set -euo pipefail

# Usage: ./chtc/stage_data.sh <netid> [lerobot_home] <dataset_name_under_local>
NETID="${1:?Usage: stage_data.sh <netid> <dataset_name_under_local>}"
LEROBOT_HOME="${HOME}/.cache/huggingface/lerobot"
DATASET_NAME="${2:?Usage: stage_data.sh <netid> <dataset_name_under_local>}"
REPO_ID="local/${DATASET_NAME}"

TRANSFER_HOST="transfer.chtc.wisc.edu"
GROUP_STAGING_DIR="/staging/groups/hagenow_group"
DATASET_DIR="${LEROBOT_HOME}/${REPO_ID}"
STAGED_TARBALL_NAME="${DATASET_NAME}_dataset.tar.gz"
TARBALL="/tmp/${STAGED_TARBALL_NAME}"

if [ ! -d "$DATASET_DIR" ]; then
  echo "ERROR: dataset directory not found: $DATASET_DIR"
  exit 1
fi

echo "Creating tarball from: $DATASET_DIR"
tar -czf "$TARBALL" -C "$LEROBOT_HOME" "$REPO_ID"
echo "Tarball size: $(du -sh "$TARBALL" | cut -f1)"

echo "Uploading to ${TRANSFER_HOST}:${GROUP_STAGING_DIR}/${STAGED_TARBALL_NAME}"
scp "$TARBALL" "${NETID}@${TRANSFER_HOST}:${GROUP_STAGING_DIR}/${STAGED_TARBALL_NAME}"
rm -f "$TARBALL"

echo "Done."
