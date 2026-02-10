#!/bin/bash
# ==============================================================================
# Stage the converted LeRobot dataset + checkpoints to CHTC /staging
#
# Usage:
#   ./chtc/stage_data.sh <netid> [lerobot_data_dir]
#
# Example:
#   # After running convert_collab_data_to_lerobot.py, stage the result:
#   ./chtc/stage_data.sh lxu ~/.cache/lerobot_data
#
# This creates a tarball of your LeRobot dataset and uploads it to
# /staging/<netid>/ on CHTC's transfer server.
#
# References:
#   https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata
# ==============================================================================

set -euo pipefail

NETID="${1:?Usage: stage_data.sh <netid> [lerobot_data_dir]}"
# Default LeRobot data home (where convert script saves datasets)
LEROBOT_HOME="${2:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}}"
TRANSFER_HOST="transfer.chtc.wisc.edu"
STAGING_DIR="/staging/${NETID}"

echo "============================================"
echo "CHTC Data Staging"
echo "  NetID:        $NETID"
echo "  LeRobot home: $LEROBOT_HOME"
echo "  Staging:      $STAGING_DIR"
echo "============================================"

# --- Verify local data exists ---
if [ ! -d "$LEROBOT_HOME" ]; then
    echo "ERROR: LeRobot data directory not found: $LEROBOT_HOME"
    echo "  Run the conversion script first:"
    echo "  uv run examples/collab/convert_collab_data_to_lerobot.py --db_dir /path/to/episodes"
    exit 1
fi

echo "Contents of $LEROBOT_HOME:"
ls -la "$LEROBOT_HOME"

# --- Create tarball ---
TARBALL="/tmp/collab_dataset.tar.gz"
echo ""
echo "Creating dataset tarball..."
tar -czf "$TARBALL" -C "$LEROBOT_HOME" .
echo "Tarball size: $(du -sh "$TARBALL" | cut -f1)"

# --- Upload to CHTC /staging ---
echo ""
echo "Uploading to ${TRANSFER_HOST}:${STAGING_DIR}/"
echo "(You will be prompted for your password)"
scp "$TARBALL" "${NETID}@${TRANSFER_HOST}:${STAGING_DIR}/collab_dataset.tar.gz"

echo ""
echo "Upload complete."
echo ""
echo "Next steps:"
echo "  1. Update chtc/train.sub:  replace <dockerhub_user> and set netid = $NETID"
echo "  2. SSH to submit host:     ssh ${NETID}@submit1.chtc.wisc.edu"
echo "  3. Submit the job:         condor_submit chtc/train.sub"
echo "  4. Monitor:                condor_q"
echo ""
echo "After training, retrieve checkpoints from /staging/${NETID}/checkpoints_*.tar.gz"

# Cleanup
rm -f "$TARBALL"
