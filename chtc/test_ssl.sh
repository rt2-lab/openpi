#!/bin/bash
set -euo pipefail

PYTHON=/.venv/bin/python

# Same SSL fix from run_train.sh
export SSL_CERT_FILE=$($PYTHON -c "import certifi; print(certifi.where())")
echo "SSL_CERT_FILE=$SSL_CERT_FILE"
echo "File exists: $(test -f "$SSL_CERT_FILE" && echo yes || echo no)"

# Quick test: can Python verify the wandb API cert?
$PYTHON -c "
import urllib.request
resp = urllib.request.urlopen('https://api.wandb.ai/healthz')
print(f'wandb API reachable, status={resp.status}')
"
echo "SSL test passed!"
