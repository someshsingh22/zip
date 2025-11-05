#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/launch_ddp.sh [CONFIG]
# Defaults to configs/train_bine.yaml and 8 processes.

CONFIG=${1:-configs/train_bine.yaml}
NPROC=${NPROC:-8}

echo "Launching DDP training with $NPROC processes using config: $CONFIG"
torchrun --nproc_per_node="$NPROC" scripts/train_bine.py --config "$CONFIG"


