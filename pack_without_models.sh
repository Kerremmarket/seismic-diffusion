#!/usr/bin/env bash
set -euo pipefail

# Package this project WITHOUT model artifacts.
# Produces KO_ARMT_data_no_models.tar.gz (default) or KO_ARMT_data_no_models.zip if --zip is passed.
#
# Excludes:
# - stage1_runs/
# - *.pt, *.pth, *.ckpt
# - wandb/
#
# Usage:
#   ./pack_without_models.sh            # creates tar.gz
#   ./pack_without_models.sh --zip      # creates zip
#
# Verify:
#   tar -tzf KO_ARMT_data_no_models.tar.gz | grep -E '\.(pt|pth|ckpt)$' || echo "No model files found"
#   unzip -l KO_ARMT_data_no_models.zip | grep -E '\.(pt|pth|ckpt)$' || echo "No model files found"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-tar}"

if [[ "$MODE" == "--zip" ]]; then
  ARCHIVE="KO_ARMT_data_no_models.zip"
  echo "Creating $ARCHIVE ..."
  # zip path is relative; we zip the current dir contents
  zip -r "$ARCHIVE" . \
    -x "stage1_runs/*" "stage1_runs/**" \
    -x "*/wandb/*" \
    -x "*.pt" "*.pth" "*.ckpt"
  echo "Done: $ARCHIVE"
else
  ARCHIVE="KO_ARMT_data_no_models.tar.gz"
  echo "Creating $ARCHIVE ..."
  tar -czf "$ARCHIVE" \
    --exclude='stage1_runs' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
    --exclude='wandb' \
    .
  echo "Done: $ARCHIVE"
fi

echo "Validate exclusion of model files using the commands in the header comments."


