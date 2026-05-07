#!/usr/bin/env bash
# Lightweight reviewer artifact smoke test.
# This script checks artifact layout only. It does not run full training.

set -eu

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Running reviewer artifact layout check..."
python scripts/check_artifact_layout.py

echo
if [ -f tools/preflight_server_run.py ]; then
  echo "Optional preflight helper detected:"
  echo "  python tools/preflight_server_run.py"
  echo
  echo "Manual preflight note:"
  echo "  Run the command above only if you want an environment/server preflight."
  echo "  It is not part of this smoke test and does not perform full training by default."
else
  echo "No tools/preflight_server_run.py helper found."
fi
