#!/usr/bin/env bash
set -euo pipefail

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/dacnet-pycache"
echo "Installing smoke-test requirements..."
pip install --no-cache-dir -r requirements-smoke.txt

echo "Running smoke tests..."
python scripts/replicate_chexnet.py --help >/dev/null
python scripts/dacnet.py --help >/dev/null
python -m py_compile scripts/replicate_chexnet.py scripts/dacnet.py

echo "Smoke test completed successfully."
