#!/usr/bin/env bash
set -euo pipefail

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
echo "Installing smoke-test requirements..."
pip install --no-cache-dir -r requirements-smoke.txt

echo "Running smoke tests..."
python - <<'PY'
import sys
import torch
print('torch', torch.__version__)
try:
    from XRay_app.utils import model_utils
    print('Imported XRay_app.utils.model_utils OK')
except Exception as e:
    print('Import failed:', e)
    sys.exit(1)
print('Smoke tests finished')
PY

echo "Done. To run the app: cd XRay_app && streamlit run app.py"
