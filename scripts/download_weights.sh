#!/usr/bin/env bash
# Download the weights for the experiment: D-FINE checkpoint (weights/) [+ HGNetv2 backbone (weight/hgnetv2/)].
#   bash scripts/download_weights.sh                 # model from experiment.yml
#   bash scripts/download_weights.sh x               # a given size: n | s | m | l | x | all
#   bash scripts/download_weights.sh x --pretrain coco --backbone   # see tools/steel/download_weights.py
set -euo pipefail
cd "$(dirname "$0")/.."
# interpreter: $PYTHON, else the repo venv made by `uv sync`, else python on PATH
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
export PYTHONIOENCODING=utf-8
"$PY" tools/steel/download_weights.py "$@"
