#!/usr/bin/env bash
# Train from the single config file:  bash train.sh [experiment.yml]
# = python tools/steel/prepare_experiment.py experiment.yml --train
set -euo pipefail
cd "$(dirname "$0")"
# interpreter: $PYTHON, else the repo venv made by `uv sync`, else python on PATH
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
export PYTHONIOENCODING=utf-8
"$PY" tools/steel/prepare_experiment.py "${1:-experiment.yml}" --train
