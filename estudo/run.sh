#!/bin/bash
cd "$(dirname "$0")"
export PYTHONWARNINGS="ignore::UserWarning:sklearn.utils.parallel,ignore::FutureWarning"
source .venv/bin/activate
python -W "ignore::UserWarning:sklearn.utils.parallel" run.py "$@" 2>&1 | grep -v -e "sklearn.utils.parallel" -e "warnings.warn" || true
