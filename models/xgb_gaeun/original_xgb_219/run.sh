#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Reuse the system scikit-learn OpenMP library, not any existing model runtime.
OMP_DIR=/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/.dylibs
if [ -d "$OMP_DIR" ]; then
    export DYLD_LIBRARY_PATH="$OMP_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec "$PROJECT_DIR/.venv/bin/python" "$@"
