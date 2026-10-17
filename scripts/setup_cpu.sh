#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ ! -f .venv/pyvenv.cfg ]; then
    uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "The CPU lock targets Python 3.12"'
uv pip install --system-certs --python .venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match -r requirements/cpu.lock
uv pip install --system-certs --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
.venv/bin/python -m urban_change --version
