#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
export UV_CACHE_DIR="$PWD/.uv-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
if command -v uv >/dev/null 2>&1; then
  uv sync
  exec uv run --no-sync python -m ntweak --open
fi
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -e .
exec .venv/bin/python -m ntweak --open
