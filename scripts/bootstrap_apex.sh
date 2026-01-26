#!/usr/bin/env bash
set -euo pipefail

APEX_DIR="${APEX_DIR:-$PWD/third_party/apex}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "VIRTUAL_ENV is not set. Activate your venv first." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to clone apex." >&2
  exit 1
fi

if [ ! -d "$APEX_DIR/.git" ]; then
  mkdir -p "$(dirname "$APEX_DIR")"
  git clone https://github.com/NVIDIA/apex "$APEX_DIR"
fi

cd "$APEX_DIR"

APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install -v --no-build-isolation .
