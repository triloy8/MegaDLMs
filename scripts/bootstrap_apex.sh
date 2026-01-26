#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/MegaDLMs"
VENV_DIR="${REPO_ROOT}/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing venv at $VENV_DIR. Create it with: uv venv" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

uv pip install setuptools wheel ninja cmake

if [ ! -d "apex/.git" ]; then
  git clone https://github.com/NVIDIA/apex
fi
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..

if [ ! -d "TransformerEngine/.git" ]; then
  git clone https://github.com/NVIDIA/TransformerEngine
fi
cd TransformerEngine
pip install -v --no-build-isolation .
cd ..
