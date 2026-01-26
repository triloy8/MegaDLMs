#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="${1:-examples/dlm_training/dlm_pretrain_simplestories_8k.sh}"
TMUX_SESSION="${TMUX_SESSION:-megadlms-train}"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required on the remote host." >&2
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

if [ ! -f env/.env ]; then
    echo "Missing env/.env; run \`just sync-env\` first" >&2
    exit 1
fi

source env/.env

CMD="uv run -- bash \"$TRAIN_SCRIPT\""
tmux new-session -d -s "$TMUX_SESSION" "source env/.env && ${CMD}"
echo "Started training in tmux session: $TMUX_SESSION"
