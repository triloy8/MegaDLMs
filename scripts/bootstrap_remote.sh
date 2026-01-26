#!/usr/bin/env bash
set -euo pipefail

err() {
    echo "bootstrap_remote: $*" >&2
}

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

export PATH="${HOME}/.local/bin:${PATH}"

apt_updated=false
apt_install() {
    if [ "${apt_updated}" = false ]; then
        ${SUDO} apt-get update
        apt_updated=true
    fi
    ${SUDO} apt-get install -y "$@"
}

if ! command -v curl >/dev/null 2>&1; then
    apt_install curl
fi

if ! command -v git >/dev/null 2>&1; then
    apt_install git
fi

if ! command -v pkg-config >/dev/null 2>&1; then
    apt_install pkg-config
fi

if ! command -v cmake >/dev/null 2>&1; then
    apt_install cmake
fi

if ! dpkg -s libcairo2-dev >/dev/null 2>&1; then
    apt_install libcairo2-dev
fi

if ! dpkg -s libdbus-1-dev >/dev/null 2>&1; then
    apt_install libdbus-1-dev
fi

if ! dpkg -s libcudnn8-dev >/dev/null 2>&1; then
    apt_install libcudnn8-dev
fi

if ! dpkg -s python3-dev >/dev/null 2>&1; then
    apt_install python3-dev
fi

if ! command -v ninja >/dev/null 2>&1; then
    apt_install ninja-build
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v just >/dev/null 2>&1; then
    curl -fsSL https://just.systems/install.sh | ${SUDO} bash -s -- --to /usr/local/bin
fi

if ! command -v tmux >/dev/null 2>&1; then
    apt_install tmux
fi

if ! command -v nvitop >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uvx nvitop
    else
        python3 -m pip install --user nvitop
    fi
fi

repo_root="${REMOTE_ROOT:-${HOME}/MegaDLMs}"
repo_url="${REMOTE_REPO_URL:-https://github.com/triloy8/MegaDLMs.git}"

if [ ! -d "${repo_root}/.git" ]; then
    if [ -d "${repo_root}" ] && [ "$(ls -A "${repo_root}")" ]; then
        err "${repo_root} exists but is not a git repo; aborting clone"
        exit 1
    fi
    git clone "${repo_url}" "${repo_root}"
else
    (
        cd "${repo_root}"
        git fetch --all --prune
        git pull --ff-only || err "git pull failed; please resolve manually"
    )
fi

mkdir -p "${repo_root}/data" "${repo_root}/runs" "${repo_root}/env"

(
    cd "${repo_root}"
    if command -v uv >/dev/null 2>&1; then
        uv venv
        uv sync
    fi
)

echo "Bootstrap complete: repo synced, uv environment ready, tooling installed."
