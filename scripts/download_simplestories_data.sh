#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [destination_dir] [hf_repo] [tokenizer_prefix]" >&2
}

data_dir="${1:-$(pwd)/data}"
hf_repo="${2:-trixyL/simplestories-4k-megatron}"
tokenizer_prefix="${3:-simplestories_4k}"
base_url="https://huggingface.co/datasets/${hf_repo}/resolve/main"

mkdir -p "${data_dir}"

download() {
    local url="$1"
    local dest="${data_dir}/$(basename "${url}")"

    if [[ -f "${dest}" ]]; then
        echo "Already present: ${dest}"
        return 0
    fi

    local tmp="${dest}.tmp.$$"
    echo "Downloading ${url}"
    curl -L --fail --retry 3 -o "${tmp}" "${url}"
    mv "${tmp}" "${dest}"
}

download "${base_url}/merges_${tokenizer_prefix}.txt"
download "${base_url}/vocab_${tokenizer_prefix}.json"
download "${base_url}/special_tokens_${tokenizer_prefix}.json"
download "${base_url}/simplestories_train_text_document.bin"
download "${base_url}/simplestories_train_text_document.idx"
download "${base_url}/simplestories_test_text_document.bin"
download "${base_url}/simplestories_test_text_document.idx"
