#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from urllib.request import urlretrieve


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set. Set it in env/.env and source it.")
    return value


def export_jsonl(jsonl_dir: str) -> tuple[str, str]:
    from datasets import load_dataset

    train_path = os.path.join(jsonl_dir, "simplestories_train.jsonl")
    valid_path = os.path.join(jsonl_dir, "simplestories_test.jsonl")

    ds = load_dataset("SimpleStories/SimpleStories")

    def dump(split: str, out_path: str) -> None:
        if os.path.isfile(out_path):
            print(f"Found existing JSONL for {split}: {out_path}")
            return
        print(f"Writing {split} JSONL to {out_path}")
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                story = row.get("story")
                if story:
                    f.write('{"text": ' + repr(story) + "}\n")
                    count += 1
                    if count % 100000 == 0:
                        print(f"{split}: wrote {count} rows")
                        f.flush()

    dump("train", train_path)
    dump("test", valid_path)
    return train_path, valid_path


def download_tokenizer_files(target_dir: str) -> tuple[str, str]:
    os.makedirs(target_dir, exist_ok=True)
    urls = {
        "merges_simplestories_8k.txt": "https://huggingface.co/datasets/trixyL/simplestories-8k-megatron/resolve/main/merges_simplestories_8k.txt",
        "vocab_simplestories_8k.json": "https://huggingface.co/datasets/trixyL/simplestories-8k-megatron/resolve/main/vocab_simplestories_8k.json",
        "special_tokens_simplestories_8k.json": "https://huggingface.co/datasets/trixyL/simplestories-8k-megatron/resolve/main/special_tokens_simplestories_8k.json",
    }

    for filename, url in urls.items():
        dest = os.path.join(target_dir, filename)
        if os.path.isfile(dest):
            continue
        tmp_dest = f"{dest}.tmp"
        print(f"Downloading {url}")
        urlretrieve(url, tmp_dest)
        os.replace(tmp_dest, dest)

    vocab_file = os.path.join(target_dir, "vocab_simplestories_8k.json")
    merge_file = os.path.join(target_dir, "merges_simplestories_8k.txt")
    return vocab_file, merge_file


def ensure_vocab_has_special_tokens(tokenizer_dir: str, vocab_file: str) -> None:
    special_json = os.path.join(tokenizer_dir, "special_tokens_simplestories_8k.json")
    if not os.path.isfile(special_json):
        return
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(special_json, "r", encoding="utf-8") as f:
        special_tokens = json.load(f)
    updated = False
    for token, token_id in special_tokens.items():
        if token in vocab and vocab[token] == token_id:
            continue
        vocab[token] = token_id
        updated = True
    if updated:
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=True)


def run_preprocess(input_path: str, output_prefix: str, vocab_file: str, merge_file: str, workers: int) -> None:
    cmd = [
        sys.executable,
        "tools/preprocess_data.py",
        "--input",
        input_path,
        "--output-prefix",
        output_prefix,
        "--tokenizer-type",
        "GPT2BPETokenizer",
        "--vocab-file",
        vocab_file,
        "--merge-file",
        merge_file,
        "--append-eod",
        "--workers",
        str(workers),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SimpleStories for MegaDLMs.")
    parser.add_argument(
        "--jsonl-dir",
        default=None,
        help="Directory to write JSONL files. Defaults to DATASETS_DIR.",
    )
    parser.add_argument(
        "--download-tokenizer",
        action="store_true",
        help="Download tokenizer files into PROJECT_DIR/data if missing.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Tokenizer workers.")
    args = parser.parse_args()

    project_dir = require_env("PROJECT_DIR")
    datasets_dir = require_env("DATASETS_DIR")

    jsonl_dir = args.jsonl_dir or datasets_dir
    os.makedirs(jsonl_dir, exist_ok=True)

    tokenizer_dir = os.path.join(project_dir, "data")
    vocab_file = os.path.join(tokenizer_dir, "vocab_simplestories_8k.json")
    merge_file = os.path.join(tokenizer_dir, "merges_simplestories_8k.txt")

    if not os.path.isfile(vocab_file) or not os.path.isfile(merge_file):
        vocab_file, merge_file = download_tokenizer_files(tokenizer_dir)

    ensure_vocab_has_special_tokens(tokenizer_dir, vocab_file)

    if not os.path.isfile(vocab_file):
        raise SystemExit(
            f"Missing tokenizer vocab: {vocab_file}. "
            "Run with --download-tokenizer or place the file there."
        )
    if not os.path.isfile(merge_file):
        raise SystemExit(
            f"Missing tokenizer merges: {merge_file}. "
            "Run with --download-tokenizer or place the file there."
        )

    train_jsonl, valid_jsonl = export_jsonl(jsonl_dir)

    run_preprocess(
        train_jsonl,
        os.path.join(datasets_dir, "simplestories_train"),
        vocab_file,
        merge_file,
        args.workers,
    )
    run_preprocess(
        valid_jsonl,
        os.path.join(datasets_dir, "simplestories_test"),
        vocab_file,
        merge_file,
        args.workers,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
