#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


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
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                story = row.get("story")
                if story:
                    f.write('{"text": ' + repr(story) + "}\n")

    dump("train", train_path)
    dump("test", valid_path)
    return train_path, valid_path


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
    parser.add_argument("--workers", type=int, default=8, help="Tokenizer workers.")
    args = parser.parse_args()

    project_dir = require_env("PROJECT_DIR")
    datasets_dir = require_env("DATASETS_DIR")

    jsonl_dir = args.jsonl_dir or datasets_dir
    os.makedirs(jsonl_dir, exist_ok=True)

    vocab_file = os.path.join(project_dir, "data", "vocab_simplestories_8k.json")
    merge_file = os.path.join(project_dir, "data", "merges_simplestories_8k.txt")
    if not os.path.isfile(vocab_file):
        raise SystemExit(f"Missing tokenizer vocab: {vocab_file}")
    if not os.path.isfile(merge_file):
        raise SystemExit(f"Missing tokenizer merges: {merge_file}")

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
