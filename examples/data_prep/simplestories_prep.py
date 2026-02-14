#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from urllib.request import urlretrieve


DEFAULT_TOKENIZER_REPO = "trixyL/simplestories-4k-megatron"
DEFAULT_TOKENIZER_PREFIX = "simplestories_4k"
DEFAULT_TEXT_DATASET = "SimpleStories/SimpleStories"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set. Set it in env/.env and source it.")
    return value


def export_jsonl(jsonl_dir: str, force: bool, dataset_name: str) -> tuple[str, str]:
    from datasets import load_dataset

    train_path = os.path.join(jsonl_dir, "simplestories_train.jsonl")
    valid_path = os.path.join(jsonl_dir, "simplestories_test.jsonl")

    ds = load_dataset(dataset_name)

    def dump(split: str, out_path: str) -> None:
        if os.path.isfile(out_path) and not force:
            print(f"Found existing JSONL for {split}: {out_path}")
            return
        print(f"Writing {split} JSONL to {out_path}")
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                story = row.get("story")
                if story:
                    f.write(json.dumps({"text": story}, ensure_ascii=True) + "\n")
                    count += 1
                    if count % 100000 == 0:
                        print(f"{split}: wrote {count} rows")
                        f.flush()

    dump("train", train_path)
    dump("test", valid_path)
    return train_path, valid_path


def download_tokenizer_files(
    target_dir: str, tokenizer_repo: str, tokenizer_prefix: str
) -> tuple[str, str]:
    os.makedirs(target_dir, exist_ok=True)
    base_url = f"https://huggingface.co/datasets/{tokenizer_repo}/resolve/main"
    urls = {
        f"merges_{tokenizer_prefix}.txt": f"{base_url}/merges_{tokenizer_prefix}.txt",
        f"vocab_{tokenizer_prefix}.json": f"{base_url}/vocab_{tokenizer_prefix}.json",
        f"special_tokens_{tokenizer_prefix}.json": (
            f"{base_url}/special_tokens_{tokenizer_prefix}.json"
        ),
    }

    for filename, url in urls.items():
        dest = os.path.join(target_dir, filename)
        if os.path.isfile(dest):
            continue
        tmp_dest = f"{dest}.tmp"
        print(f"Downloading {url}")
        urlretrieve(url, tmp_dest)
        os.replace(tmp_dest, dest)

    vocab_file = os.path.join(target_dir, f"vocab_{tokenizer_prefix}.json")
    merge_file = os.path.join(target_dir, f"merges_{tokenizer_prefix}.txt")
    return vocab_file, merge_file


def ensure_vocab_has_special_tokens(
    tokenizer_dir: str, tokenizer_prefix: str, vocab_file: str
) -> None:
    special_json = os.path.join(tokenizer_dir, f"special_tokens_{tokenizer_prefix}.json")
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
    parser.add_argument(
        "--tokenizer-repo",
        default=DEFAULT_TOKENIZER_REPO,
        help=(
            "Hugging Face dataset repo containing tokenizer files "
            "(example: trixyL/simplestories-4k-megatron)."
        ),
    )
    parser.add_argument(
        "--tokenizer-prefix",
        default=DEFAULT_TOKENIZER_PREFIX,
        help=(
            "Tokenizer filename suffix used in merges/vocab/special_tokens files "
            "(example: simplestories_4k)."
        ),
    )
    parser.add_argument(
        "--text-dataset",
        default=DEFAULT_TEXT_DATASET,
        help=(
            "Dataset ID used to export train/test JSONL text "
            "(example: SimpleStories/SimpleStories)."
        ),
    )
    parser.add_argument("--workers", type=int, default=8, help="Tokenizer workers.")
    parser.add_argument(
        "--force-jsonl",
        action="store_true",
        help="Rebuild JSONL files even if they already exist.",
    )
    args = parser.parse_args()

    project_dir = require_env("PROJECT_DIR")
    datasets_dir = require_env("DATASETS_DIR")

    jsonl_dir = args.jsonl_dir or datasets_dir
    os.makedirs(jsonl_dir, exist_ok=True)

    tokenizer_dir = os.path.join(project_dir, "data")
    vocab_file = os.path.join(tokenizer_dir, f"vocab_{args.tokenizer_prefix}.json")
    merge_file = os.path.join(tokenizer_dir, f"merges_{args.tokenizer_prefix}.txt")

    if not os.path.isfile(vocab_file) or not os.path.isfile(merge_file):
        if not args.download_tokenizer:
            raise SystemExit(
                "Tokenizer files are missing. Re-run with --download-tokenizer "
                "or place vocab/merges in PROJECT_DIR/data."
            )
        vocab_file, merge_file = download_tokenizer_files(
            tokenizer_dir, args.tokenizer_repo, args.tokenizer_prefix
        )

    ensure_vocab_has_special_tokens(tokenizer_dir, args.tokenizer_prefix, vocab_file)

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

    train_jsonl, valid_jsonl = export_jsonl(
        jsonl_dir, args.force_jsonl, args.text_dataset
    )

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
