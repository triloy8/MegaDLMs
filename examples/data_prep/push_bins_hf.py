#!/usr/bin/env python3
import argparse
import os
import sys


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise SystemExit(f"Missing file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Megatron bin/idx files to HF.")
    parser.add_argument(
        "--repo-id",
        default="trixyL/simplestories-4k-megatron",
        help="Hugging Face dataset repo (e.g., user/name).",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.getcwd(), "data"),
        help="Directory containing bin/idx files.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create repo if it does not exist.",
    )
    parser.add_argument(
        "--include-tokenizer",
        action="store_true",
        help="Also upload merges/vocab/special_tokens tokenizer files.",
    )
    parser.add_argument(
        "--tokenizer-prefix",
        default="simplestories_4k",
        help="Tokenizer filename suffix (e.g., simplestories_4k).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from exc

    files = [
        "simplestories_train_text_document.bin",
        "simplestories_train_text_document.idx",
        "simplestories_test_text_document.bin",
        "simplestories_test_text_document.idx",
    ]
    if args.include_tokenizer:
        files.extend(
            [
                f"merges_{args.tokenizer_prefix}.txt",
                f"vocab_{args.tokenizer_prefix}.json",
                f"special_tokens_{args.tokenizer_prefix}.json",
            ]
        )
    paths = [os.path.join(args.data_dir, f) for f in files]
    for path in paths:
        require_file(path)

    api = HfApi()
    if args.create:
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    for path in paths:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(f"Uploaded {len(paths)} files to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
