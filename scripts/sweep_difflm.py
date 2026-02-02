#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _get_lr_from_env() -> float | None:
    config_json = os.environ.get("WANDB_CONFIG_JSON")
    if not config_json:
        return None
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError:
        return None
    lr = cfg.get("lr")
    if lr is None:
        return None
    try:
        return float(lr)
    except (TypeError, ValueError):
        return None


def _resolve_run_name(base: str, lr: float) -> str:
    run_id = os.environ.get("WANDB_RUN_ID")
    run_name = os.environ.get("WANDB_RUN_NAME") or os.environ.get("WANDB_NAME")
    if run_name:
        return run_name
    if run_id:
        return f"{base}_lr{lr:g}_{run_id[:8]}"
    return f"{base}_lr{lr:g}"


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    parser = argparse.ArgumentParser(description="W&B sweep launcher for DLM pretraining.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate to sweep.")
    args, _unknown = parser.parse_known_args()

    lr = args.lr
    if lr is None:
        lr = _get_lr_from_env()
    if lr is None:
        raise SystemExit("Missing lr. Provide --lr or set WANDB_CONFIG_JSON with lr.")

    min_lr = 0.1 * lr

    project_dir = os.environ.get("PROJECT_DIR")
    if not project_dir:
        project_dir = str(Path(__file__).resolve().parents[1])

    datasets_dir = _require_env("DATASETS_DIR")
    ckpt_dir = _require_env("CKPT_DIR")
    log_dir = _require_env("LOG_DIR")

    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "1"))
    num_nodes = int(os.environ.get("NUM_NODES", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "6000")

    world_size = gpus_per_node * num_nodes
    model_parallel_size = 1
    pipeline_model_parallel_size = 1

    run_base = "simplestories_8k_dlm"
    run_name = _resolve_run_name(run_base, lr)

    seq_length = 512
    global_batch_size = 96
    micro_batch_size = 96
    training_tokens_per_epoch = 4_9152_000
    epochs = 1

    training_tokens = training_tokens_per_epoch * epochs
    tokens_per_batch = global_batch_size * seq_length
    train_iters = max(1, training_tokens // tokens_per_batch)

    save_interval = 1000
    eval_interval = save_interval
    target_val_tokens = 100_000
    tokens_per_val_batch = global_batch_size * seq_length
    val_iters = max(1, target_val_tokens // tokens_per_val_batch)

    tokenizer_vocab = f"{project_dir}/data/vocab_simplestories_8k.json"
    tokenizer_merges = f"{project_dir}/data/merges_simplestories_8k.txt"

    train_data_prefix = f"{datasets_dir}/simplestories_train_text_document"
    valid_data_prefix = f"{datasets_dir}/simplestories_test_text_document"

    logs_path = os.environ.get("LOG_DIR", f"{ckpt_dir}/cache/difflm/logs")
    checkpoint_path = f"{ckpt_dir}/cache/difflm/training_checkpoints/{run_name}"

    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb_project = os.environ.get("WANDB_PROJECT", "dlm")

    distributed_args = [
        "--nproc_per_node",
        str(gpus_per_node),
        "--nnodes",
        str(num_nodes),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        "--node_rank",
        str(node_rank),
    ]

    model_parallel_args = [
        "--tensor-model-parallel-size",
        str(model_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_model_parallel_size),
    ]

    gpt_model_args = [
        "--seq-length",
        str(seq_length),
        "--attention-backend",
        "flash",
        "--transformer-impl",
        "local",
        "--no-rope-fusion",
        "--attention-softmax-in-fp32",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-bias-swiglu-fusion",
        "--no-masked-softmax-fusion",
        "--num-layers",
        "12",
        "--hidden-size",
        "512",
        "--num-attention-heads",
        "8",
        "--ffn-hidden-size",
        "2048",
        "--swiglu",
        "--vocab-size",
        "8001",
        "--make-vocab-size-divisible-by",
        "1",
        "--normalization",
        "RMSNorm",
        "--max-position-embeddings",
        str(seq_length),
        "--norm-epsilon",
        "1e-6",
        "--rotary-base",
        "10000",
        "--disable-bias-linear",
        "--position-embedding-type",
        "rope",
        "--qk-layernorm",
        "--untie-embeddings-and-output-weights",
    ]

    plmt_args = [
        "--model-running-mode",
        "difflm-noshift",
        "--base-model",
        "vanilla",
        "--mask-token",
        "8000",
        "--attention-mask-type",
        "no_mask",
    ]

    training_args = [
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--no-gradient-accumulation-fusion",
        "--train-iters",
        str(train_iters),
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--init-method-std",
        "0.02",
        "--clip-grad",
        "1.0",
        "--bf16",
        "--lr",
        f"{lr}",
        "--min-lr",
        f"{min_lr}",
        "--lr-decay-style",
        "WSD",
        "--lr-warmup-iters",
        "200",
        "--lr-decay-iters",
        str(train_iters),
        "--lr-wsd-decay-style",
        "exponential",
        "--lr-wsd-decay-iters",
        "100",
        "--use-distributed-optimizer",
        "--num-distributed-optimizer-instances",
        str(num_nodes),
        "--rerun-mode",
        "validate_results",
        "--overlap-param-gather",
        "--overlap-grad-reduce",
        "--distributed-timeout-minutes",
        "60",
    ]

    data_args = [
        "--train-data-path",
        train_data_prefix,
        "--valid-data-path",
        valid_data_prefix,
        "--data-cache-path",
        f"{ckpt_dir}/cache/difflm/data/data_cache/{run_name}",
        "--tokenizer-type",
        "GPT2BPETokenizer",
        "--vocab-file",
        tokenizer_vocab,
        "--merge-file",
        tokenizer_merges,
    ]

    eval_and_logging_args = [
        "--log-interval",
        "1",
        "--log-params-norm",
        "--log-num-zeros-in-grad",
        "--log-throughput",
        "--log-progress",
        "--log-timers-to-tensorboard",
        "--log-validation-ppl-to-tensorboard",
        "--log-memory-to-tensorboard",
        "--save-interval",
        str(save_interval),
        "--non-persistent-save-interval",
        str(train_iters * 2),
        "--non-persistent-ckpt-type",
        "global",
        "--non-persistent-global-ckpt-dir",
        f"{checkpoint_path}/non_persistent",
        "--ckpt-format",
        "torch",
        "--eval-interval",
        str(eval_interval),
        "--save",
        checkpoint_path,
        "--eval-iters",
        str(val_iters),
        "--tensorboard-dir",
        f"{logs_path}/{run_name}/tensorboard",
        "--wandb-project",
        wandb_project,
        "--wandb-exp-name",
        run_name,
        "--wandb-run-id",
        os.environ.get("WANDB_RUN_ID", run_name),
        "--wandb-save-dir",
        f"{logs_path}/{run_name}/wandb",
        "--wandb-mode",
        wandb_mode,
        "--wandb-tags",
        "dlm",
        "--wandb-notes",
        "training",
        "--wandb-resume",
        "allow",
    ]

    cmd = [
        "torchrun",
        *distributed_args,
        str(Path(project_dir) / "pretrain_difflm.py"),
        *gpt_model_args,
        *training_args,
        *model_parallel_args,
        *data_args,
        *eval_and_logging_args,
        *plmt_args,
    ]

    os.makedirs(f"{logs_path}/{run_name}", exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
