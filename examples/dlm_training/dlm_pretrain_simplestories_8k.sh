#!/bin/bash
set -euo pipefail

date
nvidia-smi || true

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

: "${PROJECT_DIR:?Set PROJECT_DIR in env/.env}"
: "${DATASETS_DIR:?Set DATASETS_DIR in env/.env}"
: "${CKPT_DIR:?Set CKPT_DIR in env/.env}"
: "${LOG_DIR:?Set LOG_DIR in env/.env}"

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6000}

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((WORLD_SIZE / MODEL_PARALLEL_SIZE / PIPELINE_MODEL_PARALLEL_SIZE))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $MODEL_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
)

RUN_NAME=simplestories_8k_dlm
ROOT_DIR=$PROJECT_DIR
cd "$ROOT_DIR"

SEQ_LENGTH=512
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=64
TRAINING_TOKENS_PER_EPOCH=3276800000
EPOCHS=1

TRAINING_TOKENS=$((TRAINING_TOKENS_PER_EPOCH * EPOCHS))
TOKENS_PER_BATCH=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
TRAIN_ITERS=$((TRAINING_TOKENS / TOKENS_PER_BATCH))
if [ "$TRAIN_ITERS" -lt 1 ]; then
    TRAIN_ITERS=1
fi

SAVE_INTERVAL=$((TRAIN_ITERS / EPOCHS))
if [ "$SAVE_INTERVAL" -lt 1 ]; then
    SAVE_INTERVAL=1
fi

EVAL_INTERVAL=$SAVE_INTERVAL
TARGET_VAL_TOKENS=100000
TOKENS_PER_VAL_BATCH=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
VAL_ITERS=$((TARGET_VAL_TOKENS / TOKENS_PER_VAL_BATCH))
if [ "$VAL_ITERS" -lt 1 ]; then
    VAL_ITERS=1
fi

TOKENIZER_VOCAB=$ROOT_DIR/data/vocab_simplestories_8k.json
TOKENIZER_MERGES=$ROOT_DIR/data/merges_simplestories_8k.txt

train_data_prefix="$DATASETS_DIR/simplestories_train_text_document"
valid_data_prefix="$DATASETS_DIR/simplestories_test_text_document"

LOGS_PATH=${LOG_DIR:-"$CKPT_DIR/cache/difflm/logs"}
CHECKPOINT_PATH=$CKPT_DIR/cache/difflm/training_checkpoints/${RUN_NAME}

mkdir -p "$LOGS_PATH/$RUN_NAME"
mkdir -p "$CHECKPOINT_PATH"

GPT_MODEL_ARGS=(
    --seq-length $SEQ_LENGTH
    --attention-backend flash
    --transformer-impl local
    --no-rope-fusion
    --attention-softmax-in-fp32
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --no-masked-softmax-fusion

    --num-layers 12
    --hidden-size 512
    --num-attention-heads 8
    --ffn-hidden-size 2048
    --swiglu
    --vocab-size 8001
    --make-vocab-size-divisible-by 1
    --normalization RMSNorm
    --max-position-embeddings $SEQ_LENGTH
    --norm-epsilon 1e-6
    --rotary-base 10000
    --disable-bias-linear
    --position-embedding-type rope
    --qk-layernorm
    --untie-embeddings-and-output-weights
)

PLMT_ARGS=(
    --model-running-mode difflm-noshift
    --base-model vanilla
    --mask-token 8000
    --attention-mask-type no_mask
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --no-gradient-accumulation-fusion
    --train-iters $TRAIN_ITERS
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --clip-grad 1.0
    --bf16
    --lr 0.0002
    --min-lr 0.00002
    --lr-decay-style WSD
    --lr-warmup-iters 200
    --lr-decay-iters $TRAIN_ITERS
    --lr-wsd-decay-style exponential
    --lr-wsd-decay-iters 100
    --use-distributed-optimizer
    --num-distributed-optimizer-instances $NUM_NODES
    --rerun-mode validate_results
    --overlap-param-gather
    --overlap-grad-reduce
    --distributed-timeout-minutes 60
)

DATA_ARGS=(
    --train-data-path $train_data_prefix
    --valid-data-path $valid_data_prefix
    --data-cache-path $CKPT_DIR/cache/difflm/data/data_cache/${RUN_NAME}
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $TOKENIZER_VOCAB
    --merge-file $TOKENIZER_MERGES
)

if [ -z "${WANDB_MODE:-}" ]; then
    WANDB_MODE=offline
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-params-norm
    --log-num-zeros-in-grad
    --log-throughput
    --log-progress
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --save-interval $SAVE_INTERVAL
    --non-persistent-save-interval $((TRAIN_ITERS * 2))
    --non-persistent-ckpt-type global
    --non-persistent-global-ckpt-dir $CHECKPOINT_PATH/non_persistent
    --ckpt-format torch
    --eval-interval $EVAL_INTERVAL
    --save $CHECKPOINT_PATH
    --eval-iters $VAL_ITERS
    --tensorboard-dir ${LOGS_PATH}/${RUN_NAME}/tensorboard
    --wandb-project dlm
    --wandb-exp-name $RUN_NAME
    --wandb-save-dir ${LOGS_PATH}/${RUN_NAME}/wandb
    --wandb-mode $WANDB_MODE
    --wandb-tags 'dlm'
    --wandb-notes 'training'
    --wandb-resume allow
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_difflm.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PLMT_ARGS[@]}
