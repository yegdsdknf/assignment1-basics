#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "用法：$0 <variant> <seed> <screen|formal>"
    echo "variant: baseline | no_norm | post_norm | silu_matched"
    exit 1
fi

variant=$1
seed=$2
stage=$3

case "$variant" in
    baseline|no_norm|post_norm|silu_matched)
        ;;
    *)
        echo "不支持的 variant：$variant"
        exit 1
        ;;
esac

case "$stage" in
    screen)
        max_iters=1000
        checkpoint_interval=1000
        wandb_group="screen-1k"
        ;;
    formal)
        max_iters=5000
        checkpoint_interval=5000
        wandb_group="formal-5k"
        ;;
    *)
        echo "不支持的实验阶段：$stage"
        exit 1
        ;;
esac

run_name="${wandb_group}-${variant}-seed${seed}"
checkpoint_dir="artifacts/${run_name}_checkpoints"
log_file="artifacts/${run_name}_metrics.jsonl"

if [[ -e "$checkpoint_dir" || -e "$log_file" ]]; then
    echo "实验输出已存在，拒绝覆盖：$run_name"
    exit 1
fi

echo "启动实验：$run_name"

exec uv run python -m cs336_basics.train \
    --train-data data/tinystories_train.uint16.bin \
    --validation-data data/tinystories_valid.uint16.bin \
    --checkpoint-dir "$checkpoint_dir" \
    --log-file "$log_file" \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --rope-theta 10000 \
    --variant "$variant" \
    --batch-size 32 \
    --max-iters "$max_iters" \
    --max-learning-rate 3e-4 \
    --min-learning-rate 3e-5 \
    --warmup-iters 100 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight-decay 0.1 \
    --max-grad-norm 1.0 \
    --log-interval 10 \
    --eval-interval 100 \
    --eval-batches 20 \
    --checkpoint-interval "$checkpoint_interval" \
    --device cuda \
    --seed "$seed" \
    --eval-seed 2026 \
    --wandb-mode offline \
    --wandb-project assignment1-ablations \
    --wandb-group "$wandb_group" \
    --wandb-run-name "$run_name" \
    --wandb-dir artifacts/wandb