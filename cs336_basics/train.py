import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TinyStories Transformer LM")

    # 数据与输出路径
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--log-file", type=Path, default=Path("artifacts/training_metrics.jsonl"))
    parser.add_argument("--resume", type=Path, default=None)

    # 模型超参数
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # 日志、验证和 checkpoint
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto、cpu、cuda、cuda:0 或 mps",
    )
    parser.add_argument("--seed", type=int, default=42)

    # 调试：始终训练同一个 batch
    parser.add_argument("--overfit-single-batch", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.train_data.is_file():
        raise FileNotFoundError(f"找不到训练数据：{args.train_data}")

    if not args.validation_data.is_file():
        raise FileNotFoundError(f"找不到验证数据：{args.validation_data}")

    if args.context_length <= 0:
        raise ValueError("context_length 必须为正数")

    if args.batch_size <= 0:
        raise ValueError("batch_size 必须为正数")

    if args.max_iters <= 0:
        raise ValueError("max_iters 必须为正数")

    if not 0 <= args.warmup_iters < args.max_iters:
        raise ValueError("warmup_iters 必须位于 [0, max_iters)")

    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_device(device: torch.device) -> None:
    """让计时包含设备上尚未结束的异步计算。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_memmap(path: Path, context_length: int) -> np.memmap:
    dataset = np.memmap(path, dtype=np.uint16, mode="r")

    if len(dataset) <= context_length:
        raise ValueError(f"{path} 只有 {len(dataset)} 个 token，不足以采样 context_length={context_length}")

    return dataset


def write_log(log_path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)

    print(line, flush=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    validation_data: np.memmap,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
) -> float:
    """
    使用固定随机种子抽验证 batch。

    保存并恢复 NumPy 随机状态，避免验证过程改变训练数据的
    随机采样序列。
    """
    was_training = model.training
    numpy_random_state = np.random.get_state()

    model.eval()
    np.random.seed(seed)

    losses: list[float] = []

    try:
        for _ in range(eval_batches):
            inputs, targets = get_batch(
                dataset=validation_data, batch_size=batch_size, context_length=context_length, device=str(device)
            )
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss.item())
    finally:
        np.random.set_state(numpy_random_state)

        if was_training:
            model.train()

    return sum(losses) / len(losses)


def update_learning_rate(optimizer: AdamW, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def save_step_checkpoint(
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_path = checkpoint_dir / f"step_{iteration:07d}.pt"

    # 防止意外覆盖已有实验。
    if checkpoint_path.exists():
        print(
            f"checkpoint 已存在，跳过：{checkpoint_path}",
            flush=True,
        )
        return checkpoint_path

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=checkpoint_path,
    )

    print(
        f"checkpoint 已保存：{checkpoint_path}",
        flush=True,
    )

    return checkpoint_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    set_random_seed(args.seed)

    if device.type == "cuda":
        # 只对 CUDA 启用，不要在 MPS 上启用。
        torch.set_float32_matmul_precision("high")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data = load_memmap(args.train_data, args.context_length)
    validation_data = load_memmap(args.validation_data, args.context_length)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iteration = 0

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{args.resume}")

        start_iteration = load_checkpoint(
            src=args.resume,
            model=model,
            optimizer=optimizer,
        )

        print(
            f"从 iteration={start_iteration} 恢复训练",
            flush=True,
        )

    if start_iteration > args.max_iters:
        raise ValueError("checkpoint 的 iteration 已超过 max_iters")

    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(
        json.dumps(
            {
                "device": str(device),
                "num_parameters": num_parameters,
                "train_tokens": len(train_data),
                "validation_tokens": len(validation_data),
                "start_iteration": start_iteration,
                "max_iters": args.max_iters,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    model.train()

    fixed_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    if args.overfit_single_batch:
        fixed_batch = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=str(device),
        )

        print("已启用单 batch 过拟合模式", flush=True)

    running_loss = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    running_steps = 0
    latest_train_loss: float | None = None

    synchronize_device(device)
    training_start_time = time.perf_counter()

    for iteration in range(
        start_iteration,
        args.max_iters,
    ):
        learning_rate = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        update_learning_rate(optimizer, learning_rate)

        if fixed_batch is None:
            inputs, targets = get_batch(
                dataset=train_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=str(device),
            )
        else:
            inputs, targets = fixed_batch

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        loss.backward()

        gradient_clipping(
            model.parameters(),
            max_l2_norm=args.max_grad_norm,
        )

        optimizer.step()

        completed_steps = iteration + 1

        # detach 后只保存数值，不保留计算图。
        running_loss.add_(loss.detach())
        running_steps += 1

        should_log = completed_steps == 1 or completed_steps % args.log_interval == 0

        if should_log:
            synchronize_device(device)
            elapsed_seconds = time.perf_counter() - training_start_time

            latest_train_loss = running_loss.item() / running_steps

            session_steps = completed_steps - start_iteration
            session_tokens = session_steps * args.batch_size * args.context_length

            write_log(
                args.log_file,
                {
                    "event": "train",
                    "step": completed_steps,
                    "elapsed_seconds": elapsed_seconds,
                    "tokens_processed": (completed_steps * args.batch_size * args.context_length),
                    "session_tokens_per_second": (session_tokens / elapsed_seconds),
                    "learning_rate": learning_rate,
                    "train_loss": latest_train_loss,
                },
            )

            running_loss.zero_()
            running_steps = 0

        should_evaluate = completed_steps % args.eval_interval == 0 or completed_steps == args.max_iters

        if should_evaluate:
            validation_loss = evaluate(
                model=model,
                validation_data=validation_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_batches=args.eval_batches,
                device=device,
                seed=args.seed + 1,
            )

            # 防止极端异常 loss 让 exp 溢出。
            perplexity = math.exp(min(validation_loss, 20.0))

            synchronize_device(device)
            elapsed_seconds = time.perf_counter() - training_start_time

            write_log(
                args.log_file,
                {
                    "event": "validation",
                    "step": completed_steps,
                    "elapsed_seconds": elapsed_seconds,
                    "train_loss": latest_train_loss,
                    "validation_loss": validation_loss,
                    "perplexity": perplexity,
                },
            )

        should_checkpoint = completed_steps % args.checkpoint_interval == 0 or completed_steps == args.max_iters

        if should_checkpoint:
            save_step_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_steps,
                checkpoint_dir=args.checkpoint_dir,
            )


if __name__ == "__main__":
    main()
