"""绘制本项目 JSONL 训练日志中的学习曲线。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_METRICS_PATH = Path("artifacts/experiments/tinystories_5k_20260806/metrics.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/experiments/tinystories_5k_20260806/learning_curves.png")


@dataclass(frozen=True)
class MetricPoint:
    step: int
    elapsed_minutes: float
    loss: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制训练和验证 loss 曲线")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取由 train.py 生成的 JSONL 日志。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到指标文件：{path}")

    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON 格式错误：{path}:{line_number}") from error

            if not isinstance(record, dict):
                raise ValueError(f"每行必须是 JSON object：{path}:{line_number}")
            records.append(record)

    if not records:
        raise ValueError(f"指标文件为空：{path}")
    return records


def require_number(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{record.get('event', 'unknown')} 记录缺少数值字段 {key!r}")
    return float(value)


def extract_points(records: list[dict[str, object]]) -> tuple[list[MetricPoint], list[MetricPoint]]:
    """严格按照 train.py 的 event schema 提取指标。"""
    train_points: list[MetricPoint] = []
    validation_points: list[MetricPoint] = []

    for record in records:
        event = record.get("event")
        if event not in {"train", "validation"}:
            continue

        point = MetricPoint(
            step=int(require_number(record, "step")),
            elapsed_minutes=require_number(record, "elapsed_seconds") / 60,
            loss=require_number(record, "train_loss" if event == "train" else "validation_loss"),
        )

        if event == "train":
            train_points.append(point)
        else:
            validation_points.append(point)

    if not train_points or not validation_points:
        raise ValueError("日志必须同时包含 train 和 validation 记录")

    train_points.sort(key=lambda point: point.step)
    validation_points.sort(key=lambda point: point.step)
    return train_points, validation_points


def plot_panel(
    axis: plt.Axes,
    train_x: list[float | int],
    validation_x: list[float | int],
    train_points: list[MetricPoint],
    validation_points: list[MetricPoint],
    xlabel: str,
    best_label: str,
) -> None:
    axis.plot(train_x, [point.loss for point in train_points], color="#2563EB", label="Train loss")
    axis.plot(
        validation_x,
        [point.loss for point in validation_points],
        color="#DC2626",
        marker="o",
        markersize=4,
        label="Validation loss",
    )

    best_index = min(range(len(validation_points)), key=lambda index: validation_points[index].loss)
    best_x = validation_x[best_index]
    best_loss = validation_points[best_index].loss

    axis.scatter([best_x], [best_loss], color="#16A34A", edgecolor="white", s=70, zorder=5)
    axis.annotate(
        f"Best validation\n{best_label}, loss={best_loss:.5f}",
        xy=(best_x, best_loss),
        xytext=(12, 22),
        textcoords="offset points",
        color="#166534",
        arrowprops={"arrowstyle": "->", "color": "#16A34A"},
    )

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(alpha=0.25)
    axis.legend()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_points, validation_points = extract_points(read_jsonl(args.metrics))
    best_point = min(validation_points, key=lambda point: point.loss)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    plot_panel(
        axes[0],
        [point.step for point in train_points],
        [point.step for point in validation_points],
        train_points,
        validation_points,
        xlabel="Gradient step",
        best_label=f"step={best_point.step}",
    )
    axes[0].set_title("Loss vs Gradient Step")

    plot_panel(
        axes[1],
        [point.elapsed_minutes for point in train_points],
        [point.elapsed_minutes for point in validation_points],
        train_points,
        validation_points,
        xlabel="Elapsed time (minutes)",
        best_label=f"{best_point.elapsed_minutes:.2f} min",
    )
    axes[1].set_title("Loss vs Wall-clock Time")

    if args.title:
        figure.suptitle(args.title, fontsize=16, fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Training points: {len(train_points)}")
    print(f"Validation points: {len(validation_points)}")
    print(f"Best validation loss: {best_point.loss:.6f} at step {best_point.step}")
    print(f"Saved learning curves to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
