"""绘制正式消融实验的跨 seed 验证曲线。"""

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from cs336_basics.plot_metrics import read_jsonl, require_number

ARTIFACTS_DIR = Path("artifacts")
OUTPUT_DIR = Path("artifacts/experiments/assignment1_ablations_20260813")
CURVES_OUTPUT_PATH = OUTPUT_DIR / "validation_loss_curves.png"
FINAL_LOSS_OUTPUT_PATH = OUTPUT_DIR / "final_validation_loss.png"
SEEDS = (42, 43, 44)
VARIANTS = {
    "baseline": ("Pre-Norm + SwiGLU", "#0072B2", "o"),
    "no_norm": ("No Norm + SwiGLU", "#E69F00", "s"),
    "post_norm": ("Post-Norm + SwiGLU", "#009E73", "^"),
    "silu_matched": ("Pre-Norm + SiLU", "#CC79A7", "D"),
}


def load_validation_series(
    variant: str,
    seed: int,
) -> list[tuple[int, float]]:
    path = ARTIFACTS_DIR / f"formal-5k-{variant}-seed{seed}_metrics.jsonl"
    records = read_jsonl(path)

    points = [
        (
            int(require_number(record, "tokens_processed")),
            require_number(record, "validation_loss"),
        )
        for record in records
        if record.get("event") == "validation"
    ]

    if not points or points[-1][0] != 40_960_000:
        raise ValueError(f"实验结果不完整：{variant} seed={seed}")

    return points


def aggregate_variant(
    variant: str,
) -> tuple[list[int], list[float], list[float]]:
    series = [load_validation_series(variant, seed) for seed in SEEDS]
    tokens = [point[0] for point in series[0]]

    for seed, points in zip(SEEDS[1:], series[1:], strict=True):
        if [point[0] for point in points] != tokens:
            raise ValueError(f"{variant} seed={seed} 的验证位置与其他 seed 不一致")

    losses_by_step = zip(
        *[[point[1] for point in points] for points in series],
        strict=True,
    )

    means: list[float] = []
    standard_deviations: list[float] = []

    for losses in losses_by_step:
        values = list(losses)
        means.append(statistics.mean(values))
        standard_deviations.append(statistics.stdev(values))

    return tokens, means, standard_deviations


def plot_validation_curves() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(
        figsize=(9, 5.5),
        constrained_layout=True,
    )

    for variant, (label, color, marker) in VARIANTS.items():
        tokens, means, standard_deviations = aggregate_variant(variant)
        tokens_in_millions = [token / 1_000_000 for token in tokens]
        lower = [
            mean - deviation
            for mean, deviation in zip(
                means,
                standard_deviations,
                strict=True,
            )
        ]
        upper = [
            mean + deviation
            for mean, deviation in zip(
                means,
                standard_deviations,
                strict=True,
            )
        ]

        axis.plot(
            tokens_in_millions,
            means,
            label=label,
            color=color,
            marker=marker,
            markevery=5,
            markersize=4,
            linewidth=2,
        )
        axis.fill_between(
            tokens_in_millions,
            lower,
            upper,
            color=color,
            alpha=0.14,
        )

    axis.set_title("TinyStories Ablation Study\nMean validation loss ± 1 SD across 3 seeds")
    axis.set_xlabel("Training tokens (millions)")
    axis.set_ylabel("Validation cross-entropy loss")
    axis.set_xlim(left=0)
    axis.grid(alpha=0.25)
    axis.legend(title="Architecture")

    CURVES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        CURVES_OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"消融曲线已保存：{CURVES_OUTPUT_PATH}")


def plot_final_loss() -> None:
    labels: list[str] = []
    colors: list[str] = []
    per_variant_losses: list[list[float]] = []

    for variant, (label, color, _) in VARIANTS.items():
        losses = [load_validation_series(variant, seed)[-1][1] for seed in SEEDS]
        labels.append(label.replace(" + ", "\n+ "))
        colors.append(color)
        per_variant_losses.append(losses)

    figure, axis = plt.subplots(
        figsize=(9, 5.5),
        constrained_layout=True,
    )
    positions = list(range(len(labels)))
    all_losses = [loss for losses in per_variant_losses for loss in losses]

    for position, color, losses in zip(
        positions,
        colors,
        per_variant_losses,
        strict=True,
    ):
        mean = statistics.mean(losses)
        standard_deviation = statistics.stdev(losses)
        seed_offsets = (-0.12, 0.0, 0.12)

        # 浅色小点展示每个 seed，避免均值掩盖单次实验波动。
        axis.scatter(
            [position + offset for offset in seed_offsets],
            losses,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.8,
            s=55,
            zorder=3,
        )
        axis.errorbar(
            position,
            mean,
            yerr=standard_deviation,
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=1,
            markersize=10,
            capsize=7,
            linewidth=2.2,
            zorder=4,
        )
        axis.text(
            position,
            mean + standard_deviation + 0.005,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold",
        )

    loss_range = max(all_losses) - min(all_losses)
    margin = max(0.02, loss_range * 0.18)

    axis.set_title(
        "Final Validation Loss at 40.96M Training Tokens\nSmall dots: individual seeds; large markers: mean ± 1 SD"
    )
    axis.set_ylabel("Validation cross-entropy loss")
    axis.set_xticks(positions, labels)
    axis.set_ylim(
        min(all_losses) - margin,
        max(all_losses) + margin,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FINAL_LOSS_OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"最终 loss 对比图已保存：{FINAL_LOSS_OUTPUT_PATH}")


def main() -> None:
    plot_validation_curves()
    plot_final_loss()


if __name__ == "__main__":
    main()
