"""绘制正式消融实验的跨 seed 验证曲线。"""

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from cs336_basics.plot_metrics import read_jsonl, require_number

ARTIFACTS_DIR = Path("artifacts")
OUTPUT_PATH = Path("artifacts/experiments/assignment1_ablations_20260813/validation_loss_curves.png")
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


def main() -> None:
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"消融曲线已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
