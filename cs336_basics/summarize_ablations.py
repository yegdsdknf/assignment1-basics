"""汇总 Assignment 1 的正式消融实验结果。"""

import json
import re
import statistics
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
OUTPUT_PATH = Path("artifacts/experiments/assignment1_ablations_20260813/metrics_summary.md")
VARIANTS = ("baseline", "no_norm", "post_norm", "silu_matched")
SEEDS = (42, 43, 44)


def read_summary_value(console_text: str, key: str) -> float:
    matches = re.findall(
        rf"{re.escape(key)}\s+([0-9.]+)",
        console_text,
    )
    if not matches:
        raise ValueError(f"控制台日志中缺少指标：{key}")

    return float(matches[-1])


def load_run(variant: str, seed: int) -> dict[str, float | int | str]:
    prefix = ARTIFACTS_DIR / f"formal-5k-{variant}-seed{seed}"
    metrics_path = prefix.with_name(f"{prefix.name}_metrics.jsonl")
    console_path = prefix.with_name(f"{prefix.name}_console.log")

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    validations = [record for record in records if record["event"] == "validation"]

    if not validations or validations[-1]["step"] != 5000:
        raise ValueError(f"实验结果不完整：{variant} seed={seed}")

    final = validations[-1]
    best = min(validations, key=lambda record: record["validation_loss"])
    console_text = console_path.read_text(encoding="utf-8")

    return {
        "variant": variant,
        "seed": seed,
        "final_loss": final["validation_loss"],
        "final_ppl": final["perplexity"],
        "best_loss": best["validation_loss"],
        "best_step": best["step"],
        "throughput": read_summary_value(
            console_text,
            "system/end_to_end_tokens_per_second",
        ),
        "peak_memory_gib": read_summary_value(
            console_text,
            "system/peak_gpu_memory_gib",
        ),
    }


def mean_std(
    rows: list[dict[str, float | int | str]],
    key: str,
) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    rows = [load_run(variant, seed) for variant in VARIANTS for seed in SEEDS]

    lines = [
        "# Assignment 1 Ablation Metrics",
        "",
        "## Per-run results",
        "",
        "| Variant | Seed | Final loss | PPL | Best step | Tokens/s | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | "
            f"{float(row['final_loss']):.5f} | "
            f"{float(row['final_ppl']):.5f} | "
            f"{row['best_step']} | "
            f"{float(row['throughput']):.0f} | "
            f"{float(row['peak_memory_gib']):.3f} |"
        )

    baseline_rows = [row for row in rows if row["variant"] == "baseline"]
    baseline_loss, _ = mean_std(baseline_rows, "final_loss")

    lines.extend(
        [
            "",
            "## Aggregate results",
            "",
            "| Variant | Final loss | PPL | Δ loss vs baseline | Tokens/s | Peak GiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        loss_mean, loss_std = mean_std(variant_rows, "final_loss")
        ppl_mean, ppl_std = mean_std(variant_rows, "final_ppl")
        throughput_mean, throughput_std = mean_std(
            variant_rows,
            "throughput",
        )
        memory_mean, memory_std = mean_std(
            variant_rows,
            "peak_memory_gib",
        )

        lines.append(
            f"| {variant} | {loss_mean:.5f} ± {loss_std:.5f} | "
            f"{ppl_mean:.5f} ± {ppl_std:.5f} | "
            f"{loss_mean - baseline_loss:+.5f} | "
            f"{throughput_mean:.0f} ± {throughput_std:.0f} | "
            f"{memory_mean:.3f} ± {memory_std:.3f} |"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
