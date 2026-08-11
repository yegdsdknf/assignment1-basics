import importlib
import pkgutil
import pytest
import cs336_basics
import numpy as np
import torch

from cs336_basics.model import TransformerLM

from cs336_basics.bpe import count_pretokens_parallel
from cs336_basics.plot_metrics import extract_points
from cs336_basics.train import resolve_variant_config, evaluate, parse_args, build_wandb_config, init_wandb


def test_all_package_modules_import_without_side_effect_errors() -> None:
    for module in pkgutil.iter_modules(cs336_basics.__path__):
        importlib.import_module(f"cs336_basics.{module.name}")


def test_plot_metrics_separates_train_and_validation_events() -> None:
    records = [
        {
            "event": "train",
            "step": 100,
            "elapsed_seconds": 10.0,
            "train_loss": 2.0,
        },
        {
            "event": "validation",
            "step": 100,
            "elapsed_seconds": 11.0,
            "train_loss": 2.0,
            "validation_loss": 2.1,
        },
    ]

    train_points, validation_points = extract_points(records)

    assert len(train_points) == 1
    assert len(validation_points) == 1
    assert train_points[0].loss == 2.0
    assert validation_points[0].loss == 2.1


def test_parallel_pretokenization_has_an_explicit_boundary_contract() -> None:
    try:
        count_pretokens_parallel(
            input_path="unused.txt",
            special_tokens=["<other>"],
            num_workers=1,
        )
    except ValueError as error:
        assert "<|endoftext|>" in str(error)
    else:
        raise AssertionError("非 TinyStories 文档边界应被明确拒绝")


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("baseline", ("pre", "swiglu", 1344)),
        ("no_norm", ("none", "swiglu", 1344)),
        ("post_norm", ("post", "swiglu", 1344)),
        ("silu_matched", ("pre", "silu", 2016)),
    ],
)
def test_resolve_variant_config(variant, expected):
    assert resolve_variant_config(variant, 1344) == expected


def test_silu_variant_requires_even_base_d_ff():
    with pytest.raises(ValueError, match="偶数"):
        resolve_variant_config("silu_matched", 1343)


def test_resolve_variant_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        resolve_variant_config("unknown", 1344)


def test_train_args_use_independent_evaluation_seed():
    args = parse_args(
        [
            "--train-data",
            "train.bin",
            "--validation-data",
            "validation.bin",
            "--seed",
            "44",
        ]
    )

    assert args.seed == 44
    assert args.eval_seed == 2026


def test_evaluate_is_repeatable_and_restores_numpy_state():
    torch.manual_seed(42)

    model = TransformerLM(
        vocab_size=32,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10_000.0,
    )
    validation_data = (
        np.arange(
            128,
            dtype=np.uint16,
        )
        % 32
    )

    np.random.seed(1234)
    expected_next_random_value = np.random.randint(0, 1_000_000)

    np.random.seed(1234)
    first_loss = evaluate(
        model=model,
        validation_data=validation_data,
        batch_size=2,
        context_length=4,
        eval_batches=3,
        device=torch.device("cpu"),
        seed=2026,
    )
    actual_next_random_value = np.random.randint(0, 1_000_000)

    second_loss = evaluate(
        model=model,
        validation_data=validation_data,
        batch_size=2,
        context_length=4,
        eval_batches=3,
        device=torch.device("cpu"),
        seed=2026,
    )

    assert first_loss == second_loss
    assert actual_next_random_value == expected_next_random_value


def test_wandb_is_disabled_by_default():
    args = parse_args(
        [
            "--train-data",
            "train.bin",
            "--validation-data",
            "validation.bin",
        ]
    )

    assert args.wandb_mode == "disabled"


def test_build_wandb_config_serializes_paths():
    args = parse_args(
        [
            "--train-data",
            "train.bin",
            "--validation-data",
            "validation.bin",
            "--variant",
            "silu_matched",
        ]
    )

    config = build_wandb_config(
        args=args,
        norm_mode="pre",
        ffn_type="silu",
        effective_d_ff=2016,
        num_parameters=22_696_448,
        device=torch.device("cpu"),
    )

    assert config["train_data"] == "train.bin"
    assert config["validation_data"] == "validation.bin"
    assert config["effective_d_ff"] == 2016
    assert config["num_parameters"] == 22_696_448
    assert config["tokens_per_step"] == 8192
    assert config["total_tokens"] == 40_960_000

    # disabled 模式不能创建 W&B run。
    assert init_wandb(args, config) is None
