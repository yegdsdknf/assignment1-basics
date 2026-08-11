import importlib
import pkgutil
import pytest
import cs336_basics

from cs336_basics.bpe import count_pretokens_parallel
from cs336_basics.plot_metrics import extract_points
from cs336_basics.train import resolve_variant_config


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
