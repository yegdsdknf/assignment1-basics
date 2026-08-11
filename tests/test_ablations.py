import torch
import pytest

from cs336_basics.model import SiLUFFN, SwiGLU, RMSNorm, TransformerBlock, TransformerLM


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_silu_ffn_preserves_input_shape():
    ffn = SiLUFFN(d_model=32, d_ff=96)
    inputs = torch.randn(2, 8, 32)

    outputs = ffn(inputs)

    assert outputs.shape == inputs.shape


def test_silu_ffn_matches_swiglu_parameter_count():
    swiglu = SwiGLU(d_model=512, d_ff=1344)
    silu_ffn = SiLUFFN(d_model=512, d_ff=2016)
    # 3 × 512 × 1344 = 2 × 512 × 2016
    # swiglu有3个权重矩阵，silu有2个权重矩阵

    assert count_parameters(swiglu) == 2_064_384
    assert count_parameters(silu_ffn) == 2_064_384


def make_transformer_block(norm_mode: str = "pre", ffn_type: str = "swiglu", d_ff: int = 64) -> TransformerBlock:
    return TransformerBlock(
        d_model=32, num_heads=4, d_ff=d_ff, theta=10_000.0, max_seq_len=8, norm_mode=norm_mode, ffn_type=ffn_type
    )


def make_transformer_lm(norm_mode: str, ffn_type: str, d_ff: int) -> TransformerLM:
    return TransformerLM(
        vocab_size=100,
        context_length=8,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=d_ff,
        rope_theta=10_000.0,
        norm_mode=norm_mode,
        ffn_type=ffn_type,
    )


@pytest.mark.parametrize("norm_mode", ["pre", "post", "none"])
def test_transformer_block_norm_modes_preserve_shape(norm_mode):
    block = make_transformer_block(norm_mode)
    inputs = torch.randn(2, 8, 32)

    outputs = block(inputs)

    assert outputs.shape == inputs.shape


def test_no_norm_block_contains_no_rmsnorm():
    block = make_transformer_block("none")

    assert not any(isinstance(module, RMSNorm) for module in block.modules())


def test_post_norm_matches_definition():
    torch.manual_seed(42)

    block = make_transformer_block("post")
    inputs = torch.randn(2, 8, 32)
    token_positions = torch.arange(8)

    after_attention = block.ln1(inputs + block.attn(inputs, token_positions))
    expected = block.ln2(after_attention + block.ffn(after_attention))

    actual = block(inputs, token_positions)

    torch.testing.assert_close(actual, expected)


def test_transformer_block_rejects_invalid_norm_mode():
    with pytest.raises(ValueError, match="norm_mode"):
        make_transformer_block("invalid")


@pytest.mark.parametrize(
    ("ffn_type", "expected_type"),
    [("swiglu", SwiGLU), ("silu", SiLUFFN)],
)
def test_transformer_block_selects_ffn(ffn_type, expected_type):
    block = make_transformer_block(ffn_type=ffn_type)

    assert isinstance(block.ffn, expected_type)


def test_transformer_block_rejects_invalid_ffn_type():
    with pytest.raises(ValueError, match="ffn_type"):
        make_transformer_block(ffn_type="invalid")


@pytest.mark.parametrize(
    ("norm_mode", "ffn_type", "d_ff"),
    [
        ("pre", "swiglu", 64),
        ("post", "swiglu", 64),
        ("none", "swiglu", 64),
        ("pre", "silu", 96),
    ],
)
def test_transformer_lm_ablation_variants(
    norm_mode,
    ffn_type,
    d_ff,
):
    model = make_transformer_lm(
        norm_mode=norm_mode,
        ffn_type=ffn_type,
        d_ff=d_ff,
    )
    inputs = torch.randint(0, 100, (2, 8))

    outputs = model(inputs)

    assert outputs.shape == (2, 8, 100)


def test_no_norm_lm_contains_no_rmsnorm():
    model = make_transformer_lm(
        norm_mode="none",
        ffn_type="swiglu",
        d_ff=64,
    )

    assert not any(isinstance(module, RMSNorm) for module in model.modules())


def test_silu_lm_matches_swiglu_lm_parameter_count():
    swiglu_model = make_transformer_lm(
        norm_mode="pre",
        ffn_type="swiglu",
        d_ff=64,
    )
    silu_model = make_transformer_lm(
        norm_mode="pre",
        ffn_type="silu",
        d_ff=96,
    )

    assert count_parameters(swiglu_model) == count_parameters(silu_model)


@pytest.mark.parametrize(
    ("ffn_type", "d_ff"),
    [
        ("swiglu", 64),
        ("silu", 96),
    ],
)
def test_transformer_block_uses_requested_ffn_width(
    ffn_type,
    d_ff,
):
    block = make_transformer_block(
        ffn_type=ffn_type,
        d_ff=d_ff,
    )

    assert block.ffn.d_ff == d_ff
