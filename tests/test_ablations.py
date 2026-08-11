import torch
import pytest

from cs336_basics.model import SiLUFFN, SwiGLU, RMSNorm, TransformerBlock


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


def make_transformer_block(norm_mode: str) -> TransformerBlock:
    return TransformerBlock(d_model=32, num_heads=4, d_ff=64, theta=10_000.0, max_seq_len=8, norm_mode=norm_mode)


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
