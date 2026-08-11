import torch

from cs336_basics.model import SiLUFFN, SwiGLU


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
