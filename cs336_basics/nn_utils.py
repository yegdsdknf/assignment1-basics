import torch
from collections.abc import Iterable
from torch import nn


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    inputs:  (..., vocab_size)
    targets: (...)
    """
    # 减去最大值，防止 exp 发生数值溢出。
    max_values = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_values

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))

    target_logits = shifted.gather(  # 拿到target对应的input的shifted值
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    return torch.mean(log_sum_exp - target_logits)


def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,  # max_l2_norm限制整体梯度不超过该值
):
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm 必须为正数")

    # parameters 可能是生成器，因此先保存需要处理的梯度。
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]

    if not gradients:
        return

    # 计算所有参数梯度组成的大向量的整体 L2 范数。
    total_norm_squared = sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)
    total_norm = torch.sqrt(total_norm_squared)

    # 与 PyTorch 的实现保持一致，epsilon 防止除零。
    clip_coefficient = max_l2_norm / (total_norm + 1e-6)

    if clip_coefficient < 1:
        for gradient in gradients:
            gradient.mul_(clip_coefficient)  # mul_表示原地操作
