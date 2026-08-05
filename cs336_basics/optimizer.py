import math
from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError("lr 不能为负数")
        if eps < 0:
            raise ValueError("eps 不能为负数")
        if not 0 <= betas[0] < 1:
            raise ValueError("beta1 必须位于 [0, 1)")
        if not 0 <= betas[1] < 1:
            raise ValueError("beta2 必须位于 [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能为负数")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        loss = None

        # closure 是一个重新计算损失的函数。由于整个 step() 被 no_grad 修饰，所以调用 closure 时需要暂时重新启用梯度。
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError("AdamW 不支持稀疏梯度")

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 解耦权重衰减，不把 weight decay 加入梯度矩估计。
                parameter.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(  # 一阶动量mt​=β1​mt−1​+(1−β1​)gt​，抑制震荡
                    gradient,
                    alpha=1 - beta1,
                )
                exp_avg_sq.mul_(beta2).addcmul_(  # 二阶动量vt​=β2​vt−1​+(1−β2​)gt^2，调节步长​
                    gradient,
                    gradient,
                    value=1 - beta2,
                )
                # 初值为0时，防止初始梯度过小，进行偏差修正
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                # 计算分母
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # addcdiv_：parameter += value * exp_avg / denominator
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )

        return loss


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if it < 0:
        raise ValueError("it 不能为负数")
    if warmup_iters < 0:
        raise ValueError("warmup_iters 不能为负数")
    if cosine_cycle_iters <= warmup_iters:
        raise ValueError("cosine_cycle_iters 必须大于 warmup_iters")
    if min_learning_rate > max_learning_rate:
        raise ValueError("min_learning_rate 不能大于 max_learning_rate")

    # 第一阶段：从 0 线性预热到最大学习率。
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 第三阶段：余弦周期结束后保持最小学习率。
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 第二阶段：把当前位置映射到 [0, 1]。
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

    return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
