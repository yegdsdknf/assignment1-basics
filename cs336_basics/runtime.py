"""训练与生成入口共用的运行时工具。"""

import random

import numpy as np
import torch


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_device(device: torch.device) -> None:
    """让性能计时包含尚未结束的 CUDA 计算。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
