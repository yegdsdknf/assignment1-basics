import os
import torch

from typing import IO, BinaryIO
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module, optimizer: Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: Optimizer) -> int:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return int(checkpoint["iteration"])
