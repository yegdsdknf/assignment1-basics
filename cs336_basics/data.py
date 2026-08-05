import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset.ndim != 1:
        raise ValueError("dataset 必须是一维数组")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    if context_length <= 0:
        raise ValueError("context_length 必须为正数")
    if len(dataset) <= context_length:
        raise ValueError("dataset 长度必须大于 context_length")

    # 最高起点是 len(dataset) - context_length - 1。
    # (batch_size,)
    start_indices = np.random.randint(low=0, high=len(dataset) - context_length, size=batch_size)

    # (,context_length)
    offset = np.arange(context_length)
    # (batch_size, 1) + (1, context_length) 自动广播 -> (batch_size, context_length)
    indices = start_indices[:, None] + offset[None, :] 

    input_sequences = dataset[indices]
    target_sequences = dataset[indices + 1]

    #torch.as_tensor 会尽量避免不必要的数据复制。
    #如果设备和数据类型允许，它可能与原来的 NumPy 数组共享内存。
    inputs = torch.as_tensor(input_sequences, dtype=torch.long, device=device)

    targets = torch.as_tensor(target_sequences, dtype=torch.long, device=device)

    return inputs, targets
