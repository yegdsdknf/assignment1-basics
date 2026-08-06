from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    """检查验证集 token 文件并解码一段样本。"""
    path = Path("data/tinystories_valid.uint16.bin")

    tokens = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )

    print("文件大小：", path.stat().st_size)
    print("token 数：", len(tokens))
    print("前 20 个 token：", tokens[:20])
    print("token 范围：", tokens.min(), tokens.max())

    assert path.stat().st_size == len(tokens) * 2
    assert tokens.max() < 10_000

    tokenizer = Tokenizer.from_files(
        vocab_filepath="artifacts/tinystories_bpe/vocab.json",
        merges_filepath="artifacts/tinystories_bpe/merges.json",
        special_tokens=["<|endoftext|>"],
    )

    sample_ids = tokens[:300].astype(int).tolist()
    print(tokenizer.decode(sample_ids))


if __name__ == "__main__":
    main()
