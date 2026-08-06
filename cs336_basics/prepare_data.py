import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"


def iter_documents(
    input_path: Path,
    chunk_size: int = 1024 * 1024,
) -> Iterator[str]:
    """流式读取由 <|endoftext|> 分隔的文档。"""
    buffer = ""

    with input_path.open("r", encoding="utf-8") as input_file:
        while chunk := input_file.read(chunk_size):
            buffer += chunk
            fragments = buffer.split(END_OF_TEXT)

            # 最后一个片段可能还没读取完整，留到下一轮。
            yield from fragments[:-1]

            buffer = fragments[-1]

    # 文件末尾没有特殊 token 时，也把最后一篇文档输出
    if buffer:
        yield buffer


def encode_to_binary(
    tokenizer: Tokenizer, input_path: Path, output_path: Path, flush_tokens: int = 1_000_000
) -> dict[str, int | float | str]:
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    # 防止意外覆盖已经编码好的大文件。
    for path in (output_path, metadata_path, partial_path):
        if path.exists():
            raise FileExistsError(f"输出文件已经存在：{path}")

    max_token_id = max(tokenizer.vocab)

    if max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(f"最大 token ID 为 {max_token_id}，无法保存为 uint16")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    token_buffer: list[int] = []
    num_tokens = 0
    num_documents = 0
    start_time = time.perf_counter()

    with partial_path.open("wb") as output_file:
        for document in iter_documents(input_path):
            # 恢复被 iter_documents 移除的文档分隔符。
            token_ids = tokenizer.encode(document + END_OF_TEXT)
            token_buffer.extend(token_ids)
            num_documents += 1

            if len(token_buffer) >= flush_tokens:
                token_array = np.asarray(
                    token_buffer,
                    dtype=np.uint16,
                )
                token_array.tofile(output_file)

                num_tokens += token_array.size
                token_buffer.clear()

            if num_documents % 10_000 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"documents={num_documents:,}, tokens={num_tokens + len(token_buffer):,}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

        # 写入最后不足 flush_tokens 的部分。
        if token_buffer:
            token_array = np.asarray(
                token_buffer,
                dtype=np.uint16,
            )
            token_array.tofile(output_file)

            num_tokens += token_array.size
            token_buffer.clear()

    # 所有数据成功写完后，才把临时文件改成最终名称。
    partial_path.replace(output_path)

    elapsed_seconds = time.perf_counter() - start_time
    output_num_bytes = output_path.stat().st_size

    if output_num_bytes != num_tokens * np.dtype(np.uint16).itemsize:
        raise AssertionError(f"文件大小与 token 数量不一致：{output_num_bytes=}，{num_tokens=}")

    metadata: dict[str, int | float | str] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dtype": "uint16",
        "vocab_size": len(tokenizer.vocab),
        "num_documents": num_documents,
        "num_tokens": num_tokens,
        "num_bytes": output_num_bytes,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": num_tokens / elapsed_seconds,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 TinyStories 流式编码为 uint16 token 文件")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("artifacts/tinystories_bpe/vocab.json"),
    )
    parser.add_argument(
        "--merges",
        type=Path,
        default=Path("artifacts/tinystories_bpe/merges.json"),
    )
    parser.add_argument(
        "--flush-tokens",
        type=int,
        default=1_000_000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.flush_tokens <= 0:
        raise ValueError("flush_tokens 必须为正数")

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=[END_OF_TEXT],
    )

    metadata = encode_to_binary(
        tokenizer=tokenizer,
        input_path=args.input,
        output_path=args.output,
        flush_tokens=args.flush_tokens,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
