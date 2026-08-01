from cs336_basics.bpe import train_bpe
import argparse
import time
import json
from pathlib import Path


def save_result(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], outdir: Path, elapsed_seconds: float):
    """使用十六进制保存 bytes，保证 JSON 可以无损恢复。"""
    outdir.mkdir(parents=True, exist_ok=True)

    vocab_data = {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()}
    merges_data = [[left.hex(), right.hex()] for left, right in merges]

    longest_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    metrics = {
        "elapsed_seconds": elapsed_seconds,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_id": longest_id,
        "longest_token_num_bytes": len(longest_token),
        "longest_token_hex": longest_token.hex(),
        "longest_token_repr": repr(longest_token),
        "longest_token_text": longest_token.decode(
            "utf-8",
            errors="replace",
        ),
    }

    with (outdir / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab_data, file, indent=2)

    with (outdir / "merges.json").open("w", encoding="utf-8") as file:
        json.dump(merges_data, file, indent=2)

    with (outdir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 TinyStories 上训练 byte-level BPE tokenizer",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="TinyStories 训练或验证集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tinystories_bpe"),
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunks", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"找不到训练文件：{args.input}")

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=str(args.input),
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        num_workers=args.workers,
        num_chunks=args.chunks,
    )

    elapsed_seconds = time.perf_counter() - start_time

    expected_merges = args.vocab_size - 256 - 1

    if len(vocab) != args.vocab_size:
        raise AssertionError(f"期望词表大小 {args.vocab_size}，实际为 {len(vocab)}")

    if len(merges) != expected_merges:
        raise AssertionError(f"期望 {expected_merges} 次 merge，实际为 {len(merges)}")

    save_result(
        vocab=vocab,
        merges=merges,
        outdir=args.output_dir,
        elapsed_seconds=elapsed_seconds,
    )


if __name__ == "__main__":
    main()
