from pathlib import Path

from cs336_basics.bpe import count_pretokens_parallel


def main() -> None:
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    special_tokens = ["<|endoftext|>"]

    parallel_counts = count_pretokens_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_workers=12,
        num_chunks=64,
    )

    print("并行唯一 pre-token 数：", len(parallel_counts))
    print("并行 pre-token 总次数：", sum(parallel_counts.values()))


if __name__ == "__main__":
    main()
