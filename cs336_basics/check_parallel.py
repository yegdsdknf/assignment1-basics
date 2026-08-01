from pathlib import Path

from cs336_basics.bpe import count_pretokens_parallel


def main() -> None:
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    special_tokens = ["<|endoftext|>"]

    # 串行版本接收完整文本。
    # with input_path.open(encoding="utf-8") as file:
    #     text = file.read()

    # serial_counts = count_pretokens(
    #     text,
    #     special_tokens,
    # )

    # 并行版本接收文件路径。
    parallel_counts = count_pretokens_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_workers=8,
        num_chunks=64,
    )

    # print("串行唯一 pre-token 数：", len(serial_counts))
    print("并行唯一 pre-token 数：", len(parallel_counts))

    # print("串行 pre-token 总次数：", sum(serial_counts.values()))
    print("并行 pre-token 总次数：", sum(parallel_counts.values()))

    # assert serial_counts == parallel_counts

    # print("验证通过：串行和并行 Counter 完全一致")


if __name__ == "__main__":
    main()
