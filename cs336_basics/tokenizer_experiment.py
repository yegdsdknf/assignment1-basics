from collections.abc import Iterator, Iterable
from pathlib import Path
import random
from cs336_basics.tokenizer import Tokenizer
import time
import json

END_OF_TEXT = "<|endoftext|>"
PILE_NUM_BYTES = 825 * 1_000_000_000


def iter_documents(input_path: Path, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """流式读取由特殊 token 分隔的文档。"""
    buffer = ""

    with input_path.open(encoding="utf-8") as file:
        while chunk := file.read(chunk_size):
            buffer += chunk
            fragments = buffer.split(END_OF_TEXT)

            # 最后一个片段可能是不完整文档，留到下一轮。
            for document in fragments[:-1]:
                if document:
                    yield document

            buffer = fragments[-1]

    if buffer:
        yield buffer


def sample_documents(documents: Iterable[str], sample_size: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)  # 创建了一个独立的随机数生成器,不会影响程序其他地方的随机操作
    samples: list[str] = []

    for document_count, document in enumerate(documents, start=1):
        if len(samples) < sample_size:
            samples.append(document)
            continue
        # 放满要求数量后，再随机替换
        replacement_index = rng.randrange(document_count)

        if replacement_index < sample_size:
            samples[replacement_index] = document

    if len(samples) < sample_size:
        raise ValueError(f"语料只有 {len(samples)} 个文档，无法抽取 {sample_size} 个")

    return samples


"""
为什么不直接使用 random.sample()？如果文档已经全部放在列表中，可以直接写：
random.sample(documents, sample_size)

但这里的 documents 可能是：iter_documents(input_path)
它是一个流式生成器。
如果先转成列表：
all_documents = list(iter_documents(input_path))
就会把所有文档装进内存。假设语料库有几百万篇文档，这可能占用大量内存。
蓄水池抽样只需要保存：sample_size 篇文档,而不需要保存全部文档。
"""


def calculate_compression_ratio(
    tokenizer: Tokenizer,
    documents: list[str],
) -> tuple[int, int, float]:
    # 保留文档边界，使实验更接近真正的数据编码过程。
    text = "".join(document + END_OF_TEXT for document in documents)

    num_bytes = len(text.encode("utf-8"))
    token_ids = tokenizer.encode(text)
    num_tokens = len(token_ids)

    if num_tokens == 0:
        raise ValueError("样本文本编码后没有产生 token")

    bytes_per_token = num_bytes / num_tokens

    return num_bytes, num_tokens, bytes_per_token


"""
测量吞吐量
只测量编码过程，不把以下时间包括进去：
加载词表；
读取文件；
随机抽样；
JSON 写入。
为了降低偶然误差，应先预热，然后持续测量至少约2秒：
"""


def benchmark_tokenizer(
    tokenizer: Tokenizer, documents: list[str], minimum_seconds: float = 2.0
) -> tuple[int, float, float]:
    text = "".join(document + END_OF_TEXT for document in documents)
    num_bytes = len(text.encode("utf-8"))

    # 预热，避免首次调用开销影响结果。
    for _ in range(3):
        tokenizer.encode(text)

    processed_bytes = 0
    num_iterations = 0
    start_time = time.perf_counter()  # 适合性能测试的高精度计时器。

    while True:
        # 不保存所有结果，避免测试本身不断占用内存。
        tokenizer.encode(text)

        processed_bytes += num_bytes
        num_iterations += 1

        elapsed_seconds = time.perf_counter() - start_time

        if elapsed_seconds >= minimum_seconds:
            break

    bytes_per_second = processed_bytes / elapsed_seconds

    return num_iterations, elapsed_seconds, bytes_per_second


"""
估算处理 Pile 的时间 Pile 大约是 825GB
"""


def estimate_pile_time(
    bytes_per_second: float,
) -> tuple[float, float, float]:
    seconds = PILE_NUM_BYTES / bytes_per_second
    hours = seconds / 3600
    days = hours / 24

    return seconds, hours, days


def main() -> None:
    tokenizer = Tokenizer.from_files(
        vocab_filepath=Path("artifacts/tinystories_bpe/vocab.json"),
        merges_filepath=Path("artifacts/tinystories_bpe/merges.json"),
        special_tokens=[END_OF_TEXT],
    )

    documents = sample_documents(
        documents=iter_documents(Path("data/TinyStoriesV2-GPT4-train.txt")),
        sample_size=10,
        seed=42,
    )

    num_bytes, num_tokens, bytes_per_token = calculate_compression_ratio(
        tokenizer=tokenizer,
        documents=documents,
    )

    iterations, elapsed_seconds, bytes_per_second = benchmark_tokenizer(
        tokenizer=tokenizer,
        documents=documents,
    )

    pile_seconds, pile_hours, pile_days = estimate_pile_time(bytes_per_second)

    metrics = {
        "sample_size": len(documents),
        "num_bytes": num_bytes,
        "num_tokens": num_tokens,
        "bytes_per_token": bytes_per_token,
        "benchmark_iterations": iterations,
        "benchmark_seconds": elapsed_seconds,
        "bytes_per_second": bytes_per_second,
        "megabytes_per_second": bytes_per_second / 1_000_000,
        "pile_estimated_seconds": pile_seconds,
        "pile_estimated_hours": pile_hours,
        "pile_estimated_days": pile_days,
    }

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
