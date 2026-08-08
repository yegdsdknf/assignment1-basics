import regex as re
from collections import Counter, defaultdict
import multiprocessing as mp
import os
from os import PathLike
from typing import BinaryIO

type TokenBytes = bytes  # 一个基础或合并后的 BPE token
type TokenPair = tuple[TokenBytes, TokenBytes]  # 待统计、待合并的相邻 token 对
type PreToken = tuple[TokenBytes, ...]  # 一个预分词结果的当前符号序列
type PreTokenCounts = Counter[PreToken]  # pre-token 及其语料频率

type WordId = int  # 增量算法中 pre-token 的固定编号
type PairCounts = Counter[TokenPair]  # 全局 pair 加权频率
type PairToWordIds = dict[TokenPair, set[WordId]]  # pair 到相关 pre-token 的倒排索引
type WordSymbols = list[PreToken]
type WordFrequencies = list[int]  # 将pretoken_counts 转换为两个列表，便于多进程处理

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
END_OF_TEXT = "<|endoftext|>"


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretokens(
    text: str,
    special_tokens: list[str],
) -> PreTokenCounts:
    """按特殊 token 划分边界，并统计 GPT-2 预分词结果。"""

    counts: PreTokenCounts = Counter()

    def count_span(start: int, end: int) -> None:
        """直接在原始字符串的指定区间内执行预分词。"""
        for match in PAT.finditer(text, start, end):
            raw_bytes = match.group().encode("utf-8")

            symbols: PreToken = tuple(bytes([byte_value]) for byte_value in raw_bytes)

            counts[symbols] += 1

    if not special_tokens:
        count_span(0, len(text))
        return counts

    if any(token == "" for token in special_tokens):
        raise ValueError("特殊 token 不能为空字符串")

    # 较长 token 放在前面，正确处理存在前缀重叠的特殊 token。
    ordered_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    special_pattern = re.compile("|".join(re.escape(token) for token in ordered_tokens))

    fragment_start = 0

    # 只记录边界位置，不构造包含全部文档的 fragments 列表。
    for special_match in special_pattern.finditer(text):
        fragment_end = special_match.start()

        count_span(
            fragment_start,
            fragment_end,
        )

        # 跳过特殊 token，使其不参与普通 pair 统计。
        fragment_start = special_match.end()

    # 处理最后一个特殊 token 后的剩余文本。
    count_span(
        fragment_start,
        len(text),
    )

    return counts


# 将 Counter 转为固定 word ID
def assign_word_ids(
    pretoken_counts: PreTokenCounts,
) -> tuple[WordSymbols, WordFrequencies]:
    """为每种唯一 pre-token 分配稳定的整数 ID。"""

    word_symbols: WordSymbols = []
    word_frequencies: WordFrequencies = []

    for symbols, frequency in pretoken_counts.items():
        if frequency <= 0:
            raise ValueError("pre-token 频率必须为正数")

        # word_id 就是两个列表中的下标。
        word_symbols.append(symbols)
        word_frequencies.append(frequency)

    return word_symbols, word_frequencies


def count_word_pairs(symbols: PreToken) -> Counter[TokenPair]:
    """统计一个 pre-token 当前符号序列中的相邻 pair。"""
    return Counter(zip(symbols, symbols[1:]))


def initialize_pair_index(
    word_symbols: WordSymbols,
    word_frequencies: WordFrequencies,
) -> tuple[PairCounts, PairToWordIds]:
    """一次全量扫描，初始化 pair 频率和倒排索引。"""

    if len(word_symbols) != len(word_frequencies):
        raise ValueError("symbols 和 frequencies 长度必须一致")

    pair_counts: PairCounts = Counter()
    pair_to_word_ids: PairToWordIds = defaultdict(set)

    for word_id, symbols in enumerate(word_symbols):
        frequency = word_frequencies[word_id]

        # Counter 会正确计算同一 pair 在一个 pre-token 中出现多次的情况。
        local_pair_counts: Counter[TokenPair] = count_word_pairs(symbols)

        for pair, local_count in local_pair_counts.items():
            # 全局频率必须乘上该 pre-token 的语料频率。
            pair_counts[pair] += local_count * frequency

            # 倒排索引只记录这个 word 是否含有该 pair。
            pair_to_word_ids[pair].add(word_id)

    return pair_counts, pair_to_word_ids


def merge_pair(
    symbols: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """合并 symbols 中所有不重叠的指定相邻 token 对。"""

    left, right = pair
    result: list[bytes] = []

    index = 0

    while index < len(symbols):
        # 检查当前位置和下一个位置是否组成目标 pair
        is_match = index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right

        if is_match:
            # bytes 可以直接通过 + 拼接
            result.append(left + right)
            index += 2
        else:
            result.append(symbols[index])
            index += 1

    return tuple(result)


def apply_merge_incrementally(
    best_pair: TokenPair,
    word_symbols: WordSymbols,
    word_frequencies: WordFrequencies,
    pair_counts: PairCounts,
    pair_to_word_ids: PairToWordIds,
) -> None:
    # 必须复制，因为后续会修改 pair_to_word_ids 中的集合。
    affected_word_ids = list(pair_to_word_ids.get(best_pair, set()))

    for word_id in affected_word_ids:
        old_symbols = word_symbols[word_id]
        frequency = word_frequencies[word_id]

        old_local_counts = count_word_pairs(old_symbols)
        new_symbols = merge_pair(old_symbols, best_pair)

        if new_symbols == old_symbols:
            raise AssertionError(f"倒排索引过期：word {word_id} 不包含 {best_pair!r}")

        new_local_counts = count_word_pairs(new_symbols)

        # 包含消失、新增以及次数发生变化的所有 pair。
        changed_pairs = old_local_counts.keys() | new_local_counts.keys()

        for pair in changed_pairs:
            old_count = old_local_counts[pair]
            new_count = new_local_counts[pair]

            # 当前 pre-token 在语料中可能出现多次，因此要乘 frequency。
            delta = (new_count - old_count) * frequency

            if delta != 0:
                updated_count = pair_counts.get(pair, 0) + delta

                if updated_count < 0:
                    raise AssertionError(f"pair 频率变为负数: {pair!r}")

                if updated_count == 0:
                    pair_counts.pop(pair, None)
                else:
                    pair_counts[pair] = updated_count

            was_present = old_count > 0
            is_present = new_count > 0

            # 这个 word 不再包含该 pair。
            if was_present and not is_present:
                indexed_word_ids = pair_to_word_ids.get(pair)

                if indexed_word_ids is None or word_id not in indexed_word_ids:
                    raise AssertionError(f"倒排索引缺少 word {word_id}: {pair!r}")

                indexed_word_ids.remove(word_id)

                if not indexed_word_ids:
                    del pair_to_word_ids[pair]

            # 这个 word 新产生了该 pair。
            elif not was_present and is_present:
                pair_to_word_ids.setdefault(
                    pair,
                    set(),
                ).add(word_id)

        word_symbols[word_id] = new_symbols


def count_pretokens_chunk(
    task: tuple[
        str | PathLike[str],
        int,
        int,
        list[str],
    ],
) -> PreTokenCounts:
    """读取一个文件分块并统计其中的 pre-token。"""
    input_path, start, end, special_tokens = task

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8")

    return count_pretokens(
        chunk_text,
        special_tokens,
    )


def count_pretokens_parallel(
    input_path: str | PathLike[str],
    special_tokens: list[str],
    num_workers: int,
    num_chunks: int | None = None,
) -> PreTokenCounts:
    """并行处理文件分块并合并各 worker 的 pre-token 计数。"""
    if num_workers <= 0:
        raise ValueError("num_workers 必须为正数")

    if special_tokens != [END_OF_TEXT]:
        raise ValueError(f"并行分块只支持单一文档边界 {END_OF_TEXT!r}")

    if num_chunks is None:
        num_chunks = num_workers * 2

    split_token = END_OF_TEXT.encode("utf-8")
    with open(input_path, "rb") as file:  # 生成 chunk 任务
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=num_chunks,
            split_special_token=split_token,
        )
        tasks = [
            (
                input_path,
                start,
                end,
                special_tokens,
            )
            for start, end in zip(
                boundaries,
                boundaries[1:],
            )
        ]
        combined_counts: PreTokenCounts = Counter()  # 使用进程池汇总 Counter
        total_tasks = len(tasks)

        with mp.Pool(
            processes=num_workers,
        ) as pool:
            for completed, chunk_counts in enumerate(
                pool.imap_unordered(
                    count_pretokens_chunk,
                    tasks,
                ),
                start=1,
            ):
                combined_counts.update(chunk_counts)
                print(
                    f"已完成 {completed}/{total_tasks} 个 chunk，当前唯一 pre-token 数：{len(combined_counts)}",
                    flush=True,
                )

        return combined_counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 1,
    num_chunks: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    min_size = 256 + len(special_tokens)

    if vocab_size < min_size:
        raise ValueError(f"vocab_size 至少应为 {min_size}，因为必须容纳 256 个字节和全部特殊 token")
    if num_workers == 1:
        with open(input_path, encoding="utf-8") as file:
            text = file.read()

        pretoken_counts = count_pretokens(
            text,
            special_tokens,
        )
        del text
    else:
        pretoken_counts = count_pretokens_parallel(
            input_path=input_path,
            special_tokens=special_tokens,
            num_workers=num_workers,
            num_chunks=num_chunks,
        )

    # 1. 初始化全部单字节 token。
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    # 2. 按用户给定的顺序加入特殊 token。
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    word_symbols, word_frequencies = assign_word_ids(pretoken_counts)
    del pretoken_counts
    pair_counts, pair_to_word_ids = initialize_pair_index(word_symbols, word_frequencies)
    merges: list[TokenPair] = []

    # 3. 每轮增加一个新 token。
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        apply_merge_incrementally(
            best_pair=best_pair,
            word_symbols=word_symbols,
            word_frequencies=word_frequencies,
            pair_counts=pair_counts,
            pair_to_word_ids=pair_to_word_ids,
        )

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges
