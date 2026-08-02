import json
from collections.abc import Iterable, Iterator
from os import PathLike

import regex as re
from cs336_basics.bpe import PAT, merge_pair


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        # 使用副本，避免追加特殊 token 时修改调用者传入的词表。
        self.vocab = dict(vocab)
        self.merges = list(merges)

        # bytes -> token ID，用于编码阶段。
        self.token_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in self.vocab.items():
            if token_bytes in self.token_to_id:
                raise ValueError(f"词表中存在重复 token：{token_bytes!r}")

            self.token_to_id[token_bytes] = token_id

        # 相邻 pair -> merge 优先级。rank 越小，越先应用。
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for rank, pair in enumerate(self.merges):  # enumerate会产生序号对应rank
            if pair in self.merge_ranks:
                raise ValueError(f"merges 中存在重复 pair：{pair!r}")

            self.merge_ranks[pair] = rank

        # 去重并保持用户传入的原始顺序。
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        # 字典的键不能重复,fromkeys把列表元素作为字典键，所以可以去重,同时字典保留顺序
        self.special_to_id: dict[str, int] = {}

        # 不能直接使用 len(vocab)，因为 token ID 不一定连续。
        next_token_id = max(self.vocab, default=-1) + 1

        for special_token in self.special_tokens:
            if not special_token:
                raise ValueError("特殊 token 不能为空字符串")

            token_bytes = special_token.encode("utf-8")
            # special_token存在
            token_id = self.token_to_id.get(token_bytes)

            # 如果特殊 token 不在词表中，则追加到词表。
            if token_id is None:
                token_id = next_token_id
                next_token_id += 1

                self.vocab[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id

            self.special_to_id[special_token] = token_id

        # 长特殊 token 必须优先匹配，解决特殊 token 重叠问题。
        ordered_special_tokens = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        if ordered_special_tokens:
            alternatives = "|".join(
                re.escape(token)  # 转义特殊字符
                for token in ordered_special_tokens
            )

            # 捕获组使 re.split() 保留匹配到的特殊 token。
            self.special_pattern = re.compile(f"({alternatives})")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """从训练脚本生成的十六进制 JSON 文件加载 tokenizer。"""
        with open(vocab_filepath, encoding="utf-8") as file:
            serialized_vocab = json.load(file)

        with open(merges_filepath, encoding="utf-8") as file:
            serialized_merges = json.load(file)

        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in serialized_vocab.items()}
        merges = [
            (
                bytes.fromhex(left_hex),
                bytes.fromhex(right_hex),
            )
            for left_hex, right_hex in serialized_merges
        ]

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def _encode_pretoken(self, raw_bytes: bytes) -> list[int]:
        """对一个 pre-token 应用已经训练好的 BPE merges。"""
        if not raw_bytes:
            return []

        symbols = tuple(bytes([byte_value]) for byte_value in raw_bytes)

        while len(symbols) >= 2:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            for pair in zip(symbols, symbols[1:]):
                rank = self.merge_ranks.get(pair)

                # 当前 pair 不在训练得到的 merges 中。
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            # 当前符号序列已经没有可以继续应用的 merge。
            if best_pair is None:
                break

            symbols = merge_pair(symbols=symbols, pair=best_pair)

        return [self.token_to_id[symbol] for symbol in symbols]

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """编码不包含特殊 token 的普通文本片段。"""
        token_ids: list[int] = []

        for match in PAT.finditer(text):
            raw_bytes = match.group().encode("utf-8")

            token_ids.extend(self._encode_pretoken(raw_bytes))

        return token_ids

    def encode(self, text: str) -> list[int]:
        """将文本编码成 token ID 列表。"""
        if not text:
            return []

        # 没有配置特殊 token 时，直接编码全部文本
        if self.special_pattern is None:
            return self._encode_ordinary_text(text)

        token_ids = []

        # special_pattern 包含捕获组，所以特殊 token 会保留在结果中。
        fragments = self.special_pattern.split(text)

        for fragment in fragments:
            if not fragment:
                continue

            special_token_id = self.special_to_id.get(fragment)

            if special_token_id is not None:
                # 特殊 token 不进行预分词和 BPE merge。
                token_ids.append(special_token_id)
            else:
                token_ids.extend(self._encode_ordinary_text(fragment))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐段编码文本，避免一次性将整个输入加载到内存。"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """将 token ID 序列解码成文本。"""
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)

        return token_bytes.decode("utf-8", errors="replace")
