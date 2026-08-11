import math
import torch
from einops import einsum, rearrange
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        # W 的形状必须是 (d_out, d_in)
        self.weight = nn.Parameter(
            torch.empty(  # 分配内存，但未初始化
                out_features, in_features, device=device, dtype=dtype
            )
        )

        std = math.sqrt(2 / (in_features + out_features))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        # 普通正态分布理论上可能采样出非常大的极端值。虽然概率很低，但大模型包含大量参数，极端值出现的可能性会增加。
        # 截断后可以避免初始化权重中出现异常大的数。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # 词表大小
        embedding_dim: int,  # token向量维度
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # 每个隐藏维度对应一个可学习的缩放参数。
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        # 低精度张量平方可能溢出，因此先提升到 float32。
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            # 取约 8/3 * d_model，并向上调整为 64 的倍数。
            approximate_d_ff = 8 * d_model / 3
            d_ff = math.ceil(approximate_d_ff / 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff

        # W1、W3：d_model -> d_ff
        # W2：d_ff -> d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activate = silu(self.w1(x))
        gate = self.w3(x)

        return self.w2(activate * gate)


#     """
# 这里的 w1 和 w3 是两条并行分支：w1和silu充当门控值，w3去发现新特征，两者逐元素相乘
#              ┌─ W1 ─ SiLU ─┐
# 输入 x ──────┤              × ─ W2 ─ 输出
#              └─ W3 ────────┘
#     """


class SiLUFFN(nn.Module):
    """用于消融实验的普通 SiLU 前馈网络。"""

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,  # query,key维度
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k 必须为偶数")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 每一对维度使用不同的旋转频率。
        dimension_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1 / (theta ** (dimension_indices / d_k))

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # angles.shape = (max_seq_len, d_k // 2)
        angles = torch.outer(positions, inv_freq)

        # 这些值不是可学习参数，但需要随 module 一起迁移设备。
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)

        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        Args:
            x:
                形状为 (..., sequence_length, d_k)。
            token_positions:
                每个 token 对应的位置，通常形状为
                (sequence_length,) 或 (..., sequence_length)。

        Returns:
            与 x 形状相同的旋转结果。
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(f"输入最后一维应为 {self.d_k}，实际为 {x.shape[-1]}")
        input_dtype = x.dtype
        x_float = x.to(torch.float32)

        token_positions = token_positions.to(device=self.cos_cache.device, dtype=torch.long)

        # 根据 token 的真实位置选出对应的 sin/cos。
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 如果 x 包含 head 维，而 positions 没有，则插入广播维度。
        #
        # 例如：
        # x:   (batch, heads, seq, d_k)
        # cos: (batch, seq, d_k/2)
        #
        # 调整成：
        # cos: (batch, 1, seq, d_k/2)
        while cos.ndim < x_float.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 将最后一维拆成相邻的偶数维和奇数维。
        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]

        rotate_even = x_even * cos - x_odd * sin
        rotate_odd = x_even * sin + x_odd * cos

        # 先恢复成 (..., d_k/2, 2)，再交错展开为 (..., d_k)。
        rotated = torch.stack((rotate_even, rotate_odd), dim=-1).flatten(start_dim=-2)

        return rotated.to(input_dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 这一步找到 x 在指定维度上的最大值。
    # 保留维度后，可以方便地进行广播减法
    max_value = torch.max(
        x,
        dim=dim,
        keepdim=True,
    ).values

    # 减去最大值不会改变结果，但可以避免大数经过 exp() 后变成 inf。
    shifted = x - max_value
    exp_values = torch.exp(shifted)

    return exp_values / torch.sum(exp_values, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        Q:
            (..., num_queries, d_k)
        K:
            (..., num_keys, d_k)
        V:
            (..., num_keys, d_v)
        mask:
            可广播到 (..., num_queries, num_keys) 的布尔张量。
            True 表示可以关注，False 表示屏蔽。

    Returns:
        (..., num_queries, d_v)
    """
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q 和 K 的最后一维必须相同")

    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K 和 V 的 num_keys 维必须相同")

    d_k = Q.shape[-1]

    # scores.shape = (..., num_queries, num_keys)
    # 计算Q、K相似度
    scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask 必须是布尔张量")

        # False 位置变成 -inf，使其 softmax 权重变成 0。
        scores = scores.masked_fill(~mask, float("-inf"))

    # 对每个 query，在全部 key 上归一化。
    attention_weights = softmax(scores, dim=-1)

    # output.shape = (..., num_queries, d_v)
    # 加权value
    output = einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

    return output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        if (theta is None) != (max_seq_len is None):
            raise ValueError("theta 和 max_seq_len 必须同时提供，或者同时为 None")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        参数：
            x: (..., seq_len, d_model)
            token_positions: (..., seq_len)

        返回：
            (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]
        # (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (..., num_heads, seq_len, d_head)
        q = rearrange(q, "... seq (heads d_head) -> ... heads seq d_head", heads=self.num_heads)

        k = rearrange(k, "... seq (heads d_head) -> ... heads seq d_head", heads=self.num_heads)

        v = rearrange(v, "... seq (heads d_head) -> ... heads seq d_head", heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # 加入 head 的广播维度：
            # (batch, seq) -> (batch, 1, seq)
            # rope_position = token_positions.unsqueeze(-2)

            # RoPE 只作用于 Q 和 K。
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # mask[i, j] == True 表示位置 i 可以关注位置 j。
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # (..., num_heads, seq_len, d_head)
        attended = scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal_mask)
        # (..., num_heads, seq_len, d_head)
        #     -> (..., seq_len, d_model)

        attended = rearrange(attended, "... heads seq d_head -> ... seq (heads d_head)")

        return self.output_proj(attended)


# 支持归一化位置和 FFN 类型消融的 Transformer Block。
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu",
    ):
        super().__init__()

        if norm_mode not in {"pre", "post", "none"}:
            raise ValueError(f"不支持的 norm_mode：{norm_mode}")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"不支持的 ffn_type：{ffn_type}")

        self.ffn_type = ffn_type
        self.norm_mode = norm_mode

        if norm_mode == "none":
            # Identity 不包含参数，确保 No-Norm 实验真正移除归一化层。
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            x = x + self.ffn(self.ln2(x))
            return x

        if self.norm_mode == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
            return x

        x = x + self.attn(x, token_positions)
        x = x + self.ffn(x)
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu",
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=rope_theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                    norm_mode=norm_mode,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        if norm_mode == "none":
            self.ln_final = nn.Identity()
        else:
            self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        seq_len = in_indices.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(f"序列长度 {seq_len} 超过 context_length {self.context_length}")

        x = self.token_embeddings(in_indices)

        token_positions = torch.arange(seq_len, device=in_indices.device)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)
