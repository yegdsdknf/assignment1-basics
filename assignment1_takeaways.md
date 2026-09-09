# CS336 Assignment 1 重大收获梳理

> 本文从 tokenizer/BPE、模型与训练、测试验证、工程方法四个方面回顾 Assignment 1。
> 证据来源：`cs336_basics/` 源码、`tests/` 测试、`answer.md`、`artifacts/` 实测产物与 Git 历史。
> 测试基线：`uv run pytest -q` → **79 passed, 1 xfailed, 15.14s**；`uv run ruff check cs336_basics tests` → **All checks passed**。

## 0. 一句话总览

| # | 重大收获 | 仓库中的关键证据 |
|---|---|---|
| 1 | 字节级 BPE 把“任意 Unicode 文本”转化为“256 个基础字节 + 有限次 merge”，天然无 `<unk>`、可无损 roundtrip | `cs336_basics/bpe.py`、`tests/test_tokenizer.py` |
| 2 | BPE 训练必须做**增量 pair 统计 + 倒排索引**；tie-breaking 必须逐字节字典序确定 | 133KB 语料上 vocab=500 限时 **<1.5s**（`tests/test_train_bpe.py`） |
| 3 | 先 profile 再优化：完整 TinyStories BPE 训练 91.78s 中，并行预分词 61.71s，占 **67.2%**；单独预分词 CPU 1101%，完整训练被串行 merge 拉低到 790% | `artifacts/tinystories_bpe/train.log`、`test_pretoken/` |
| 4 | 多进程内存不能按“worker 数 × chunk 大小”估算；2.23GB 文件曾触发 ~8.3GiB RSS 的 OOM，换小 chunk + `finditer` 后峰值降至 ~189MiB | `answer.md` 第 85–136 行、`bpe.py` |
| 5 | 从零实现 Transformer 的每个细节都有坑：Linear 权重形状、RMSNorm 浮点精度、RoPE 广播、attention mask 约定、SwiGLU 门控 | `cs336_basics/model.py`、`tests/test_model.py` |
| 6 | AdamW 的核心是“解耦权重衰减 + 偏置修正 + 正确的原地操作边界”；梯度裁剪必须在 `backward()` 与 `step()` 之间 | `cs336_basics/optimizer.py`、`answer.md` 第 217–518 行 |
| 7 | 消融实验的可信度来自实验设计而不是单次运气：固定 token 预算、3 seeds、固定 `eval_seed=2026`、成对差值、主动写局限性 | `run_ablation.sh`、`artifacts/experiments/assignment1_ablations_20260813/` |
| 8 | 结论：本实验范围内 **Post-Norm + SwiGLU** 验证质量最好；SwiGLU 的优势**不能**由参数量解释 | 消融报告第 64–69、97–113 行 |

---

## 1. Unicode 与 Tokenizer：字节级 BPE 的设计收获

1. **选择 UTF-8 训练 tokenizer 的核心是 ASCII 兼容。** ASCII 字符（0–127）在 UTF-8 中仍占 1 字节，英文语料不膨胀；字节级 BPE 的基础词表只有 256 个单字节 token，覆盖所有 Unicode，不需要 `<unk>`（`answer.md` 第 26–30 行；`bpe.py` 第 81–83 行）。

2. **`repr()` 与 `print()` 表示同一字符的不同视图。** `chr(0)` 是真实字符，`repr()` 显示 `'\x00'`，`print()` 显示“空白”；拼接后字符串长度 +1。这说明“看不见”不等于“不存在”，对文本清洗与 tokenizer 调试很重要（`answer.md` 第 1–24 行）。

3. **UTF-8 解码必须整段进行。** 逐字节 `bytes([b]).decode("utf-8")` 会把多字节字符（如 `你` 的 `0xe4`）拆散，抛 `UnicodeDecodeError`；`b"\xC0\x80"` 这类“过长编码”更是任何合法 Unicode 都不接受的两字节序列（`answer.md` 第 31–46 行）。

4. **字节串经 JSON 无损持久化要用 hex。** 词表/merges 中任意 bytes 都通过十六进制序列化（`train_tinystory.py` 第 12–13 行；`tokenizer.py` 第 89–96 行），`decode` 用 `errors="replace"` 兜底非法字节（`tokenizer.py` 第 179 行）。

5. **特殊 token 必须“先隔离、再统计、不参与 merge”。** 预分词统计阶段按长度降序匹配特殊 token，用 `finditer` 找边界，普通区间才跑 GPT-2 正则；`Tokenizer` 编码时同样用捕获组保留特殊 token。测试还覆盖了重叠特殊 token 的优先级（`<|endoftext|><|endoftext|>` 应整体命中一个 token）与特殊 token 后换行等边界（`tests/test_tokenizer.py` 第 248–380 行）。

6. **确定性 tie-breaking 是正确性的一部分，不是实现细节。** `max(pair_counts, key=lambda pair: (count, pair))` 在频率相同时取 bytes 字典序较大者；`tests/test_train_bpe.py` 要求 merges 与参考实现**逐项相等**，任何偏差都会直接失败（`bpe.py` 第 397–403 行）。

7. **BPE 的性能门槛由算法复杂度决定。** 若每轮合并后全量重数 pair，133KB 语料上 vocab=500 约需 3s，无法通过 1.5s 限制；本实现用“pair 全局频率 = 局部次数 × 语料频率”的初始化 + `apply_merge_incrementally` 只更新受影响词，并维护 pair→word 倒排索引（`bpe.py` 第 151–270 行）。

8. **并行分块切边界要看文档边界而不是字节中位。** `find_chunk_boundaries` 在猜测边界附近 4096 字节窗口内找 `<|endoftext|>`，保证切出的 chunk 起始于 ASCII 文档边界，`decode("utf-8")` 不会切断多字节字符（`bpe.py` 第 23–67 行）。

---

## 2. 并行预分词：Profile、OOM 排查与调参收获

### 2.1 最大瓶颈来自测量，而不是猜测

完整 TinyStories BPE 训练（12 workers / 64 chunks）GNU time 墙钟 **91.78s**，其中单独预分词 **61.71s（67.2%）**；预分词单独跑 CPU 利用率为 **1101%**，完整训练为 **790%**——串行 BPE merge 阶段限制了整体并行度。这是 Amdahl 定律在本作业中的直接体现（`artifacts/tinystories_bpe/train.log`；`test_pretoken/check_parallel.resources_12worker64chunk.txt`）。

### 2.2 OOM 的排查链

对 2,227,753,162 字节（约 2.08GiB）训练文件用 4 workers / 8 chunks 并行预分词时，远程连接断开。排查顺序是：

```text
SSH 断连 → 查系统日志发现 "Out of memory: Killed process (python)"
        → anon-rss: 8677712 kB（≈8.3 GiB）
        → 结论：不是网络问题，而是 Linux OOM Killer
```

这教会一个方法论：**远程训练中断先区分“会话被断”与“进程被杀”，用 `dmesg`/`/usr/bin/time -v` 拿量化证据**；`tmux` 只防普通断线，不防 OOM（`answer.md` 第 85–136 行）。

### 2.3 Python 多进程的内存模型

内存峰值不能按“worker 数 × chunk 大小”估算，还要叠加：

| 内存来源 | 原因 |
|---|---|
| `chunk_bytes` | worker 读入的原始二进制 |
| `chunk_text` | bytes 解码后的第二份数据 |
| `fragments` | `split()` 一次性创建的全部文档字符串 |
| pre-token `Counter` | tuple/bytes/字典项/整数的 Python 对象开销 |
| IPC 缓冲区 | worker 向父进程 pickle Counter 的副本 |
| `combined_counts` | 父进程持续汇总的结果 |

对应优化：更小 chunk、`del chunk_bytes`（当前代码在 64-chunk 配置下未显式保留该语句，但实测峰值已无压力）、用 `finditer` 替代 `split()`、不要默认 `maxtasksperchild=1`（实测会在 5/32 chunk 后出现 worker 替换阻塞）、每完成一个 chunk `flush=True` 输出进度以便区分“正常 / 卡住 / 被杀”（`answer.md` 第 100–120 行；`bpe.py` 第 345–348 行）。

### 2.4 2.23GB 预分词基准

| Workers | Chunks | 墙钟 | CPU | 峰值 RSS |
|---:|---:|---:|---:|---:|
| 2 | 32 | 215.27s | 199% | 364,152 KiB |
| 4 | 32 | 116.88s | 396% | 363,864 KiB |
| 8 | 32 | 69.67s | 790% | 364,160 KiB |
| 8 | 64 | 70.37s | 788% | 193,488 KiB |
| 12 | 64 | 61.71s | 1101% | 193,868 KiB |
| 16 | 64 | 58.79s | 1160% | 193,452 KiB |

关键判断（`answer.md` 第 138–215 行，已与 `test_pretoken/*.resources*.txt` 逐项核对）：

- 2→4→8 workers 加速比 1.84×、1.68×，扩展良好；12→16 workers 仅 1.05×（只快 2.92s），非自愿上下文切换却达 **116,516 次**——超过 12 逻辑线程后只剩调度竞争。
- 固定 8 workers，32→64 chunks 墙钟只慢约 1%，峰值 RSS 从约 355.6MiB 降到约 189.0MiB（**-47%**）——用几乎零性能损失换内存减半。
- 因此默认推荐 8w/64c，独占机器 12w/64c，chunk 数取 worker 数 4–8 倍。
- 注意：`answer.md` 的“默认 8/64”建议与当前 `check_parallel.py`/`train_tinystory.py` 的代码默认值 `12/64` 尚未统一。

### 2.5 数据编码与 memmap

原始 2.23GB 文本被预编码为 **541,229,348 个 uint16 token（1,082,458,696 字节，约 1.01GiB）**，编码速度 388,868 tokens/s；训练时 `np.memmap(mode="r")` 只读映射，不需要把数据读进内存（`data/tinystories_train.uint16.bin.json`；`train.py` 第 130–136 行）。`prepare_data.py` 还展示了流式文档切分、`.partial` 临时文件 + 原子改名、字节数断言等数据管线习惯。

### 2.6 Tokenizer 吞吐量外推

样本 9,080 字节 / 2,214 tokens，压缩率 **4.1012 bytes/token**；单进程 Python `encode()` 预热后吞吐约 **1,635,184 bytes/s**，外推编码 825GB Pile 需约 **140.15 小时（5.84 天）**。收获有二：一是外推要先预热、按“跑满时间”而非次数测吞吐；二是 825GB 是十进制 GB 还是 GiB 这种单位问题会直接改变结论（`answer.md` 第 58–84 行；`tokenizer_experiment.py` 第 53–81 行）。

---

## 3. Transformer 组件与数值细节的收获

1. **自定义 `Linear` 的权重形状必须是 `(d_out, d_in)`，前向用 einops 显式写维度语义**（`einsum(x, W, "... d_in, d_out d_in -> ... d_out")`）。显式维度名避免 batch/seq/head 混用（`model.py` 第 15–29 行）。

2. **参数初始化用截断正态而非普通正态。** `std = sqrt(2 / (in_features + out_features))`、`a=-3σ, b=3σ`；大模型参数很多，普通正态出现极端值的概率不可忽略（`model.py` 第 22–24、47 行）。

3. **RMSNorm 要先升 float32 再平方/开方。** 低精度平方可能溢出；计算完 `x / rms * weight` 再回到原 dtype（`model.py` 第 53–73 行）。

4. **RoPE 的 cos/sin 用 `register_buffer(..., persistent=False)`。** 它们随设备迁移但不进 state_dict，不是可学习参数；前向先转 float32，位置索引转 long，`cos.ndim < x.ndim` 时在 `-3` 位置插入 head 维广播，再按偶/奇维旋转并交错还原（`model.py` 第 143–208 行）。

5. **Scaled Dot-Product Attention 的 mask 约定是 bool，`True=可关注`，`False` 填 `-inf`。** `-inf` 经 softmax 变成 0；实现还校验 `Q[-1]==K[-1]`、`K[-2]==V[-2]`。softmax/cross_entropy 都通过“减最大值”保证数值稳定（`model.py` 第 211–270 行；`nn_utils.py` 第 6–21 行）。

6. **SwiGLU 是“SiLU 门控 × 线性变换”的乘积，不是换一个激活函数。** 默认 `d_ff = ceil(8/3 * d_model / 64) * 64`；消融中的参数匹配 SiLU FFN 把 `d_ff` 从 1344 提到 2016 才使两个 FFN 参数量相等（各 2,064,384）（`model.py` 第 80–131 行；`tests/test_ablations.py` 第 20–27 行）。

7. **Pre/Post/No-Norm 必须由同一模块树干净地表达。** no-norm 用 `nn.Identity()` 真正移除 RMSNorm（否则消融里会藏着少量参数）；post-norm 把残差放到 norm 内部；`TransformerLM` 超长序列直接报错而不是静默截断（`model.py` 第 369–468 行）。

8. **测试用参考实现 + numpy snapshot 把“数值正确性”变成硬门槛。** Linear/Embedding/RMSNorm/RoPE/SwiGLU/SDPA/MHA/TransformerBlock/LM 全部要 `atol≈1e-4` 级对齐；softmax 对 `x+100`、cross_entropy 对 1000 倍缩放仍要保持稳定（`tests/test_model.py`；`tests/test_nn_utils.py`）。

---

## 4. AdamW、数据加载与训练循环的收获

1. **AdamW 与 Adam+L2 不等价。** `parameter.mul_(1 - lr * weight_decay)` 直接衰减参数，不进入一阶/二阶矩；若把 `λθ` 加进梯度，会被动量与自适应缩放污染（`optimizer.py` 第 77–78 行；`answer.md` 第 270–290 行）。

2. **偏置修正必须“先 `step += 1` 再算 `1 - β**step`”。** 第一次更新 step=0 会让分母为 0；矩状态懒初始化为 `zeros_like`（`optimizer.py` 第 64–94 行）。

3. **`step()` 要整体 `@torch.no_grad()`；有 closure 时临时 `enable_grad()`。** 参数更新不应被 autograd 记录（`optimizer.py` 第 37–47 行）。

4. **原地操作边界是 AdamW 的经典陷阱。** `exp_avg_sq.sqrt()` 返回新张量、`div_()/add_()` 只改临时分母；误用 `sqrt_()` 会永久破坏二阶矩缓存（`optimizer.py` 第 93 行；`answer.md` 第 362–383 行）。

5. **梯度裁剪顺序固定：`zero_grad → forward → backward → clip → step`。** 裁剪范数用 `detach()` 计算，避免范数进入计算图；`zero_grad(set_to_none=True)` 更省显存（`train.py` 第 424–436 行；`nn_utils.py` 第 24–46 行）。

6. **余弦调度要精确覆盖 warmup、余弦、min_lr 三段。** `it=0` 返回 0、`it=warmup_iters` 返回 max_lr、越过周期保持 min_lr；测试用 25 个期望点精确校验（`optimizer.py` 第 106–131 行）。

7. **`get_batch` 的越界上界是 `len - context_length`（开区间），起点最大取 `len - context_length - 1`。** 广播生成 `x`/`y` 恰好错 1 位；测试用 1000 次采样验证起点分布落在期望 ±5σ（`data.py` 第 18–35 行；`tests/test_data.py`）。

8. **checkpoint 三件套缺一不可：`model_state_dict + optimizer_state_dict + iteration`。** 只存模型无法精确恢复训练；加载返回 iteration（`serialization.py` 第 9–27 行）。

9. **生成用 top-p 时要保留“第一个越过 top_p 的 token”，并重新归一化；`temperature=0` 走 argmax。** 自回归循环只回喂最近 `context_length` 个 token，不必每步重新编码全序列（`generate.py` 第 14–71 行）。

---

## 5. 消融实验：结论与实验方法双重收获

### 5.1 结论（4 变体 × 3 seeds，40.96M-token 预算）

| 变体 | Final val loss | Perplexity | Δ loss vs baseline | Tokens/s | 峰值显存 GiB |
|---|---:|---:|---:|---:|---:|
| baseline（Pre-Norm + SwiGLU） | 1.90582 ± 0.00583 | 6.72499 ± 0.03927 | — | 39,604 ± 499 | 3.992 |
| no_norm（SwiGLU） | 1.91874 ± 0.00179 | 6.81239 ± 0.01218 | +0.01292 | 43,358 ± 1,868 | 3.710 |
| post_norm（SwiGLU） | **1.88980 ± 0.00923** | **6.61826 ± 0.06125** | **-0.01601** | 42,501 ± 31 | 3.991 |
| silu_matched（Pre-Norm + SiLU，d_ff=2016） | 1.98206 ± 0.00385 | 7.25772 ± 0.02798 | +0.07624 | 42,764 ± 21 | 3.914 |

- **Post-Norm 在当前浅层 TinyStories 设置下取得最低验证损失**，三个 seed 方向一致；但不能外推到深层/长训练场景。
- **移除归一化稳定可训练、吞吐高约 9.5%、显存低 0.282GiB，但质量下降**——这是“质量换效率”的取舍点。
- **SwiGLU 的优势不能由参数量解释**：参数匹配的 SiLU FFN 仍明显更差（+0.07624，约 4.0%），三个 seed 差值高度一致，支持“门控结构本身有效”。

### 5.2 方法收获

1. **训练 seed 与评估 seed 分离。** 训练用 42/43/44，评估固定 `eval_seed=2026`；`evaluate` 保存并恢复 NumPy 随机状态，评估不“偷走”训练随机流（`train.py` 第 93–94、214–252 行）。
2. **以固定 token 预算（5000 × 8192 = 40.96M）而不是 wall-clock 作为比较终点**，避免吞吐量差异改变训练量（`run_ablation.sh`；消融报告第 47 行）。
3. **成对差值优于只看均值。** 同一 seed 内 `variant - baseline` 消除种子差异：post_norm 三个差值全部为负，no_norm/silu_matched 全部为正。
4. **CUDA 计时先 sync、峰值显存先 reset。** 异步 CUDA 不 sync 会漏掉计算时间（`runtime.py` 第 24–27 行；`train.py` 第 398–402 行）。
5. **W&B 曲线统一用 `tokens_processed` 做横轴**，不同 batch size 的实验才能对齐；config 里记录全部超参与随机种子（`train.py` 第 173–211 行）。
6. **主动写局限性、标注证据强度。** 报告明确区分“3 seeds 方向一致”与“统计显著性”，并承认吞吐量未随机化顺序、只有 TinyStories、只有 4 层、验证只采 20 batches（消融报告第 141–149 行）。
7. **汇总脚本先校验完整性再聚合。** 要求每个 run 存在且最后一步为 5000；图同时展示均值、±1 SD 与单 seed 点，防止用“最好的那个 seed”当结论（`summarize_ablations.py` 第 33–34 行；`plot_ablations.py`）。

---

## 6. 工程过程与实验纪律的收获

1. **分阶段增量验证。** 5MB fixture 验证并行与串行 Counter 相等 → 22MB 验证集测耗时/内存 → 2.23GB 训练集 1 worker 跑通 → 再测并行加速。大文件问题一上来就全量跑，会既难归因又容易 OOM（`answer.md` 第 121–130 行）。

2. **基准测试一次只改一个变量，并同时看多个指标。** 墙钟、CPU%、User/System、峰值 RSS、Swap、页缓存、非自愿上下文切换都要记录；单次结果只判趋势，严格对比用 3 次中位数（`answer.md` 第 209–215 行）。

3. **进度输出用 `flush=True`。** 长任务逐 chunk 打印已完成数量与唯一 pre-token 数，进程被 OOM 杀掉时日志能定位到最后一个完成点（`bpe.py` 第 345–348 行）。

4. **产物分层管理。** `.gitignore` 忽略大数据、checkpoint、原始 metrics 流、wandb 本地目录与缓存，但提交 config、摘要、图表、报告等“小清单”；checkpoint 用 SHA256 记录而非入库（`.gitignore`；`git ls-files artifacts`）。

5. **版本控制小步提交。** 消融工作拆成 `feat:`/`chore:` 系列提交（加 runner → 只存 final checkpoint → 汇总 → 曲线 → 结果图）；反面教材是 `2ae17c2 完成assignment1` 一次合入 13 个文件的大爆炸提交。当前 24 个 commit 整体可回溯（`git log --oneline`）。

6. **跟随上游更新 = 同时读 handout 与 code/test 的 changelog。** `CHANGELOG.md` 每条都并列 `handout:` 与 `code:`，且 code 侧经常包含 fixture/测试更新（如 AdamW bugfix 的 fixture），同步后应跑全套测试与 lint（`CHANGELOG.md` 第 5–27 行）。

---

## 7. 测试基线与质量门禁

- `uv run pytest -q`：**79 passed, 1 xfailed**，约 15.14s；唯一 xfailed 是 `test_encode_memory_usage`（课程预期的内存限制用例）。
- `uv run ruff check cs336_basics tests`：**All checks passed**。
- 测试覆盖要点：组件数值 snapshot、tokenizer 与 tiktoken 对齐、BPE merges 逐项一致与速度限制、AdamW 双实现等价、checkpoint 完整往返、数据采样统计检验、消融变体结构约束、实验脚本可复现性（`tests/`）。

---

## 8. 主代理交叉核验备注（已知不一致 / 待办）

这些不是“重大收获”，但会影响证据引用，记录如下：

1. `answer.md` 第 62 行称压缩率样本取自 **validation**，但 `tokenizer_experiment.py` 第 102 行实际读的是 **train** 文件；本次梳理以代码为准。
2. 吞吐量 `1,635,184 B/s` 与用四舍五入输入重算的约 `1,635,179 B/s` 有约 5 B/s 的舍入差；Pile 外推内部用的 `1,635,183.63` 属同量级口径混用，不影响“约 5.84 天”的结论。
3. `answer.md` 建议默认 `8 workers / 64 chunks`，但代码默认值仍是 `12 / 64`（`check_parallel.py`、`train_tinystory.py` 第 62–63 行）。
4. `answer.md` 提到的 `del chunk_bytes` 优化在当前 `count_pretokens_chunk` 中没有显式保留；当前 64-chunk 配置实测峰值约 189MiB，暂不构成问题，但文档与代码应统一。
5. `artifacts/experiments/tinystories_5k_20260806/`（seed42 最终验证 loss 1.868567）与消融实验的 baseline seed42（1.90228）同为“5000 步”但数字不同，说明两次运行之间代码/评估采样版本不同；**跨版本运行不可直接比较**，这本身就是“记录代码版本与 eval_seed”重要性的例证。
6. `artifacts/experiments/assignment1_ablations_20260813/experiment_report.md` 目前是唯一未跟踪文件，尚未提交。

## 总结

本次作业最有价值的收获不是某个模型跑出了多低的 loss，而是形成了一条完整的闭环：

```text
理解编码与算法（Unicode → BPE）
  → 用 profile 找瓶颈（预分词占 67.2%）
  → 用系统证据定位故障（OOM 而非断网）
  → 用受控基准做参数权衡（workers/chunks、内存 vs 时间）
  → 用数值测试守住正确性（79 passed）
  → 用可复现实验纪律回答架构问题（Post-Norm、SwiGLU）
  → 把结论的适用范围与局限性写清楚
```

这条“先测量、再优化、用证据说话”的闭环，是本次作业最大的收获。
