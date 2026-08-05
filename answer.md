# Problem (unicode1):  Understanding Unicode (1 point)
1. What Unicode character does chr(0) return?
    - '\x00'
2. How does this character’s string representation (\__repr__( ) ) differ from its printed
representation?
    - \__repr__() 负责告诉程序员“这个对象到底是什么”，通常带引号和转义符；print() 负责告诉用户“这个对象显示出来是什么样”。
3. What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
```
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

# Problem (unicode2):  Unicode Encodings (3 points)
1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
    - ASCII字符（0-127）在UTF-8中的编码与ASCII码完全一致，且只占1个字节。这种向后兼容的特性，让它比UTF-16、UTF-32等编码方式更为常用。
2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string
into a Unicode string. Why is this function incorrect? Provide an example of an input byte
string that yields incorrect results.
```>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
                    ~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
 ```
这个函数是把每个字节单独解码，这对于英文常常是有效的，但是对于中文这种单个字符对应多个字节的，单个字节无法完整表示，因而是错误的。正确应该`return bytestring.decode("utf-8")`对bytestring整个解码
3. Give a two-byte sequence that does not decode to any Unicode character(s).
    - b"\xC0\x80",用两个字节表示一个本来一个字节就能表示的字符

# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

## (a) Training results

使用完整 TinyStories 训练集、10,000 的最大词表大小、<|endoftext|> 特殊 token，以及 12 workers + 64 chunks 进行训练，墙钟时间为 91.78 秒，GNU time 报告的最大常驻内存为 194,176 KiB（约 189.6 MiB）。最长 token 是 15 字节的 b' accomplishment'；这是合理的，因为该词在 TinyStories 中较常见，并且 GPT-2 预分词会保留单词前的空格。

## (b) Profiling results

预分词是最耗时的阶段，独立测量约为 61.71 秒，占完整训练时间的约 67.2%；完整任务的平均 CPU 利用率为 790%，说明并行预分词有效，而串行 BPE merge 阶段限制了总体并行度。

# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

## (a) TinyStories compression ratio

使用固定随机种子 42，从 TinyStories validation 中随机抽取 10 篇文档，并在每篇文档末尾保留 `<|endoftext|>`。样本共包含 9,080 个 UTF-8 字节，编码后得到 2,214 个 token，因此 10K TinyStories Tokenizer 的压缩率为：

```text
9080 bytes / 2214 tokens = 4.1012 bytes/token
```

即每个 token 平均表示约 4.10 个原始 UTF-8 字节。

## (c) Tokenizer throughput and Pile estimate

在当前单进程 Python 实现上，仅测量 `encode()` 的编码时间，不包括加载 Tokenizer、读取语料和磁盘 I/O。预热后对上述样本重复编码 361 次，共耗时 2.0046 秒，测得吞吐量约为 1,635,184 bytes/s（1.635 MB/s）。

按 Pile 大小为十进制 825GB，即 `825 × 10^9` bytes 估算：

```text
825,000,000,000 bytes / 1,635,183.63 bytes/s
= 504,530.49 seconds
= 140.15 hours
= 5.84 days
```

因此，假设吞吐量保持不变且忽略磁盘 I/O，当前实现编码完整 Pile 预计需要约 140.15 小时，即 5.84 天。

# 学习经验：2.23GB TinyStories 并行预分词的 OOM 排查

## 1. 现象与证据

运行 `cs336_basics/check_parallel.py` 对 2,227,753,162 bytes（约 2.08 GiB）的 TinyStories 训练文件进行并行预分词时，远程连接突然断开。服务器约有 15 GiB 内存和 4 GiB Swap，系统日志记录：

```text
Out of memory: Killed process ... (python)
anon-rss: 8677712 kB
```

这说明断连的主要原因不是网络波动，而是 Python 进程占用约 8.3 GiB 常驻内存后触发了 Linux OOM Killer。

## 2. 根因

测试使用 4 个 worker 和 8 个 chunk，每个原始 chunk 约 266 MiB。实际峰值内存还会叠加以下对象：

| 内存来源 | 原因 |
| --- | --- |
| `chunk_bytes` | worker 读取的原始二进制数据 |
| `chunk_text` | bytes 解码后形成的第二份数据 |
| `fragments` | `special_pattern.split(text)` 一次创建的全部文档字符串 |
| pre-token `Counter` | tuple、bytes、字典项和频率整数的对象开销 |
| IPC 缓冲区 | worker 向父进程 pickle Counter 时产生的副本 |
| `combined_counts` | 父进程持续汇总全部 chunk 的结果 |

多进程内存不能简单按“worker 数乘以原始 chunk 大小”估算；字符串复制、Python 小对象和进程间通信会显著放大内存。

## 3. 优化措施

1. 将配置改为 `num_workers=2, num_chunks=32`，使单个 chunk 降至约 66 MiB；若仍有压力，先用 1 个 worker 和 32 个 chunk 验证完整流程。
2. 在 `chunk_bytes.decode()` 后立即执行 `del chunk_bytes`，避免预分词期间同时持有 bytes 和 str。
3. 用 `special_pattern.finditer()` 确定文档边界，并通过 `PAT.finditer(text, start, end)` 处理原字符串区间，避免 `split()` 创建完整 fragments 列表。
4. 不使用 `maxtasksperchild=1` 作为默认配置。实测该设置会在完成 `5/32` 个 chunk 后出现 worker 替换阻塞：内存充足、没有 OOM，但父子进程 CPU 使用率都很低。若确实需要定期回收 worker，可另行验证 `spawn` 启动方式配合较大的 `maxtasksperchild`（如 4）。
5. 每完成一个 chunk 后使用 `flush=True` 输出进度，以区分正常计算、卡住和进程被系统杀死。

## 4. 分阶段验证

| 阶段 | 输入 | 配置 | 目标 |
| --- | --- | --- | --- |
| 1 | 5MB fixture | 2 workers / 4 chunks | 串行与并行 Counter 完全相等 |
| 2 | 22MB validation | 2 workers / 8 chunks | 测量耗时和峰值内存 |
| 3 | 2.23GB train | 1 worker / 32 chunks | 验证完整文件能够稳定完成 |
| 4 | 2.23GB train | 2 workers / 32 chunks | 测量实际并行加速效果 |

使用 `/usr/bin/time -v uv run python -m cs336_basics.check_parallel` 记录 `Maximum resident set size`，并通过 `free -h`、`ps` 监控父进程和 worker 的 RSS。`tmux` 可以防止普通 SSH 断线终止会话，但不能防止 OOM。

## 5. 额外发现与结论

`train_bpe()` 中还存在一处独立问题：取得并行 `pretoken_counts` 后，又无条件读取完整文件并执行串行预分词。它不影响只调用 `count_pretokens_parallel()` 的 `check_parallel.py`，但会在完整 BPE 训练时覆盖并行结果并提高峰值内存，需要删除重复逻辑。

核心经验是：多进程可以缩短 CPU 计算时间，却会放大内存占用。处理大文件时应先用小数据验证一致性，再逐级扩大规模，并同时考虑 bytes 到 str 的复制、Python 容器开销、序列化副本和父进程汇总结果。

# 2.23GB TinyStories 并行预分词配置基准与选择

## 1. 测试环境与统计方法

- CPU：AMD Ryzen 5 9600X，6 个物理核心、12 个逻辑线程（SMT）。
- 输入：TinyStoriesV2-GPT4 训练集，约 2.23GB。
- 命令：`/usr/bin/time -v uv run python -m cs336_basics.check_parallel`。
- 所有实验均正常结束（`Exit status: 0`），且没有使用 Swap。

`time -v` 的 `Maximum resident set size` 不能视为所有 worker 同时占用内存的总和，更适合比较单进程/单任务的内存峰值。若要观察进程组的实时总内存，需要通过 `ps` 汇总父进程和所有 worker 的 RSS。

## 2. 实验结果

| Workers | Chunks | 墙钟耗时 | CPU | User 时间 | System 时间 | 峰值 RSS | 非自愿切换 | 文件系统输入 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 32 | 215.27 秒 | 199% | 425.61 秒 | 3.74 秒 | 364152 KiB（355.6 MiB） | 1327 | 22504 |
| 4 | 32 | 116.88 秒 | 396% | 460.08 秒 | 3.66 秒 | 363864 KiB（355.3 MiB） | 1053 | 264 |
| 8 | 32 | 69.67 秒 | 790% | 546.31 秒 | 4.71 秒 | 364160 KiB（355.6 MiB） | 1033 | 0 |
| 8 | 64 | 70.37 秒 | 788% | 550.47 秒 | 4.60 秒 | 193488 KiB（189.0 MiB） | 1374 | 112688 |
| 12 | 64 | 61.71 秒 | 1101% | 673.41 秒 | 6.36 秒 | 193868 KiB（189.3 MiB） | 89207 | 0 |
| 16 | 64 | 58.79 秒 | 1160% | 676.18 秒 | 6.22 秒 | 193452 KiB（188.9 MiB） | 116516 | 0 |

部分实验的文件系统输入量不同，说明操作系统页缓存状态并不完全一致。因此单次结果主要用于判断趋势；严格对比时，应让同一配置运行 3 次并取墙钟耗时中位数，无需主动清理系统缓存。

## 3. Worker 数量的选择

| 对比 | 加速比 | 耗时下降 | 结论 |
| --- | ---: | ---: | --- |
| 2 → 4（32 chunks） | 1.84× | 45.7% | 扩展效果很好，4 个 worker 基本满载 |
| 4 → 8（32 chunks） | 1.68× | 40.4% | 仍有明显收益，8 workers 的 CPU 利用率接近上限 |
| 8 → 12（64 chunks） | 1.14× | 12.3% | 继续加速，但 CPU 开销和调度竞争明显增加 |
| 12 → 16（64 chunks） | 1.05× | 4.7% | 仅节省 2.92 秒，已进入明显的收益递减区间 |

9600X 有 6 个物理核心和 12 个逻辑线程。6 workers 基本对应物理核心数，适合强调资源效率、低功耗和系统响应，但本轮尚未进行同条件实测；8 workers 使用部分 SMT 线程，在速度、稳定性和交互体验之间较均衡；12 workers 对应全部逻辑线程，适合机器空闲时追求吞吐量；超过 12 workers 会形成过度订阅。

12-worker 实验平均使用约 11.01 个逻辑 CPU，16-worker 实验也只提高到约 11.60 个逻辑 CPU。16 workers 的非自愿上下文切换达到 116516 次，却只比 12 workers 快 2.92 秒，说明额外进程主要增加了调度竞争。

## 4. Chunk 数量的选择

| Chunks | 单个 chunk 平均大小 | 8 workers 时每 worker 平均任务数 | 特点 |
| ---: | ---: | ---: | --- |
| 32 | 约 71 MiB | 4 | 调度和 IPC 次数较少，但单任务内存较高 |
| 64 | 约 36 MiB | 8 | 负载均衡更好，单任务内存明显降低 |

固定 8 workers 时，32 chunks 耗时 69.67 秒，64 chunks 耗时 70.37 秒，只差 0.70 秒（约 1%），性能可视为基本相同；与此同时，记录到的峰值 RSS 从约 355.6 MiB 降至约 189.0 MiB，下降约 47%。两次测试的页缓存状态不同，不能认定 32 chunks 确实快 0.70 秒，但可以确认 64 chunks 以几乎没有性能损失的代价显著降低了单任务内存峰值。

对于 16 workers，32 chunks 平均每个 worker 只有 2 个任务，而 64 chunks 平均有 4 个任务，后者还能改善尾部负载不均。因此 64 chunks 更适合作为当前 2.23GB 数据集的默认值。

## 5. 最终建议

| 使用场景 | 推荐配置 | 原因 |
| --- | --- | --- |
| 默认开发与测试 | `8 workers + 64 chunks` | CPU 利用率约 99%，内存较低，上下文切换少 |
| 独占机器、追求吞吐量 | `12 workers + 64 chunks` | 利用 12 个逻辑线程，比 8 workers 再缩短约 8.66 秒 |
| 同时使用 VS Code、浏览器等程序 | `6～8 workers + 64 chunks` | 为 Windows、WSL 和交互程序保留资源 |
| 只追求单次最低墙钟时间 | `16 workers + 64 chunks` | 当前单次结果最快，但仅比 12 workers 快 2.92 秒 |

> 默认推荐 `num_workers=8, num_chunks=64`；机器空闲并只运行批处理时可使用 `num_workers=12, num_chunks=64`。不建议默认使用 16 workers，因为它已经超过 9600X 的 12 个逻辑线程，额外收益很小且调度竞争显著。

```python
# 默认配置：兼顾速度、内存和系统响应
num_workers = 8
num_chunks = 64
```

```python
# 独占机器执行批处理：优先吞吐量
num_workers = 12
num_chunks = 64
```

## 6. 可复用的实验方法

1. 调整并行参数时一次只改变一个变量，例如固定 64 chunks 后比较不同 worker 数。
2. 每组配置至少运行 3 次并取耗时中位数，记录页缓存状态。
3. 同时比较墙钟时间、CPU 总时间、上下文切换、Swap 和实时总 RSS，不能只看某一个指标。
4. worker 数通常不应显著超过逻辑 CPU 数；chunk 数可以先取 worker 数的 4～8 倍，再根据内存和调度开销调整。
5. 对大文件优先通过较小 chunk 控制单任务内存，再用基准测试确认调度开销是否可接受。

# 额外学习笔记：AdamW 原理与 `optimizer.py` 代码解析

## 1. 优化器在做什么

神经网络训练可以简化成下面的循环：

```text
输入数据 → 前向计算 → loss → backward() 计算梯度
                              ↓
                    optimizer.step() 修改参数
```

`loss.backward()` 只负责计算每个参数的 `parameter.grad`，不会主动修改参数。优化器负责根据梯度判断参数往哪个方向移动、移动多远。

可以把训练想象成在雾中下山：

- 参数是当前位置；
- loss 是当前海拔；
- 梯度指向最陡的上坡方向，因此负梯度是下坡方向；
- 学习率决定一步迈多大；
- AdamW 还会记录过去的方向和坡度，为不同参数调整步幅，并抑制参数无限变大。

这个比喻不能描述高维优化的全部细节，但能说明 AdamW 为什么不仅使用当前梯度，还要保存历史状态。

## 2. 从 SGD 到 AdamW

| 方法 | 核心想法 | 主要作用 |
| --- | --- | --- |
| SGD | 直接沿负梯度更新 | 最基本的参数优化 |
| Momentum | 累积历史梯度方向 | 减少震荡，强化持续一致的方向 |
| RMSProp | 用历史梯度平方调整步幅 | 适应不同参数的梯度尺度 |
| Adam | 同时使用一阶矩和二阶矩 | 结合动量与自适应学习率 |
| AdamW | Adam 加解耦权重衰减 | 更合理地限制参数变大 |

AdamW 中的 W 指 weight decay。它的自适应更新主体仍然是 Adam，关键改进是把权重衰减从梯度矩估计中分离出来。

## 3. 代码变量与数学符号

| 符号 | 代码名称 | 含义 |
| --- | --- | --- |
| `θ` | `parameter` | 当前模型参数 |
| `g_t` | `gradient` | 第 t 步的当前梯度 |
| `m_t` | `exp_avg` | 梯度的一阶指数移动平均 |
| `v_t` | `exp_avg_sq` | 梯度平方的指数移动平均 |
| `β₁` | `beta1` | 一阶矩的历史保留比例，通常为 0.9 |
| `β₂` | `beta2` | 二阶矩的历史保留比例，通常为 0.999 |
| `α` | `lr` | 学习率 |
| `ε` | `eps` | 防止除零的极小数 |
| `λ` | `weight_decay` | 权重衰减强度 |
| `t` | `state["step"]` | 当前参数已更新的次数 |

## 4. AdamW 的更新原理

### 4.1 解耦权重衰减

```python
parameter.mul_(1 - lr * weight_decay)
```

对应：

```text
θ ← (1 - αλ)θ
```

参数每一步都会缩小一点。例如 `lr=0.001`、`weight_decay=0.01` 时，每一步乘以 `0.99999`。单次变化很小，但长期训练中可以抑制参数持续增大。

传统 L2 正则常把 `λθ` 加到梯度：

```text
g ← g + λθ
```

在普通 SGD 中，这和权重衰减非常接近；但 Adam 还会对梯度进行动量累计和自适应缩放，所以两者不再等价。AdamW 直接衰减参数，不让衰减项进入一阶矩和二阶矩，这就是“解耦”。

### 4.2 更新一阶矩

```python
exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
```

```text
m_t = β₁m_(t-1) + (1 - β₁)g_t
```

一阶矩是近期梯度方向的平滑平均。连续多步方向一致时，它会积累这个方向；某一步梯度突然抖动时，历史平均能减弱噪声影响。

`mul_`、`add_` 末尾的下划线表示原地修改，代码直接更新优化器保存的状态张量。

### 4.3 更新二阶矩

```python
exp_avg_sq.mul_(beta2).addcmul_(
    gradient,
    gradient,
    value=1 - beta2,
)
```

```text
v_t = β₂v_(t-1) + (1 - β₂)g_t²
```

`addcmul_(a, b, value=c)` 表示：

```text
当前张量 += c × a × b
```

此处 `a`、`b` 都是梯度，所以加入梯度平方。二阶矩反映梯度的典型尺度：

- 梯度长期较大的参数，其分母会变大，更新步幅被压小；
- 梯度长期较小的参数，其相对更新步幅会更大。

因此 Adam 能给不同参数提供不同的自适应步幅。

### 4.4 为什么需要偏置修正

第一次使用参数时，代码把两个矩都初始化为零：

```python
state["exp_avg"] = torch.zeros_like(parameter)
state["exp_avg_sq"] = torch.zeros_like(parameter)
```

若 `β₁=0.9`，第一步的一阶矩只有 `0.1g₁`，明显偏向初始值 0。代码使用：

```python
bias_correction1 = 1 - beta1**step
bias_correction2 = 1 - beta2**step
```

得到修正后的矩：

```text
m_hat_t = m_t / (1 - β₁^t)
v_hat_t = v_t / (1 - β₂^t)
```

因此必须先执行 `state["step"] += 1`，再计算修正项。第一次真正更新时 `step` 应为 1；若使用 0，修正分母会变成 0。

### 4.5 参数更新

```python
denominator = (
    exp_avg_sq.sqrt()
    .div_(math.sqrt(bias_correction2))
    .add_(eps)
)
step_size = lr / bias_correction1

parameter.addcdiv_(
    exp_avg,
    denominator,
    value=-step_size,
)
```

整体等价于：

```text
θ ← θ - α × m_hat_t / (sqrt(v_hat_t) + ε)
```

`addcdiv_(a, b, value=c)` 表示 `parameter += c × a / b`。由于 `value` 是负的，所以这是参数减法。

这里 `exp_avg_sq.sqrt()` 没有下划线，会先创建一个新张量。后面的 `div_()` 和 `add_()` 只修改这个临时分母，不会破坏保存在 `state["exp_avg_sq"]` 中的二阶矩。若误用 `sqrt_()`，后续步骤将基于被开过平方根的错误状态继续计算。

## 5. `optimizer.py` 逐段解析

### 5.1 继承 `Optimizer`

```python
class AdamW(Optimizer):
```

继承 PyTorch 的 `Optimizer` 后，可以复用：

- `param_groups` 参数组；
- `state` 参数状态；
- `zero_grad()`；
- `state_dict()` 和 `load_state_dict()`；
- checkpoint 保存与恢复能力。

构造函数先检查 `lr`、`eps`、`betas` 和 `weight_decay` 的取值，再把它们放入 `defaults`。调用 `super().__init__(params, defaults)` 后，PyTorch 会建立参数组。参数组允许不同层使用不同学习率或权重衰减。

### 5.2 为什么 `step()` 使用 `@torch.no_grad()`

参数更新本身不应进入 autograd 计算图，否则会增加内存并形成错误的高阶依赖：

```python
@torch.no_grad()
def step(...):
```

若调用者传入 `closure`，closure 需要重新执行可求导计算，因此临时恢复梯度记录：

```python
with torch.enable_grad():
    loss = closure()
```

普通语言模型训练通常不需要 closure，但支持它可以保持标准优化器接口。

### 5.3 参数组、空梯度和稀疏梯度

```python
for group in self.param_groups:
    for parameter in group["params"]:
```

外层遍历不同超参数配置，内层遍历实际参数。若 `parameter.grad is None`，说明参数被冻结、未参与本次计算，或还没有反向传播，应直接跳过。

当前实现拒绝稀疏梯度，因为普通 AdamW 的逐元素状态更新不能直接套用到稀疏梯度。

### 5.4 每个参数的 `state`

```text
state["step"]       此参数已更新的次数
state["exp_avg"]    与参数同形状的一阶矩
state["exp_avg_sq"] 与参数同形状的二阶矩
```

状态采用懒初始化：只有某个参数第一次真正拥有梯度时才创建。不同参数可能参与不同计算，所以它们各自保存独立步数。

AdamW 需要为每个参数额外保存两个同形状张量，仅优化器状态通常约为模型参数内存的两倍；训练总内存还包括参数、梯度和前向激活。

## 6. 一次更新的数值例子

假设只有一个标量参数：

```text
θ=10，g=2，lr=0.1
β₁=0.9，β₂=0.999，weight_decay=0.01
```

忽略极小的 `eps`，第一步为：

1. 权重衰减：`θ = 10 × (1 - 0.1×0.01) = 9.99`。
2. 一阶矩：`m₁ = 0.9×0 + 0.1×2 = 0.2`。
3. 二阶矩：`v₁ = 0.999×0 + 0.001×2² = 0.004`。
4. 偏置修正：`m_hat₁=2`，`v_hat₁=4`。
5. Adam 更新：`θ = 9.99 - 0.1×2/sqrt(4) = 9.89`。

这说明权重衰减和梯度更新是两条互不污染的路径；偏置修正则让第一步的矩估计恢复到合理尺度。

## 7. 在训练循环中的位置

```python
optimizer.zero_grad()

logits = model(inputs)
loss = cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    targets.reshape(-1),
)

loss.backward()
gradient_clipping(model.parameters(), max_l2_norm=1.0)
optimizer.step()
```

顺序是：

1. 清除上一轮梯度；
2. 前向计算 loss；
3. `backward()` 产生当前梯度；
4. 梯度裁剪限制整体范数；
5. `step()` 用裁剪后的梯度更新参数。

PyTorch 默认累积梯度。忘记 `zero_grad()` 会让不同 batch 的梯度叠加；除非有意进行梯度累积，否则通常是错误。

## 8. 常见错误与快速复习

| 常见错误 | 后果 |
| --- | --- |
| 第一次更新使用 `step=0` | 偏置修正分母为 0 |
| 每步重新初始化矩状态 | 丢失历史信息，Adam 失去意义 |
| 把 `weight_decay × parameter` 加进梯度矩 | 变成 Adam 加 L2，而非解耦 AdamW |
| 忘记 `@torch.no_grad()` | 更新操作被 autograd 记录 |
| 不跳过 `grad is None` | 错误处理冻结或未参与计算的参数 |
| 对 `exp_avg_sq` 原地开平方 | 永久破坏二阶矩状态 |
| 忘记 `zero_grad()` | 不同 batch 的梯度意外累积 |

AdamW 可以概括为三件事：

1. 一阶矩平滑梯度方向；
2. 二阶矩为每个参数调整步幅；
3. 解耦权重衰减直接缩小参数。

主动回忆：

1. 为什么一阶矩和二阶矩需要偏置修正？
2. 为什么 Adam 中的 L2 正则与解耦权重衰减不等价？
3. 为什么 `exp_avg_sq.sqrt()` 不会破坏二阶矩缓存？
4. 为什么梯度裁剪应位于 `backward()` 和 `step()` 之间？

对应测试命令：

```bash
uv run pytest tests/test_optimizer.py -k test_adamw -v
```
