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
