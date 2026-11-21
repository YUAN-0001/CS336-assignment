# M1 报告：BPE 分词器（A1–A4、C1–C2 + 选做 B1–B3）

- 姓名/学号：**马立博 / 525033910041**
- 代码 commit：`f93493e`（"完成 M1：实现 B 版增量 BPE 优化 (B1-B3)"），其父提交 `cc779d0`（"完成 M1：实现 BPE 分词器的训练、编码与解码"）
- 环境：Windows 11 家庭中文版 10.0.26300；CPU Intel Core i7-14650HX；Python 3.13.5；regex 2026.7.19；pytest 8.3.4。仓库内 `.venv` 由 WSL 创建（`pyvenv.cfg` 的 `home=/usr/bin`，只有 `lib64 -> lib` 软链、没有 `Scripts/`），在 Windows 下 `uv run` 无法使用（`error: failed to remove file ... .venv\lib64: 拒绝访问。 (os error 5)`），所以全部实验都用系统 Python 直接 `python -m ...` 运行；控制台编码为 GBK，跑含 emoji 的输出前需设置 `PYTHONIOENCODING=utf-8`，否则抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f642'`。

## 1. 验收命令与结果

工作目录 `CS336-assignment1-git`，数据为仓库自带的 `tokenizer_lab_data/tiny.txt`（264 字节）。

文档里给的是 `uv run python -m tokenizer_lab_code.cli ...`；因为本机 `.venv` 是 WSL 创建的、`uv run` 无法使用（见上），实际以系统 Python 执行等价命令：

```bash
export PYTHONIOENCODING=utf-8
export PYTHONPATH=.
python -m tokenizer_lab_code.cli pretokenize --input tokenizer_lab_data/tiny.txt --output var/m1-report/counts.json
python -m tokenizer_lab_code.cli train --cache var/m1-report/counts.json --variant naive --vocab-size 280 --trace --output var/m1-report/tokenizer.json
python -m tokenizer_lab_code.cli encode --model var/m1-report/tokenizer.json --text "你好，hello world!🙂"
python -m lab_tools.check m1
```

实测结果：

| 命令 | 结果 |
| --- | --- |
| `pretokenize` | `{"unique_pretokens": 31, "pretoken_occurrences": 58, "input_seconds": 0.0009737}` |
| `train --variant naive --vocab-size 280 --trace` | `{"vocab_size": 280, "merges": 23, "input_seconds": 0.0008111, "training_seconds": 0.0018714}` |
| `encode` | `{"ids": [278, 273, 266, 270, 111, 279, 100, 33, 240, 159, 153, 130], "decoded": "你好，hello world!🙂"}`，roundtrip 一致 |
| `python -m lab_tools.check m1` | **49 passed in 0.97s**（含 `test_incremental.py` 的 9 个用例，单独跑亦为 `9 passed in 0.10s`） |

单次训练的毫秒级数字随进程启动与缓存状态有约 10–20% 抖动（同一命令另一轮测得 `input_seconds = 0.00257`、`training_seconds = 0.00203`），但 `merges`、词表内容、id 分配在任何一轮都完全一致。

## 2. 语料、目标/实际词表大小、前几轮合并

**语料**：`tokenizer_lab_data/tiny.txt`，9 行、264 字节、4 个以 `<|endoftext|>` 分隔的段落——英文（`the cat in the hat.` / `the dog and the cat played.`）、重复词（`hello hello world!`）、中文与 emoji（`你好，世界！你好，语言模型。🙂`）、以及专门构造的 `aaaa aaa aa a. banana banana bandana.`。预分词（GPT-2 风格正则）后得到 **31 个 unique pre-token、58 次出现**，平均每个 pre-token 出现约 1.9 次，是一个"小但覆盖了中英文、多字节、特殊 token、重叠字符"的语料。

**词表大小**：目标是 280。实际正好 280，构成为 `256` 个基础字节 + `1` 个特殊 token `<|endoftext|>`（id 256）+ `23` 个合并 token（id 257–279），合计 256 + 1 + 23 = 280。注意特殊 token 占用了 256 这个 id，因此第一轮合并的结果落在 257，而不是 256。

**前几轮合并**（`--trace` 的完整 23 轮，格式为 `merge_id / pair / 合并后 bytes / 频次`）：

| # | pair | 结果 | freq | id |
| --- | --- | --- | --- | --- |
| 0 | (h, e) | `he` | 10 | 257 |
| 1 | (t, he) | `the` | 7 | 258 |
| 2 | (a, n) | `an` | 7 | 259 |
| 3 | (a, t) | `at` | 6 | 260 |
| 4 | (a, a) | `aa` | 6 | 261 |
| 5 | (` `, the) | `" the"` | 5 | 262 |
| 6 | (\xef, \xbc) | `efbc` | 3 | 263 |
| 7 | (l, o) | `lo` | 3 | 264 |
| 8 | (l, lo) | `llo` | 3 | 265 |
| 9 | (he, llo) | `hello` | 3 | 266 |
| 10 | (c, at) | `cat` | 3 | 267 |
| 11 | (b, an) | `ban` | 3 | 268 |
| 12 | (an, a) | `ana` | 3 | 269 |
| 13 | (` `, w) | `" w"` | 3 | 270 |
| 14 | (` `, cat) | `" cat"` | 3 | 271 |
| 15 | (` `, ban) | `" ban"` | 3 | 272 |
| 16 | (efbc, 8c) | `"，"` | 2 | 273 |
| 17 | (e5, a5) | `e5a5` | 2 | 274 |
| 18 | (e5a5, bd) | `"好"` | 2 | 275 |
| 19 | (e4, bd) | `e4bd` | 2 | 276 |
| 20 | (e4bd, a0) | `"你"` | 2 | 277 |
| 21 | (`"你"`, `"好"`) | `"你好"` | 2 | 278 |
| 22 | (r, l) | `rl` | 2 | 279 |

几点值得注意：

- 高频结构先成形：`h+e → he` 频次 10 最高，于是第二轮才有 `t+he → the`；`the` 又参与第 5 轮的 `(" ", the)`，逐步长出 `" the"`。
- 合并结果是**字节串**，不必是合法 UTF-8。id 263 `efbc`、274 `e5a5`、276 `e4bd` 都是某个汉字的 UTF-8 编码前缀（`，` / `好` / `你` 的前两个字节），它们只是中间产物，靠后续轮次补齐成完整字符（273、275、277），最终 278 `你好` 是 6 字节、也是全表最长的合并 token。
- 语料里 `banana` 出现两次、`bandana` 一次，所以 `(b, an) → ban` 拿到 3；而 `ana` 是在 `ban` 之后才成形，说明合并顺序是"当前频次最大的对优先"，不是一次性按词频排序。
- 最后剩下的 `(r, l)` 来自 `world`：`w` 已被 `" w"` 吃掉之后，剩余的 `orl` 里 `r+l` 才凑够 2。

## 3. `aaa` 的统计与合并为何不同

`count_pairs` 统计的是**相邻对的出现次数**（一个 pair 在一个词里出现 k 次就算 k 次，再乘词频），而合并一次只消除**一处非重叠**的相邻对。所以两者天然不等：

```python
count_pairs([Word((b"a", b"a", b"a"), 3)])      # {(b"a", b"a"): 6}   3 个相邻位置 × 词频 3
merge_sequence((b"a", b"a", b"a"), (b"a", b"a"))          # (b"aa", b"a")    只消掉位置 0–1 这一处
merge_sequence((b"a", b"a", b"a", b"a"), (b"a", b"a"))   # (b"aa", b"aa")   4 个 a 消掉两处
```

`aaa` 里 `(a, a)` 有 2 个相邻位置（0–1、1–2），词频 3 → 统计值 6；但它是**重叠**的，一次合并只能取一个，从左到右贪心后剩下 `(aa, a)`。这也解释了 4 个 a 的 `aaaa` 为什么是 `(aa, aa)`：4 个 a 有 3 个相邻位置（统计 9），但只消得掉两处、不重叠。

**结论**：统计值是"上界"，合并操作数是"下界"——下一轮的统计必须基于**合并后的序列**重新计算，而不是把统计值减去一次。这正是朴素版每轮都要重建 `count_pairs` 的原因，也是增量版只需更新受影响词的原因。

## 4. tie-break 与编码 merge rank 的含义

**tie-break**：`select_pair` 在所有频次最大的 pair 里取 `max(candidates)`，即按 `bytes` 元组的**字典序**取较大者，而不是随机或按出现顺序。实测：

```python
Counter({(b"a", b"n"): 7, (b"t", b"he"): 7, (b"x", b"y"): 3})  ->  (b"t", b"he")   # 7 平局，b"the" 的 t(0x74) > a(0x61)
Counter({(b"a", b"z"): 5, (b"b", b"c"): 5})                ->  (b"b", b"c")
```

这是确定性的、与语料顺序无关的规则，保证同一份语料在任何机器上训练出完全相同的词表与 id 分配。

**merge rank 的含义**：`merge_ranks` 把 pair 映射到它被学到的序号（0 最早、22 最晚）。`encode_pretoken` 每轮先找出**当前序列里 rank 最小**的相邻 pair，对它调用 `merge_sequence`，重复到没有 pair 出现在 `merge_ranks` 里为止。因此编码等价于"按训练时学到的优先级，把该 pre-token 能拆成的最粗粒度 token 还原出来"——rank 越小优先级越高，先长出来的结构先被合并。训练与编码必须共享同一套 rank，否则切分与词表 id 对不上。

## 5. 中文、emoji、特殊 token 的编码解码记录

全部经 `encode` → `decode` 验证，roundtrip 均为 `True`：

| 输入 | ids | 说明 |
| --- | --- | --- |
| `你好，hello world!🙂` | `[278, 273, 266, 270, 111, 279, 100, 33, 240, 159, 153, 130]` | 12 个 id；`你好`(278)、`，`(273)、`hello`(266)、`" w"`(270)、`o`(111)、`rl`(279)、`d`(100) |
| `你好，世界！你好，语言模型。🙂` | 31 个 id，前 10 个 `[278, 273, 228, 184, 150, 231, 149, 140, 263, 129]` | 未在训练集出现的字（`世`=e4b896、`界`=e7958c、`！`=efbc81…）只能按字节切：263 是 `efbc`、129 是 `0x81`，**单个 token 停在半个字符上** |
| `the cat in the hat.` | `[258, 271, 32, 105, 110, 262, 32, 104, 260, 46]` | `the`(258)、`" cat"`(271)、`" the"`(262)、`at`(260) 复用了 trace 里的合并 |
| `aaa aa a` | `[261, 97, 32, 261, 32, 97]` | `aa`(261) 与单 `a` |
| `<\|endoftext\|>` | `[256]` | 特殊 token 被整体识别为单个 id，**不参与 BPE** |
| `Once upon a time<\|endoftext\|>the cat` | `[79, 110, 99, 101, 32, 117, 112, 111, 110, 32, 97, 32, 116, 105, 109, 101, 256, 258, 271]` | 特殊 token 是硬边界，右侧 `the`/`" cat"` 仍正常合并 |
| `hello<\|endoftext\|>hello` | `[266, 256, 266]` | 特殊 token 两侧的 `hello` 不跨边界合并 |
| 单字 `你` (e4bda0) | `[277]` | `你` 是训练集里学到的完整 token |
| 单字 `🙂` (f09f9982) | `[240, 159, 153, 130]` | emoji 不在词表，拆成 4 个字节 id |

几个要点：

1. **中文按字节粒度合并**。BPE 的字母表是 256 个字节，一个汉字是 3 个 UTF-8 字节，所以 `你好` 先经历 `e5a5`、`e4bd` 这种"半个汉字"的中间 token，最后才拼成完整字符。`decode` 必须**先把所有 id 的字节拼接、再整体 UTF-8 解码**（`b"".join(vocab[i])` 后 `decode("utf-8", errors="replace")`），不能逐个 token 解码。
2. **emoji 是 4 字节**，训练语料里只出现一次，凑不够频次，所以保持 4 个单字节 id；但它 roundtrip 仍然正确，因为拼接后再解码。
3. **特殊 token 优先级最高**：`<|endoftext|>` 在预分词阶段就被切成独立片段并整体匹配成 id 256，既不参与合并，也不会与左右文合并（见 `hello<|endoftext|>hello`）。这是训练时把 `<|endoftext|>` 作为段落分隔符、并且词表里第 256 个位置留给了它的直接结果。

## 6. 为什么 roundtrip 不能独立证明编码正确

`decode(encode(x)) == x` 只证明"编码是可逆的、且没有丢字节"，它约束不了**切分方式**。任何保留全部字节的切分都能 roundtrip：

- 最极端的反例是"逐字节恒等编码"——把 UTF-8 每个字节都当一个 id，roundtrip 100% 正确，但它根本不是 BPE；
- 词表里出现 `efbc`、`e5a5`、`e4bd` 这类**不是合法 UTF-8 的中间 token**，正说明"id 序列合法"与"token 本身是合法文本"是两回事，判据只能是"拼字节后解码"；
- 我实际做的一个探针：把 `merge_ranks` 整体反序后重新编码 `你好，hello world!🙂`，得到的 ids 与正常编码**完全相同**——在这条文本上反序恰好没改变贪心结果。这反而说明单靠一个"看起来对"的例子无法排除 rank 用错，必须靠定向测试。

roundtrip 查不出、但确实会错的情况至少有四类：

1. **merge rank 用错**（例如按 id 而不是 rank 排序）：切分粒度变了、id 也变了，但字节一个不少，roundtrip 照样过；
2. **跨 pre-token 合并**：例如训练时把 `the` 与下一个 pre-token 开头的空格合并成一个 token，字节仍完整，但破坏了预分词边界，编码结果与训练时的统计口径不再对应；
3. **特殊 token 边界被破坏**：`<|endoftext|>` 若被拆成字节再参与合并，roundtrip 依然正确（字节能拼回去），但特殊 id 256 失效、段落分隔语义丢失；
4. **decode 的拼接顺序错误**（例如按 id 排序拼接）：只要字节多重集相同，某些文本仍能还原。

因此正确性必须由**行为测试**兜底，我实际依赖的用例包括 `test_c2_join_bytes_before_utf8_decode`（先 join 再 decode）、`test_special_longest_and_no_cross_boundary_merge`（最长匹配、不跨边界）、以及 `test_incremental` 里对 `select_pair` 平局、`count_pairs` 重叠计数的断言；`lab_tools.check m1` 的 49 个用例就是这套判据。

## 7. 选做 B 版：两版一致性、重复计时、资源条件与解释

**一致性**（`--debug` 会在每轮合并后校验两版）：

```bash
python -m tokenizer_lab_code.cli compare --cache var/m1-report/counts.json \
  --vocab-size 280 --debug --output var/m1-report/compare-debug.json
# {"equal": true, "requested_vocab_size": 280, "actual_vocab_size": 280, ...}
```

`equal: true` 且 `actual_vocab_size = 280`：两版在每一轮 merge 后的 pair 统计与合并序列完全一致，最终词表内容与 id 分配逐项相同。debug 模式的计时（naive 0.00183 s vs incremental 0.00548 s）**不能**当加速比——增量版每轮要多做全量一致性校验，反而更慢，`timing_note` 里也明确写了这一点。

**计时**（`--repeats 3`，关掉 debug 的全量校验，仅保留轻量 trace）：

```bash
python -m tokenizer_lab_code.cli compare --cache var/m1-report/counts.json \
  --vocab-size 280 --repeats 3 --output var/m1-report/compare-timing.json
```

| 轮次 | naive | incremental | speedup |
| --- | --- | --- | --- |
| 1 | 0.00185 s | 0.00079 s | 2.345× |
| 2 | 0.00174 s | 0.00072 s | 2.430× |
| 3 | 0.00173 s | 0.00071 s | 2.434× |

三次加速比 2.345× / 2.430× / 2.434×，稳定在 2.4 倍左右。单次 `train --trace` 的朴素版 `training_seconds` 为 0.00187–0.00203 s，与上表同量级。

**资源条件与解释**：

- 语料只有 264 字节、31 个 unique pre-token、58 次出现，23 轮合并就到 280，**训练本身只有毫秒级**，所以计时主要反映 Python 解释器开销，绝对数值很小、抖动也明显（同一命令不同轮次差 10–20%，但三轮内增量版始终比朴素版快 2.3 倍以上）。
- 朴素版每轮都把全部 31 个 pre-token 重新扫一遍统计 pair，复杂度是 `O(轮数 × 总 token 数)`；增量版维护 `build_index` 的 pair → 词位置索引，只重算**包含被合并 pair 的那几个词**（`update_counts` / `update_index`），并用 `PairAgenda` 取最大频次，其余词的统计保持有效。在 tiny 上 31 个词几乎都被波及，所以只快 2.4 倍；语料越大、被波及的词占比越小，优势越明显（这正是 B1–B3 要考察的规模效应）。
- 峰值内存没有专门测（语料太小没有意义）；增量版多存一份 pair→词的倒排索引，空间是 `O(不同 pair 数 + 词数)`，与朴素版同阶但常数更大。

## 8. 关于模板第 9–13 条

选做的原规模训练（TinyStories 10K / OWT 32K）、其中的预分词/缓存/BPE 训练耗时与峰值内存、语料分析（最长普通 token、两种词表差异、压缩率）都属于 `m1_tokenizer.md` 中 `original-data-training` 锚点**之后**的内容，本次报告按要求只覆盖锚点之前的部分，故未执行，也未编造数据。