# M1 报告

- 姓名/学号：**马立博 / 525033910041**
- 代码 commit：`082d66d`（HEAD，`fix .gitignore for data dir`）。M1 实现落在 `f93493e`（`完成 M1：实现 B 版增量 BPE 优化 (B1-B3)`），其父提交 `cc779d0`（`完成 M1：实现 BPE 分词器的训练、编码与解码`）；那之后只改过 `.gitignore`（`515cdca`、`082d66d` 忽略 `.venv/` 与 `data/`），`git diff f93493e..HEAD -- tokenizer_lab_code` 为空。
- 环境：Windows 11 家庭中文版 10.0.26300；CPU Intel Core i7-14650HX；Python 3.13.5；regex 2026.7.19；pytest 8.3.4；以下命令都在仓库根目录执行。仓库自带的 `.venv` 由 WSL 创建（`pyvenv.cfg` 的 `home=/usr/bin`，没有 `Scripts/`），Windows 下 `uv run` 直接报 `error: failed to remove file ... .venv\lib64: 拒绝访问。 (os error 5)`，所以 Windows 侧全部以系统 Python 跑等价命令；控制台编码为 GBK，输出 emoji 前需设 `PYTHONIOENCODING=utf-8`，否则抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f642'`。选做部分（§8–§10）另外用到两台机器：云主机 Ubuntu 22.04 / 8 vCPU（见 §9）与 WSL 的 `.venv`（Python 3.12.3、torch 2.11.0），后者用于 Python 调用入口与原测试切换。

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
| `python -m lab_tools.check m1` | **49 passed in 0.97s**（复跑 49 passed in 1.21s；含 `test_incremental.py` 的 9 个用例，单独跑亦为 `9 passed in 0.10s`） |

单次训练的毫秒级数字随进程启动与缓存状态有约 10–20% 抖动（同一命令另一轮测得 `input_seconds = 0.00257`、`training_seconds = 0.00203`），但 `merges`、词表内容、id 分配在任何一轮都完全一致。

## 2. 语料、目标/实际词表大小、前几轮合并

**语料**：`tokenizer_lab_data/tiny.txt`，9 行、264 字节、**5 个段落**（4 个 `<|endoftext|>` 分隔）：

- 第 1 段 `the cat in the hat. the cat sat on the mat.`
- 第 2 段 `the dog and the cat played. the dog was happy.`
- 第 3 段 `hello hello world! hello world!`（`hello` 3 次、`world` 2 次，后者正是 `(r, l)` 频次 2 的来源）
- 第 4 段 `你好，世界！你好，语言模型。🙂`
- 第 5 段 `aaaa aaa aa a. banana banana bandana.`

预分词（GPT-2 风格正则）后得到 **31 个 unique pre-token、58 次出现**，平均每个 pre-token 出现约 1.9 次——一个“小但覆盖了中英文、多字节、特殊 token、重叠字符”的语料。

**词表大小**：目标是 280，实际正好 280。构成是 `256` 个基础字节 + `1` 个特殊 token `<|endoftext|>`（id 256）+ `23` 个合并 token（id 257–279），合计 256 + 1 + 23 = 280。特殊 token 占用了 256 这个 id，所以第一轮合并的结果落在 257 而不是 256。

**前几轮合并**（`--trace` 的完整 23 轮，格式为 `merge_id / pair / 合并后 bytes / 频次 / id`）：

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
- 合并结果是**字节串**，不必是合法 UTF-8。id 263 `efbc`、274 `e5a5`、276 `e4bd` 都是某个汉字的 UTF-8 编码前缀（`，` / `好` / `你` 的前两个字节），它们只是中间产物，靠后续轮次补齐成完整字符（273、275、277），最终 278 `你好` 是 6 字节，也是这份语料词表里最长的合并 token。
- 语料里 `banana` 出现两次、`bandana` 一次，所以 `(b, an) → ban` 拿到 3；而 `ana` 是在 `ban` 之后才成形，说明合并顺序是“当前频次最大的对优先”，不是一次性按词频排序。
- 最后剩下的 `(r, l)` 来自 `world`：`w` 已被 `" w"` 吃掉之后，剩余 `orl` 里 `r+l` 才凑够 2。

## 3. `aaa` 的统计与合并为何不同

`count_pairs` 统计的是**相邻对的出现次数**（一个 pair 在一个词里出现 k 次就算 k 次，再乘词频），而合并一次只消除**一处非重叠**的相邻对。所以两者天然不等：

```python
count_pairs([Word((b"a", b"a", b"a"), 3)])      # {(b"a", b"a"): 6}   3 个相邻位置 × 词频 3
merge_sequence((b"a", b"a", b"a"), (b"a", b"a"))          # (b"aa", b"a")    只消掉位置 0–1 这一处
merge_sequence((b"a", b"a", b"a", b"a"), (b"a", b"a"))   # (b"aa", b"aa")   4 个 a 消掉两处
```

`aaa` 里 `(a, a)` 有 2 个相邻位置（0–1、1–2），词频 3 → 统计值 6；但它是**重叠**的，一次合并只能取一个，从左到右贪心后剩下 `(aa, a)`。这也解释了 4 个 a 的 `aaaa` 为什么是 `(aa, aa)`：4 个 a 有 3 个相邻位置（统计 9），但只消得掉两处、不重叠。

**结论**：统计值是“上界”，合并操作数是“下界”——下一轮的统计必须基于**合并后的序列**重新计算，而不是把统计值减去一次。这正是朴素版每轮都要重建 `count_pairs` 的原因，也是增量版只需更新受影响词的原因。

## 4. tie-break 与编码 merge rank 的含义

**tie-break**：`select_pair` 在所有频次最大的 pair 里取 `max(candidates)`，即按 `bytes` 元组的**字典序**取较大者，而不是随机或按出现顺序。实测：

```python
Counter({(b"a", b"n"): 7, (b"t", b"he"): 7, (b"x", b"y"): 3})  ->  (b"t", b"he")   # 7 平局，b"the" 的 t(0x74) > a(0x61)
Counter({(b"a", b"z"): 5, (b"b", b"c"): 5})                     ->  (b"b", b"c")
```

这是确定性的、与语料顺序无关的规则，保证同一份语料在任何机器上训练出完全相同的词表与 id 分配。

**merge rank 的含义**：`merge_ranks` 把 pair 映射到它被学到的序号（0 最早、22 最晚）。`encode_pretoken` 每轮先找出**当前序列里 rank 最小**的相邻 pair，对它调用 `merge_sequence`，重复到没有 pair 出现在 `merge_ranks` 里为止。因此编码等价于“按训练时学到的优先级，把该 pre-token 能拆成的最粗粒度 token 还原出来”——rank 越小优先级越高，先长出来的结构先被合并。训练与编码必须共享同一套 rank，否则切分与词表 id 对不上。

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

1. **中文按字节粒度合并**。BPE 的字母表是 256 个字节，一个汉字是 3 个 UTF-8 字节，所以 `你好` 先经历 `e5a5`、`e4bd` 这种“半个汉字”的中间 token，最后才拼成完整字符。`decode` 必须**先把所有 id 的字节拼接、再整体 UTF-8 解码**（`b"".join(vocab[i])` 后 `decode("utf-8", errors="replace")`），不能逐个 token 解码。
2. **emoji 是 4 字节**，训练语料里只出现一次，凑不够频次，所以保持 4 个单字节 id；但它 roundtrip 仍然正确，因为拼接后再解码。
3. **特殊 token 优先级最高**：`<|endoftext|>` 在预分词阶段就被切成独立片段并整体匹配成 id 256，既不参与合并，也不会与左右文合并（见 `hello<|endoftext|>hello`）。这是训练时把 `<|endoftext|>` 作为段落分隔符、并且词表里第 256 个位置留给它的直接结果。

## 6. 为什么 roundtrip 不能独立证明编码正确

`decode(encode(x)) == x` 只证明“编码是可逆的、且没有丢字节”，它约束不了**切分方式**。任何保留全部字节的切分都能 roundtrip：

- 最极端的反例是“逐字节恒等编码”——把 UTF-8 每个字节都当一个 id，roundtrip 100% 正确，但它根本不是 BPE；
- 词表里出现 `efbc`、`e5a5`、`e4bd` 这类**不是合法 UTF-8 的中间 token**，正说明“id 序列合法”与“token 本身是合法文本”是两回事，判据只能是“拼字节后解码”；
- 我实际做的一个探针：把 `merge_ranks` 整体反序后重新编码 `你好，hello world!🙂`，得到的 ids 与正常编码**完全相同**——在这条文本上反序恰好没改变贪心结果。这反而说明单靠一个“看起来对”的例子无法排除 rank 用错，必须靠定向测试。

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
- 峰值内存没有专门测（语料太小没有意义）；增量版多存一份 pair→词的倒排索引，空间是 `O(不同 pair 数 + 词数)`，与朴素版同阶但常数更大。原规模训练的真实峰值见 §9（TinyStories 训练 124 MB，OWT 训练 9.34 GiB）。

## 8. 选做原规模训练：数据来源与实际规模、缓存校验和、训练版本、模型文件路径、目标/实际词表大小

**数据来源**：两份语料都来自原作业的公开数据（`docs/m1_tokenizer.md:57-64` 指向 HuggingFace）。本机 DNS 解析不了 `huggingface.co`（`wget -c` 只落 0 字节文件），因此改用 hf-mirror.com 镜像逐字节下载，并按官方 LFS 给出的 sha256 逐项核对。OWT 用的是 `stanford-cs336/owt-sample` 的子样本，不是自行下载的完整 OpenWebText。

**实际规模与校验和**：

| 文件 | 字节数 | sha256 |
| --- | ---: | --- |
| TinyStoriesV2-GPT4-train.txt | 2,227,753,162 | 6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443 |
| TinyStoriesV2-GPT4-valid.txt | 22,502,601 | 6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d |
| owt_train.txt.gz | 4,591,240,837 | b19ae88cfbc4016b304c348522455fe38ebac48fffed955adcc7191a89e38ccf |
| owt_train.txt（解压后） | 11,920,511,059 | bbeb7f291a981ecfd5cf44b84d0f654b9e96c53dff99f2556b7d2cccaf8c1918 |
| owt_valid.txt.gz | 111,785,382 | bc73db5da2f19c360836b9c8a88094e13346ec83798c2f40060be39135768c80 |
| owt_valid.txt（解压后） | 289,998,753 | 2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660 |

**缓存校验和**：`pretokenize` 会把源文件的 sha256 写进 counts 的 `source_sha256` 字段。TinyStories 缓存 `var/m1/tinystories-counts.json`（2,772,501 字节，unique pre-token 59,933、出现 536,592,168 次）里就是 `6418d412…`。OWT 缓存 `var/m1/owt-counts.json`（188,212,549 字节，unique 6,601,890、出现 2,471,753,091 次）是 8 段 counts 合并出来的，合并脚本没有重算 11.9 GB 源文件的哈希，字段先是 `null`；训练结束后单独实测 `bbeb7f29…` 并回填（回填只改元数据，unique 与 occurrences 不变）。

**训练版本**：两个词表都用 `--variant incremental`（B 版）训练，命令即 `docs/m1_tokenizer.md:99-101`（TinyStories）与 `docs/m1_tokenizer.md:115-117`（OWT）。OWT 有 660 万个 unique pre-token，A 版“每轮全表重扫”在 16 GB 机器上不可行，所以只在仓库外的启动器 `/root/tools/train_fast.py` 里 monkeypatch 了 `incremental.update_counts`（before 里逐 pair 减、频次 ≤0 就地删；after 只加增量），`tokenizer_lab_code/` 一个字符都没改；TinyStories 的 59,933 个 unique pre-token 不需要补丁，原样 165 s 跑完。

**调用入口的三种切换**（`docs/m1_tokenizer.md:128-162`）：

- 教学 CLI：`--variant naive` / `--variant incremental`，§1 与 §7 已用它完成必做与一致性对比。
- Python API：`train_bpe(..., variant=...)` 直接返回 `(vocab, merges)`，再用 `save_vocab_merges` 落盘、`Tokenizer.from_files` 加载。实测在 5 MB 样本（`tests/fixtures/tinystories_sample_5M.txt`，vocab 1000）上两版产出**完全相同**的词表与合并表（vocab sha256 前 16 位 `6232f6744015b501`，merges `9c9d1e619de3e49b`），`encode("Hello, 世界！")` 得 `[72, 535, 111, 44, 32, 228, 184, 150, 231, 149, 140, 239, 188, 129]`、解码回原串，roundtrip 为 True；耗时 incremental 2.871 s、naive 16.638 s。云主机上同一脚本在完整 2.2 GB 语料（`num_workers=2`）上跑出 `train_bpe_seconds 333.96`、vocab 10,000 / merges 9,743。
- 原测试 adapter：`tests/adapters.py:661` 读取环境变量 `CS336_BPE_VARIANT`（缺省 `naive`）。同一条命令 `pytest tests/test_tokenizer.py tests/test_train_bpe.py`：设 `incremental` 时 **26 passed, 1 xfailed**（含 1.5 s 速度门槛的 `test_train_bpe_speed`）；设 `naive` 时 **25 passed, 1 failed, 1 xfailed**，唯一失败就是速度用例（3.56 s > 1.5 s）。两版功能等价、只有 A 版过不了性能门槛，这正是文档“不要求跑全量 A 版”的依据。另有 1 个快照用例 `test_train_bpe_special_tokens` 需要 syrupy 插件，本机 `.venv` 没有 pip、装不上，未运行。

**模型文件路径**：训练产物在云主机 `/root/cs336/var/m1/`，本地备份在 `var/cloud/var/m1/`。单文件模型 `tinystories-B-10k.json` 882,256 字节（vocab 454,769、merges 427,119、resource.txt 1,015），`owt-B-32k.json` 2,935,743 字节（vocab 1,514,517、merges 1,420,867、resource.txt 989）。`data/` 与 `var/` 都在 `.gitignore` 里，语料、缓存与模型文件不进提交。

**目标/实际词表大小**：TinyStories 目标 10,000、实际 10,000（256 个基础字节 + 1 个 `<|endoftext|>` + 9,743 个 merge）；OWT 目标 32,000、实际 32,000（256 + 1 + 31,743）。两次都刚好命中目标，没有出现“提前无 pair 可合”导致实际词表偏小的情况。

## 9. 选做原规模训练：预分词/缓存读取/BPE 训练/整条命令耗时、峰值内存及测量方法、实际 CPU 型号与预分词进程数

**机器**：云主机 Ubuntu 22.04，8 vCPU Intel Xeon Platinum 8336C @ 2.30GHz，15.4 GB 内存，40 GB 系统盘；仓库在 `/root/cs336`，Python 用 conda env `m1`（3.12.16）。

**分阶段耗时与峰值内存**（每个阶段包一次 `/usr/bin/time -v`，单次运行）：

| 阶段 | 墙钟 | 峰值 RSS | 备注 |
| --- | ---: | ---: | --- |
| TinyStories 预分词（2.2 GB） | 2:49.21 | 5,454,888 KB（5.2 GiB） | `--workers 2`；`input_seconds` 166.668；198% CPU |
| TinyStories 读缓存 + BPE 训练（10K） | 2:45.46 | 123,812 KB | 读缓存 0.045 s，训练 165.226 s |
| OWT 预分词（11.9 GB，切 8 段串行） | 合计 17:00 | 每段约 3.63 GB | 每段 111.7–117.8 s；`--workers 2`；`input_seconds` 102.7–108.7 |
| OWT 合并 8 份 counts | 约 21 s | — | unique 6.60M、occurrences 2.47B，产出 175–188 MB 缓存 |
| OWT 读缓存 + BPE 训练（32K） | 11:25.53 | 9,791,076 KB（9.34 GiB） | 读缓存 5.779 s，训练 678.636 s；CPU 99% |

**整条命令耗时**：TinyStories 下载 2.2 GB 约 17 分钟 + 预分词 2:49 + 训练 2:45 ≈ 23 分钟。OWT 下载 4.59 GB 约 22 分钟 + `gzip -d` 94 s + 8 段预分词 17 分钟 + 合并 21 s + 训练 11:26，整条流水线从 14:29:38 到 15:21:54 共 **52 分 16 秒**。

**测量方法**：`/usr/bin/time -v` 取 `Elapsed (wall clock) time`、`Maximum resident set size`、`Percent of CPU this job got` 三项；预分词按段分别计时再相加，峰值取各段最大值（8 段是串行跑的）；训练就是一次 `train --cache` 子命令，`input_seconds`（读缓存）与 `training_seconds`（合并循环）由 CLI 自报，不含模型保存；预分词每一步的 `unique_pretokens` / `pretoken_occurrences` 也逐段记录并对齐了 8 段之和。注意 GNU time 的单位是 KiB，多进程预处理时它偏小，不能当成所有并发进程的精确总和（`docs/m1_tokenizer.md:197` 也这么提醒），这里 `--workers 2` 的峰值是实测最大值，不是理论加和。

**实际 CPU 型号与预分词进程数**：`Intel Xeon Platinum 8336C`，8 vCPU，`--workers 2`——8 核只开 2 个，把单段峰值压在 3.6 GB 左右，给 15.4 GB 内存留余量。OWT 整文件不切段时峰值约 24 GB，超过机器内存，所以按文档边界切成 8 段、串行预分词再合并 counts；合并后的 unique / occurrences 与整文件预分词等价（同一套 pre-token 计数只是分块累加）。

## 10. 选做语料分析：最长普通 token（bytes 长度及解释）、两种词表差异、验证文档压缩率与跨语料表现

**最长普通 token**：

- TinyStories（10K）最长 **15 字节**，三个并列：id 9379 ` responsibility`、id 9143 ` disappointment`、id 7160 ` accomplishment`；14 字节还有 id 9095 ` granddaughter`、id 8868 ` extraordinary`、id 8699 ` congratulated`。它们都带一个前导空格，正好对应 GPT-2 正则的 `?\p{L}+`——一个 pre-token 是“空格+字母串”，所以最长 token 天然等于最长单词 + 1。长度 ≥8 字节的普通 token 共 2,256 个。
- OWT（32K）最长 **64 字节**，而且前几名全是重复标点或乱码：id 25836 是 64 个 `-`；id 25822 是 64 字节的二次编码乱码（UTF-8 的 `c3 83 c3 82` 序列重复 16 次：原文的 U+00C2 先被当成 latin-1 再编码一次）；id 31274 是 16 个 em dash（48 字节）；id 31162 是 32 个 `*`；id 28585 是 32 个 `.`；id 25146 是 32 个 `=`。长度 ≥8 字节的共 10,023 个，其中“字符种类 ≤2”的重复标点串 23 个、纯 mojibake 形态 7 个。BPE 只看频次、不看字符类别，网页里的分隔线、重复标点、乱码一旦高频出现就会长成超长 token，把整段噪声压成一个 id。

**两种词表差异**（去掉 256 个基础字节与特殊 token 后，比较 merge 产生的词条集合）：TinyStories 9,743 条、OWT 31,743 条，**共有 7,062** 条，TinyStories 独有 **2,681** 条，OWT 独有 **24,681** 条（7,062 + 2,681 = 9,743，7,062 + 24,681 = 31,743，两边各自对上）。

- TinyStories 独有几乎都是儿童故事词汇：`granddaughter`、`congratulated`、`strawberries`、`caterpillars`、`imaginations`、`marshmallows`、`veterinarian`、`motorcycles`、`butterflies`、`adventurous`、`reluctantly`。
- OWT 独有以长标点串、mojibake 和网页长尾词为主：64 个 `-`、`=`×32、`.`×32、em dash 串，以及 `telecommunications`、`disproportionately`、`environmentalists`、`responsibilities`。词表从 10K 扩到 32K，多出来的 22,000 个位置基本被网页长尾英文与噪声占满。
- 若改按 merge **pair** 集合比，共有只有 5,179 对：同一个词条可能由不同的 pair 拼出来，所以“同一词条”比“同一 pair”更宽。报告用词条口径（文档要求比较的是词条）。

**验证文档压缩率与跨语料表现**：口径是 UTF-8 字节数 / token 数，每份语料取 valid 集前 10 篇非空文档（按 `<|endoftext|>` 切分、去掉分隔符）：

| 词表 | TinyStories valid（10 篇，10,778 B） | OWT valid（10 篇，50,460 B） |
| --- | --- | --- |
| TinyStories 10K | 2,687 tokens → 4.011 B/token | 14,821 tokens → 3.405 B/token |
| OWT 32K | 2,787 tokens → 3.867 B/token | 11,201 tokens → 4.505 B/token |

对角线（词表与语料同源）压缩最好：TinyStories 词表在 TinyStories 上 4.011 B/token，OWT 词表在 OWT 上 4.505 B/token；换上另一份语料都退化——TinyStories 词表拿到网页文本上从 4.011 掉到 3.405（−15.1%），OWT 词表拿到儿童故事上从 4.505 掉到 3.867（−14.2%）。OWT 词表跨语料退化略小，因为它与 TinyStories 仍共享 7,062 个词条；反过来 TinyStories 词表遇到网页长尾词与噪声只能退回字节，损失更大。两组样本量差 4.7 倍（10.8 KB vs 50.5 KB），只能在同一列内比较、不能跨列比绝对值；全量编码整个 valid 文件仍属选做，本次没有做。

本次报告覆盖模板全部 11 条：第 1–8 条对应 §1–§7，第 9–11 条对应 §8–§10。
