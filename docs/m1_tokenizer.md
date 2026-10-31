# M1：文本变成 token

接口位置、PDF 页码与 TODO 迁移见 [PDF 接口对照](pdf_interfaces.md)。

## 目标与范围

本阶段完成 byte-level BPE：把文本转为 UTF-8 bytes，通过反复合并高频相邻项得到词表，再用已学习的合并规则编码新文本。所有实验使用 CPU。

必做为 A1–A4、C1–C2，共 6 个 TODO；B1–B3 是选做。详细概念、边界条件与代码导航见 [BPE 自包含实验说明](tokenizer_lab.md)。下面给出本课程 M1 的完成路径和提交边界。

## 步骤

1. 阅读 `Word(tokens, frequency)`：一个 pre-token 在语料中出现多少次，与其中某个 pair 出现多少次是两回事。
2. 完成 [naive.py](../tokenizer_lab_code/naive.py) 中 A1 加权计数、A2 选 pair、A3 非重叠合并、A4 登记。
3. 完成 [tokenizer.py](../tokenizer_lab_code/tokenizer.py) 中 C1 按 merge rank 编码、C2 拼接 bytes 后解码。
4. 在 tiny 语料训练词表，记录前几次合并；解释重复字符、频次平局、中文和特殊 token。
5. 正式验收，不要求先完成增量优化。

```sh
uv run python -m tokenizer_lab_code.cli pretokenize --input tokenizer_lab_data/tiny.txt --output var/m1/counts.json
uv run python -m tokenizer_lab_code.cli train --cache var/m1/counts.json --variant naive --vocab-size 280 --trace --output var/m1/tokenizer.json
uv run python -m tokenizer_lab_code.cli encode --model var/m1/tokenizer.json --text "你好，hello world!🙂"
uv run python -m lab_tools.check m1
```

一个 `aaa` 包含两个用于统计的 `aa`，但执行一轮非重叠合并只变成 `(aa,a)`。频次相同时比较 bytes pair 元组的字典序。编码新文本时使用训练时的 merge rank，不再按当前文本的频次训练。特殊 token 是硬边界，不能跨过它合并。

## 阶段验收与提交

验收只收集 CPU 设施、A 版与编码解码测试；不收集 M2、增量 BPE 或原作业 1.5 秒速度测试。

提交自己修改的两个文件、测试结果、小词表及合并 trace，按 [报告模板](../templates/m1_report.md) 写简短报告。必须验证中文/emoji roundtrip，并解释为什么 roundtrip 正确仍不足以证明编码规则正确。

本阶段结束，你可以用自己的分词器给 M4 的 demo 生成数据。正式 TinyStories 实验会使用统一发放的 tokenizer 和配套 token 数组，以保证不同组结果可比；仍然调用你自己的编码/解码类处理 prompt 和输出。

## 最后再做：进阶与耗时任务

- B 版：补齐 [incremental.py](../tokenizer_lab_code/incremental.py)，运行 `uv run pytest tokenizer_lab_tests/test_incremental.py -q`；在同一缓存上逐轮比较 A/B。
- 先 correctness，再关闭 debug 测性能；样本级运行三次，比较预处理和 BPE 训练的分项耗时。
- 更大样本、10K/32K 词表、全量编码均非本阶段必做。不要用朴素版直接开始全量语料训练。

<a id="original-data-training"></a>

## 按原作业训练 TinyStories / OpenWebText 分词器（选做）

下面保留原作业第2章的真实语料实验路径。先通过 A/C 测试，再完成 B 版并核对 A/B 一致性。完整数据的训练放在本阶段最后，作为扩展实验，不改变前面的必做验收。

### 1. 数据与原实验目标

| 实验 | 输入训练集 | 最大词表大小 | 原实验关注点 |
|---|---|---:|---|
| `train_bpe_tinystories` | `TinyStoriesV2-GPT4-train.txt` | 10,000 | 保存词表与 merges，记录耗时/内存，观察最长 token，分析训练瓶颈 |
| `train_bpe_expts_owt` | `owt_train.txt` | 32,000 | 保存词表与 merges，观察最长 token，比较与 TinyStories 的差异 |

词表目标包含256个基础字节和特殊 token；保留 `<|endoftext|>` 作为文档边界。这里只用 **train** 文件学习合并规则，valid 文件留给编码和泛化比较。OWT 指原作业提供的 **OpenWebText 子样本**，不是自行下载的完整 OpenWebText。

平台已提供数据时，直接将下面 `data/...` 换成共享数据的实际路径，无需每人复制。需要自行准备时，可按[上游 README 的数据来源](https://github.com/stanford-cs336/assignment1-basics#download-data)在课程 Linux 终端下载。以下命令从仓库根目录执行：

```sh
mkdir -p data
wget -c -O data/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -c -O data/TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget -c -O data/owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget -c -O data/owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gzip -dk data/owt_train.txt.gz
gzip -dk data/owt_valid.txt.gz
```

如果解压后的文件已存在，不必再次解压。训练接口接收 **UTF-8 明文 `.txt`**，不能直接传 `.gz`。下载命令用于 Linux；下面的单行 `uv run ...` 训练命令也适用于 PowerShell。

### 2. 先用样本确认 A/B 切换正确

**A 版名称是 `naive`，B 版名称是 `incremental`。** 两版使用同一份缓存、同一词表目标和共用的编码器。B 版仍会调用 A 版的基础函数，所以应先完成 A1–A4，再完成 B1–B3。

```sh
uv run python -m tokenizer_lab_code.cli pretokenize --input tests/fixtures/tinystories_sample.txt --workers 1 --special-token '<|endoftext|>' --output var/m1/sample-counts.json
uv run python -m tokenizer_lab_code.cli train --cache var/m1/sample-counts.json --variant naive --vocab-size 500 --trace --output var/m1/sample-A.json
uv run python -m tokenizer_lab_code.cli train --cache var/m1/sample-counts.json --variant incremental --vocab-size 500 --trace --output var/m1/sample-B.json
uv run python -m tokenizer_lab_code.cli compare --cache var/m1/sample-counts.json --vocab-size 500 --debug --output var/m1/sample-check.json
```

`compare` 会执行两版并核对每轮 pair、频次及最终 vocab/merges。测速度时去掉 `--debug`，加上 `--repeats 3`。完整语料不要用 `compare` 起步，它会把慢的 A 版也完整跑一遍。

### 3. TinyStories：训练 10K 词表

下面分为预处理和 BPE 训练两步，预分词缓存可重复利用。`--workers 2` 是示例，可以按需调整参数。

使用naive版代码：

```sh
uv run python -m tokenizer_lab_code.cli pretokenize --input data/TinyStoriesV2-GPT4-train.txt --workers 2 --special-token '<|endoftext|>' --output var/m1/tinystories-counts.json
uv run python -m tokenizer_lab_code.cli train --cache var/m1/tinystories-counts.json --variant naive --vocab-size 10000 --output var/m1/tinystories-B-10k.json
uv run python -m tokenizer_lab_code.cli encode --model var/m1/tinystories-B-10k.json --text 'Once upon a time, there was a little rabbit.<|endoftext|>'
```

使用incremental版代码：

```sh
uv run python -m tokenizer_lab_code.cli pretokenize --input data/TinyStoriesV2-GPT4-train.txt --workers 2 --special-token '<|endoftext|>' --output var/m1/tinystories-counts.json
uv run python -m tokenizer_lab_code.cli train --cache var/m1/tinystories-counts.json --variant incremental --vocab-size 10000 --output var/m1/tinystories-B-10k.json
uv run python -m tokenizer_lab_code.cli encode --model var/m1/tinystories-B-10k.json --text 'Once upon a time, there was a little rabbit.<|endoftext|>'
```

第一步不调用学生 TODO；第二步要求 A/B 已完成；第三步还要求 C1–C2 已完成。如果只想从原始文本一次完成训练，也可以用以下等价入口，**无需把两种方式都跑一遍**：

```sh
uv run python -m tokenizer_lab_code.cli train --input data/TinyStoriesV2-GPT4-train.txt --workers 2 --special-token '<|endoftext|>' --variant incremental --vocab-size 10000 --output var/m1/tinystories-B-10k-direct.json
```

### 4. OpenWebText：训练 32K 词表

完成 TinyStories 并确认内存占用后，再运行 OWT。建议确认incremental版效率后，只使用incremental版进行训练。

```sh
uv run python -m tokenizer_lab_code.cli pretokenize --input data/owt_train.txt --workers 2 --special-token '<|endoftext|>' --output var/m1/owt-counts.json
uv run python -m tokenizer_lab_code.cli train --cache var/m1/owt-counts.json --variant incremental --vocab-size 32000 --output var/m1/owt-B-32k.json
uv run python -m tokenizer_lab_code.cli encode --model var/m1/owt-B-32k.json --text 'A tokenizer converts text into a sequence of token IDs.<|endoftext|>'
```

要对这两份缓存运行 A 版，只需将训练命令中的 `--variant incremental` 改为 `--variant naive`，同时将输出文件名中的 `B` 改为 `A`，其他参数保持一致。例如 TinyStories 为：

```sh
uv run python -m tokenizer_lab_code.cli train --cache var/m1/tinystories-counts.json --variant naive --vocab-size 10000 --output var/m1/tinystories-A-10k.json
```

这说明调用方式，并不要求运行全量 A 版。A 版每轮重新扫描全部 pre-token，通常应只在样本上作为正确性参照。只完成 A/C 的同学仍可完成 M1 必做任务。

### 5. Python 调用与原测试如何切换

| 入口 | 选择 A 版 | 选择 B 版 |
|---|---|---|
| 教学 CLI `train` | `--variant naive` | `--variant incremental` |
| `api.train_bpe` / `api.train_counts` | `variant="naive"` | `variant="incremental"` |
| 原测试 `tests.adapters.run_train_bpe` | 默认 A；或环境变量 `CS336_BPE_VARIANT=naive` | 环境变量 `CS336_BPE_VARIANT=incremental` |

**环境变量只影响原测试 adapter，不改变 CLI 或直接调用 API 的默认值。** CLI/API 未指定时仍默认 A。adapter 显式传入 `variant=` 时优先于环境变量。编码时无需再选 A/B，直接加载对应模型文件即可。

Python 文件示例（放在仓库根目录，用 `uv run python 文件名.py` 运行）：

```python
from tokenizer_lab_code.api import train_bpe
from tokenizer_lab_code import Tokenizer
from tokenizer_lab_code.storage import save_vocab_merges

if __name__ == "__main__":  # 多进程入口保护，Windows 下也需要
    variant = "incremental"  # 换成 "naive" 即调用 A 版
    specials = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=specials,
        variant=variant,
        num_workers=2,
    )
    vocab_path = f"var/m1/tinystories-{variant}-vocab.json"
    merges_path = f"var/m1/tinystories-{variant}-merges.json"
    save_vocab_merges(vocab_path, merges_path, vocab, merges)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, specials)
    print(tokenizer.decode(tokenizer.encode("Hello, 世界！")))
```

OWT 对应改为 `data/owt_train.txt`、`vocab_size=32000` 及独立输出名。若已经有缓存，使用 `storage.load_cache()` 与 `api.train_counts()`，避免重复预处理。原测试切换命令见 [详细说明第8节](tokenizer_lab.md#8-与原-cs336-测试衔接)。

### 6. 产物、计时与分析

PDF 风格的直接调用使用 `save_vocab_merges` 与 `Tokenizer.from_files`，保存、读取两个文件。CLI 可同时增加 `--vocab-output` 和 `--merges-output` 输出这两个文件。特殊 token 列表须再次传给 `from_files`，双文件使用默认预分词规则。

为兼容已有实验，CLI 的 `--output` 继续生成一个合并模型 JSON，同时保存 `vocab`、有序 `merges`、特殊 token、正则与元数据，已满足保存词表和合并规则的要求。bytes 用十六进制存储，不能直接当成普通文本词条阅读；请用 `storage.load_model()` 加载。

可以将下面内容保存为仓库根目录的 `inspect_tokenizer.py`，用 `uv run python inspect_tokenizer.py var/m1/tinystories-B-10k.json` 观察最长的普通 token，随后换成 OWT 模型路径：

```python
import sys
from tokenizer_lab_code.storage import load_model

tokenizer = load_model(sys.argv[1])
specials = {s.encode("utf-8") for s in tokenizer.special_tokens}
token_id, raw = max(
    ((i, b) for i, b in tokenizer.vocab.items() if b not in specials),
    key=lambda item: len(item[1]),
)
print("vocab_size =", len(tokenizer.vocab), "merges =", len(tokenizer.merges))
print("longest ordinary token:", token_id, len(raw), repr(raw))
print("UTF-8 preview:", raw.decode("utf-8", errors="replace"))
```

长度按 bytes 计；并列时此示例只展示其中一个。观察 token 为什么出现，并比较两种语料得到的词条。还可按原 `tokenizer_experiments` 各取10篇完整验证文档，比较 `UTF-8 字节数 / token 数`，再用 TinyStories 词表编码 OWT 文档观察跨语料差异；保持文档选择和特殊 token 计入方式一致。全量编码仍为选做。

计时时区分三个部分：预分词、缓存读写、BPE 合并。`pretokenize` 的 `input_seconds` 是预分词时间，不含随后计算源文件 SHA-256 和保存缓存；`train --cache` 的 `input_seconds` 是读缓存时间，`training_seconds` 是 BPE 时间，不含模型保存。`train --input` 的 `input_seconds` 则是预分词时间。不要把缓存读取时间误报成预分词时间。

在课程 Linux 上，可在同一命令前加 `/usr/bin/time -v` 记录整条命令的耗时和最大常驻内存；预处理和训练分别测。示例：

```sh
/usr/bin/time -v -o var/m1/tinystories-B-10k-resource.txt uv run python -m tokenizer_lab_code.cli train --cache var/m1/tinystories-counts.json --variant incremental --vocab-size 10000 --output var/m1/tinystories-B-10k.json
```

该示例会重新训练，正式运行时用它替换前面的训练命令即可。GNU time 的 Maximum resident set size 单位为 KiB，多进程预处理时不能将它视为所有并发进程内存的精确总和；整体峰值以平台作业监控为准。定位瓶颈可以在小样本上把 `python -m ...` 改为 `python -m cProfile -o var/m1/bpe.prof -m ...`，性能比较不要混入 profiling 或 `--debug` 的开销。

当前预处理会把每个文本块读入内存，缓存 JSON 和 B 版索引也有额外开销。记录实际语料、算法版本、进程数、词表大小与测得的耗时/内存；样本运行不能称为全量语料结果。
