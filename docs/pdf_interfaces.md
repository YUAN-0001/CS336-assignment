# PDF 接口与 TODO 对照

公开函数、类和输入输出对应 [原作业 PDF](../cs336_assignment1_basics.pdf) 橙黄色框中的建议。下表页码为 PDF 印刷页码。没有完整签名的题目，按框中列出的参数与行为组织代码；文件名由本课程约定。

工程细节由教师提供，学生填写关键算法。**保留类与函数签名，在 TODO 所在位置补计算，并删除对应的 `raise NotImplementedError(...)`。** 若框架先放置了 `result = None` 等占位赋值，将它替换为计算结果；不要删除后面已经提供的返回、状态复制或设备搬运代码。

## 对照表

| 阶段 | PDF 问题名／页码 | 文件与公开入口 | 学生填写位置 |
|---|---|---|---|
| M1 | train_bpe／9 | [api.py](../tokenizer_lab_code/tokenizer_lab/api.py)：`train_bpe(input_path, vocab_size, special_tokens)` → `(vocab, merges)` | [naive.py](../tokenizer_lab_code/tokenizer_lab/naive.py) A1–A4；完整训练调度与预分词已提供 |
| M1 | tokenizer／11–12 | [tokenizer.py](../tokenizer_lab_code/tokenizer_lab/tokenizer.py)：`Tokenizer(vocab, merges, special_tokens=None)` | C1 编码普通 pre-token，C2 `decode`；特殊 token 与正则外围已提供 |
| M1 | tokenizer／12 | `Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=None)`、`encode`、`encode_iterable` | 文件读取及迭代入口已提供，调用同一份学生编码算法 |
| M2 | linear／18 | [layers.py](../transformer_lab_code/layers.py)：`Linear(in_features, out_features, device=None, dtype=None)` | T1 `forward(x)`；无 bias |
| M2 | embedding／18–19 | 同文件：`Embedding(num_embeddings, embedding_dim, device=None, dtype=None)` | T2 `forward(token_ids)` |
| M2 | rmsnorm／20 | 同文件：`RMSNorm(d_model, eps=1e-5, device=None, dtype=None)` | T3 `forward(x)`，赋值 `result` |
| M2 | positionwise_feedforward／22 | 同文件：`SwiGLU(d_model, d_ff, device=None, dtype=None)` | T4 `silu`、T5 `SwiGLU.forward` |
| M2 | rope／23 | [attention.py](../transformer_lab_code/attention.py)：`RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=None)` | T8 `forward(x, token_positions)`，赋值 `rotated_even`、`rotated_odd` |
| M2 | softmax／24 | layers.py：`softmax(x, dim=-1)` | T6，dim 对应题目中的维度 i |
| M2 | scaled_dot_product_attention／24–25 | attention.py：`scaled_dot_product_attention(q, k, v, mask=None)` | T7；True 表示允许关注 |
| M2 | multihead_self_attention／25–26 | attention.py：`MultiHeadSelfAttention(d_model, num_heads, ...)` | T9 `causal_mask`；Q/K/V 投影、多头变形、RoPE 调度已提供 |
| M2 | transformer_block／26 | [model.py](../transformer_lab_code/model.py)：`TransformerBlock(d_model, num_heads, d_ff, ...)` | T10 `forward` 中的两段残差 |
| M2 | transformer_lm／26–27 | 同文件：`TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, ...)` | T11 `forward(token_ids)` 中的完整前向链路 |
| M3 | cross_entropy／29 | [losses.py](../training_lab_code/losses.py)：`cross_entropy(logits, targets)` | O1，形状整理后赋值 `mean_loss` |
| M3 | adamw／32 | [optimizer.py](../training_lab_code/optimizer.py)：`AdamW(params, lr, betas, eps, weight_decay)` | `step()` 内 O2 赋值 `m_new/v_new`，O3 赋值 `parameter_new` |
| M3 | learning_rate_schedule／33–34 | 同文件：`cosine_schedule(step, max_lr, min_lr, warmup_steps, total_steps)` | O4；依次对应 t、αmax、αmin、Tw、Tc |
| M3 | gradient_clipping／34 | 同文件：`clip_gradients(parameters, max_norm)` | O5，赋值全局 `norm/scale`；框架原地缩放 `.grad` |
| M4 | data_loading／34–35 | [data.py](../training_lab_code/data.py)：`get_batch(data, batch_size, context_length, device)` | S1，赋值 numpy 窗口数组 `x/y`；返回设备上的 long 张量 |
| M4 | checkpointing／36 | [checkpoint.py](../training_lab_code/checkpoint.py)：`save_checkpoint(model, optimizer, iteration, out)`、`load_checkpoint(src, model, optimizer)` | 已提供；支持路径和二进制文件对象，加载返回 iteration |
| M4 | training_together／36–37 | [train.py](../lab_tools/train.py) 的完整 `run` 循环 | [student.py](../training_lab_code/student.py)：S2 训练一步，S3 验证汇总 |
| M4 | decoding／38 | [sampling.py](../generation_lab_code/sampling.py)：`generate(model, token_ids, max_new_tokens, temperature, top_p, eos_id, seed)` | G1–G3；完整循环、停止条件和采样调度已提供 |
| M5 | experiment_log 与 §7 实验框 | [M5 说明](m5_experiments.md)、train/generate/plot 入口 | 自主改动及对照；PDF 此处没有统一的类或函数签名 |

`d_ff` 由调用者显式指定。原 PDF 建议约为 `8/3*d_model` 并取合适的 64 倍数；微型正确性测试可以使用其他正整数，不人为限制测试形状。

## M1：直接使用 PDF 接口

完成 A/C 后，从仓库根目录运行以下 Python 代码。切换 `variant="incremental"` 即使用选做 B 版，输入输出不变。

```python
from tokenizer_lab_code.tokenizer_lab import Tokenizer, train_bpe
from tokenizer_lab_code.tokenizer_lab.storage import save_vocab_merges

vocab, merges = train_bpe(
    input_path="tokenizer_lab_data/tiny.txt",
    vocab_size=280,
    special_tokens=["<|endoftext|>"],
    variant="naive",
)
save_vocab_merges("var/m1/vocab.json", "var/m1/merges.json", vocab, merges)
tokenizer = Tokenizer.from_files(
    vocab_filepath="var/m1/vocab.json",
    merges_filepath="var/m1/merges.json",
    special_tokens=["<|endoftext|>"],
)
ids = tokenizer.encode("你好，world！")
assert tokenizer.decode(ids) == "你好，world！"
```

双文件格式：词表文件含 `format="cs336-vocab-hex-v1"` 和 `[整数ID, bytes十六进制字符串]` 列表；合并文件含 `format="cs336-merges-hex-v1"` 和按训练顺序排列的两个 bytes 十六进制字符串列表。ID 与任意字节均无损保存，学生无需手写序列化。

双文件保留词表中的特殊 token 字节和 ID，但不记录哪些词条需要作为特殊 token 识别。调用 `from_files` 时应显式传入 `special_tokens` 列表。未传时与 `Tokenizer(vocab, merges)` 相同，不主动识别特殊 token。PDF 未规定具体 JSON 编码，本课程统一使用上述格式。

CLI 原命令继续可用，同时可导出 PDF 双文件：

```sh
python -m tokenizer_lab_code.tokenizer_lab.cli train --input tokenizer_lab_data/tiny.txt --variant naive --vocab-size 280 --output var/m1/tokenizer.json --vocab-output var/m1/vocab.json --merges-output var/m1/merges.json
python -m lab_tools.prepare --vocab var/m1/vocab.json --merges var/m1/merges.json --special-token '<|endoftext|>' --train data/train.txt --valid data/valid.txt --out var/tokenized_data
```

第二条命令中的文本路径应换成实际训练/验证文件。`prepare` 使用 `Tokenizer.from_files`，随后把同一词表封装为数据目录内部的 `tokenizer.json`，以便数据身份验证、训练和生成沿用同一份编码器。它不重新训练词表。

旧合并文件仍用 `storage.load_model(path)` 或 `prepare --tokenizer path` 读取。合并文件包含特殊 token、pattern、trace 等元数据；双文件对应 PDF 的默认正则预分词规则，自定义 pattern 请继续用合并格式。

## M2：不需要先学习配置类

```python
import torch
from transformer_lab_code.model import TransformerBlock, TransformerLM

block = TransformerBlock(d_model=32, num_heads=4, d_ff=64, context_length=16)
model = TransformerLM(
    vocab_size=280, context_length=16, d_model=32,
    num_layers=2, num_heads=4, d_ff=64,
)
logits = model(torch.randint(0, 280, (2, 8)))
assert logits.shape == (2, 8, 280)
```

构造函数仍负责参数注册与初始化，数学 TODO 在对应 `forward`。配置文件工具内部仍可使用 `ModelConfig` 校验输入，调用时转换成显式参数；这不是学生实例化模块的前置知识。
`rope_theta`、`use_rope`、`use_norm`、`ffn_type`、`tie_embeddings` 是课程扩展关键字。保留旧的权重键名；修改代码后继续同一次旧实验仍会触发代码指纹检查，应使用新的实验目录，不绕过恢复检查。

## 从旧骨架迁移已有代码

| 旧位置 | 新位置 | 迁移时注意 |
|---|---|---|
| `RMSNorm.normalize` | `RMSNorm.forward` 的 T3 | 将结果赋值 `result`，保留最终 dtype 恢复 |
| `rotate_pairs` | `RotaryPositionalEmbedding.forward` 的 T8 | 将两个返回值改为赋值 `rotated_even/rotated_odd`，保留交错拼接 |
| `TransformerLM.predict` | `TransformerLM.forward` 的 T11 | 保留输入长度检查，返回 logits |
| `cross_entropy_rows` | `cross_entropy` 的 O1 | 使用 `flat_logits/flat_targets`，最终赋值 `mean_loss` |
| `update_moments/update_parameter` | `AdamW.step` 的 O2/O3 | 赋值局部结果，不提前 return；保留参数状态管理和复制 |
| `clipping_scale` | `clip_gradients` 的 O5 | 赋值 `norm/scale`，保留原地缩放 |
| `shifted_windows` | `get_batch` 的 S1 | 使用已采样的 starts 构造 x/y，保留设备搬运 |
| `TransformerLM(config)` | `TransformerLM(vocab_size=..., context_length=..., ...)` | 配置工具可用 `TransformerLM(**asdict(config))`；普通代码直接传参数 |

A1–A4、B1–B3、C1–C2、T1/T2/T4–T7/T9/T10、O4、S2/S3、G1–G3 的算法任务不变。TODO 总数仍为必做 28 个，B 版选做 3 个。这里调整位置与公开接口，不增加新算法要求。

## 阶段验证

```sh
python -m lab_tools.check --support
python -m lab_tools.check m1
python -m lab_tools.check m2
python -m lab_tools.check m3
python -m lab_tools.check m4
python -m lab_tools.check m5
```

代码未补齐时，阶段测试报告 `NotImplementedError` 是预期状态。每阶段默认回归之前的必做代码；支持设施测试不依赖 TODO。
M1 增加直接训练→保存→from_files→编码解码测试，M2 增加直接构造和调用测试；AdamW 的状态及参数手算测试直接调用 `step`。测试不靠 adapter 替学生的公开接口补齐缺失结构。
