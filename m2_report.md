# M2 报告

- 姓名/学号：**马立博 / 525033910041**
- 代码 commit：`199dc3b`（"M2:T1-T11"），其父提交 `3b1b15e`（m1-submission）
- 测试结果：`python -m lab_tools.check m2` → **68 passed in 4.98s**（含 M1 全量回归、M2 里程碑四项、`tests/test_model.py` 13 项、softmax 对照），无 skip、无 xfail；T1–T11 全部完成，`transformer_lab_code/` 下已无生效的 `NotImplementedError`。
- 环境：WSL2 Ubuntu + 仓库自带 `.venv`（Python 3.12.3、torch 2.11.0+cu130、24 核 CPU），以下命令都在仓库根目录执行。

## 张量形状

以本仓库配置 B=2、T=8、D=256、H=8、Dh=32、F=704、V=10000 为例（表中同时给出通用形状）：

| 环节 | 输入形状 | 输出形状 | 哪个维度的含义发生变化 |
|---|---|---|---|
| Embedding | `[B,T]` = `[2,8]`（int64 ID） | `[B,T,D]` = `[2,8,256]` | 整数 ID 维变为特征维 D；T 仍是序列位置，不参与任何计算 |
| Q/K/V 投影与拆头 | `[B,T,D]` = `[2,8,256]` | 投影后各 `[B,T,D]`，拆头后 `[B,H,T,Dh]` = `[2,8,8,32]` | 特征维 D 被分解为 H×Dh（只改内存布局，元素总数不变）；B、T 不变 |
| attention 分数和 softmax | q `[B,H,T,Dh]`、kᵀ `[B,H,Dh,T]` | 分数 `[B,H,T,T]` = `[2,8,8,8]`，softmax 后同形 | **新增**最后一维 key 位置；两个 T 含义不同（行是 query 位置，列是 key 位置），softmax 沿 key 维 |
| 合头、输出投影 | 加权后 `[B,H,T,Dh]` | 合头 `[B,T,D]`，投影后 `[B,T,D]` | H×Dh 合回 D，key 维消失；输出投影只在特征维做线性混合 |
| FFN | `[B,T,D]` = `[2,8,256]` | `[B,T,D]`，中间 `[B,T,F]` = `[2,8,704]` | 特征维 D→F→D，T、B 不动；门控是逐元素乘，不混位置 |
| LM head | `[B,T,D]` = `[2,8,256]` | `[B,T,V]` = `[2,8,10000]` | 特征维 D 变词表维 V；每个位置独立得到一个 V 维分数，模型末尾不加 softmax |

实测一次前向：输入 `torch.randint(0, 10000, (2, 8))` → 输出 `[2, 8, 10000]`，float32。

## 理解与验证

- **Parameter、buffer、ModuleList 各有什么作用？** `nn.Parameter` 把张量注册成可学习权重，会出现在 `.parameters()` 里、被优化器更新、写进 `state_dict()`；本阶段的 Linear 权重、RMSNorm 的 γ、Embedding 矩阵都是 Parameter。`register_buffer(..., persistent=False)` 注册的是**不可学习**的张量：RoPE 的 cos/sin 随 `model.to(device)` 一起搬家，但不进优化器，也不写入权重快照（`persistent=False`），因为它可以由 theta、d_k、max_seq_len 完全重建。`nn.ModuleList` 的作用是**注册**每一层：普通 Python list 里的子模块不会出现在 `.parameters()` 和 `state_dict()` 里，`.to(device)` 也会漏掉它们；用 ModuleList 后 L 个 block 都能被遍历、保存和迁移，而它本身不定义 forward。
- **为什么因果 mask 包含对角线？为何 softmax 沿 key 维度？** 第 i 行允许第 0..i 列，对角线 `M[i,i]=True` 表示当前位置能看到自己——预测第 i+1 个 token 时，位置 i 的输入 token 本来就是已知输入，不属于"未来"，所以不遮挡；被挡掉的是 j>i 的未来 token。给定当前位置的输入预测下一个 token，因此看到自己不会泄漏标签。mask 的语义在本仓库统一为 **True = 允许关注**，`apply_mask` 把 False 位置在 softmax 前填成 -inf。实测：取常量 q、k（全 0，分数全相等）与三行 v，`causal_mask(3)` 下输出恰好是第一行取 v₀、第二行取 (v₀+v₁)/2、第三行取 (v₀+v₁+v₂)/3，说明屏蔽方向正确。softmax 必须沿 **key 维**：它要回答"当前位置对各 key 分配多少权重"，归一化后每一行权重和为 1，再对 V 加权求和；若沿 query 维归一化，等于对每个 key 汇总所有位置的注意力，语义完全变了。
- **pre-norm 残差的两步数据流；模型为什么返回 logits？** 第一步 `y = x + Attention(RMSNorm₁(x))`：先归一化再进注意力，注意力分支的输出加回**未归一化**的 x，残差主干保持恒等通路；第二步 `z = y + FFN(RMSNorm₂(y))`，注意第二段归一化的输入是第一步得到的 y 而不是 x，两段串行。整个模型是 `Embedding → L×Block → 最终 RMSNorm → Linear(D,V)`。返回 logits 而不接 softmax，是因为下游用途不同：训练时 `cross_entropy` 内部做 log-softmax（数值上更稳，避免显式 softmax 再取 log），生成时温度、top-k/top-p 都要在原始分数上操作，模型末尾固定 softmax 反而限制了下游。
- **修改未来输入后的因果性检查；各参数是否有有限梯度？** 用 `milestone_tests/test_m2.py` 的构造自己复测（随机种子 0，V=10000、C=256、D=256、L=4、H=8、F=704）：把位置 5–7 的 token 改成别的 ID，前 5 个位置的 logits 与改前**逐元素最大差 0.0**；把输入截断成前 5 个 token 单独前向，前 5 位最大差 5.36e-07（浮点重算误差量级）。梯度侧：对 logits² 的均值反传后，39 个参数张量**全部** `grad is not None` 且 `isfinite` 全为 True，梯度范数落在 7.9e-05 到 2.8e-02 之间，没有被 mask 的 -inf 污染出 NaN。
- **补充不变量（同一脚本实测）。** RoPE 在位置 0 处 max|x−y| = 0.0（θ⁰=1、角度 0，旋转即恒等），且旋转前后每个 token 向量的 L2 范数最大漂移 1.9e-06；RoPE 只作用于 q、k，不作用于 v，因此"位置零旋转"与"范数保持"同时成立。RMSNorm 对 4 个随机向量（输入 RMS 4.53 / 4.63 / 4.96 / 7.22）输出 RMS 全部 ≈1.0（1e-7 量级），证明除以的是最后一维的均方根而不是标准差（不减均值）；float16 输入 300.0 时输出仍为 float16 且全部有限（内部转 float32 计算再转回，避免 300² 溢出 fp16）。

## 资源

**参数量。** 指定配置取 **GPT-2 XL**：V=50257、C=1024、D=1600、L=48、H=25、F=6400，不共享输入输出 embedding。按文档公式：

参数量 = 2VD + L(4D² + 3DF + 2D) + D

逐项代入：2VD = 160,822,400；每个 block 的 4D² + 3DF + 2D = 10,240,000 + 30,720,000 + 3,200 = 40,963,200，48 层共 1,966,233,600；再加最终 norm 的 1,600，合计 **2,127,057,600 ≈ 2.13B**。

**与真实 GPT-2 XL（1.558B）的差异可以逐项对上**，这反过来验证了公式：真实 GPT-2 XL 共享输入输出 embedding（少 VD = 80,411,200），FFN 是两层 D→4D→D 而不是三个 D×F 矩阵（每层少 3DF − 2D·4D = 30,720,000 − 20,480,000 = 10,240,000，48 层少 491,520,000）。2,127,057,600 − 80,411,200 − 491,520,000 = 1,555,126,400 ≈ 1.56B，与公开数字一致。

本仓库实际小模型（V=10000、C=256、D=256、L=4、H=8、F=704）用同一公式算得 8,333,568，与 `sum(p.numel() for p in model.parameters())` **逐位相等**；分解为 embedding 2,560,000、lm_head 同样 2,560,000、每 block 803,328（attention 4×65,536、FFN 3×180,224、两个 RMSNorm 共 512）、最终 norm 256。可见 4 个 block 只占 38.6%，词表两端（embedding + lm_head）占 61.4%——小模型里"两头"才是参数大头。

**计算量（每序列，batch=1，T 为序列长度）。** 矩阵乘法 `[m,n] @ [n,p]` 记 2mnp FLOPs；QKV 是三次 D×D，FFN 是两次 D×F 加一次 F×D：

| 计算 | 矩阵形状 | FLOPs | GPT-2 XL，T=1024 |
|---|---|---|---|
| QKV 投影 | `[T,D]`×`[D,D]`，三次 | 6TD² | 15.73 G |
| QKᵀ 分数 | `[H,T,Dh]`×`[H,Dh,T]` | 2T²D | 3.36 G |
| softmax | `[H,T,T]` 逐元素 | 约 5T²H（非矩阵乘，可忽略） | 0.13 G |
| 对 V 加权 | `[H,T,T]`×`[H,T,Dh]` | 2T²D | 3.36 G |
| 输出投影 | `[T,D]`×`[D,D]` | 2TD² | 5.24 G |
| FFN（SwiGLU） | `[T,D]`×`[D,F]` 两次 + `[T,F]`×`[F,D]` | 6TDF | 62.91 G |
| LM head | `[T,D]`×`[D,V]` | 2TDV | 164.68 G |
| 合计 | | 8TD² + 4T²D + 6TDF + 2TDV | **255.28 G** |

**T 变长时变化最大的部分。** 只有两项按 T² 增长：QKᵀ 分数 2T²D 和对 V 加权 2T²D，合起来 4T²D——它们把"每个 query 与所有 key 比较"写死成平方级。其余都是 T 的线性项：QKV 与输出投影 8TD²、FFN 6TDF、LM head 2TDV。在 T=1024 时平方项只有 6.71 G（占 2.6%），因为交叉点很远：4T²D 追上 8TD² 要 T > 2D = 3200，追上 6TDF 要 T > 1.5F = 9600。所以对 GPT-2 的 1024 上下文，真正的大头是 **LM head（64.5%）** 与 FFN（24.6%），不是注意力；只有把上下文拉到数千以上，T² 项才会成为主导。

**每 token 视角。** 8TD² + 6TDF 摊到每个 token 是 8D² + 6DF = 82.0 M FLOPs，恰好等于 2×（不含 embedding 的参数量 40.96M）——"每参数每 token 约 2 FLOPs"的经验法则在这里成立；再叠上 LM head 的 2DV = 160.8 M/token（V≫D 时它比整个 transformer 主干还贵），合计 242.7 M/token。
