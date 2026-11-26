"""M2 基础层。框架已提供，填写 T1–T6；公式和验收见 docs/m2_transformer.md。"""

import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """TODO T1-init：初始化线性层权重，不使用 bias。

        weight 的创建已提供。请使用截断正态分布初始化：均值为 0，
        标准差为 sqrt(2 / (in_features + out_features))，截断范围为 ±3 倍标准差。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        # raise NotImplementedError("TODO T1-init: Linear.__init__ — 初始化 self.weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """T1：Y = X Wᵀ。x [..., Din]，weight [Dout, Din] → [..., Dout]，没有 bias。"""
        return x @ self.weight.T
        # raise NotImplementedError("TODO T1: Linear.forward")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """TODO T2-init：初始化嵌入矩阵。

        weight 的创建已提供。请使用均值为 0、标准差为 1、
        截断范围为 [-3, 3] 的正态分布初始化。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
        self.weight.data = self.weight.data.clamp(min=-3, max=3)
        # raise NotImplementedError("TODO T2-init: Embedding.__init__ — 初始化 self.weight")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """T2：E(i) = W 的第 i 行。整数 IDs [...] → 向量 [..., D]。"""
        return self.weight[token_ids]
        # raise NotImplementedError("TODO T2: Embedding.forward")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """TODO T3-init：保存归一化所需的配置。

        可学习增益 γ 的初值为全 1；归一化稳定项 ε = eps。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        # raise NotImplementedError("TODO T3-init: RMSNorm.__init__ — 保存 eps")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PDF rmsnorm / T3：输入 [..., D]，返回相同形状与原 dtype。

        y_i = γ_i x_i / sqrt((1/D) Σ_j x_j² + ε)。γ 为可学习增益。
        求和范围为最后一维的 D 个特征；不减均值。
        """
        original_dtype = x.dtype
        x = x.float()
        result = self.weight * x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # raise NotImplementedError("TODO T3: RMSNorm.forward — 计算 result")
        return result.to(original_dtype)


def silu(x):
    """T4：SiLU(x) = x σ(x)，其中 σ(x) = 1 / (1 + exp(-x))；逐元素计算。"""
    return x / (1 + torch.exp(-x))
    #    raise NotImplementedError("TODO T4: silu")


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """TODO T5-init：使用本文件的 Linear 组装三个无偏置投影。

        self.w1 和 self.w3：d_model → d_ff；self.w2：d_ff → d_model。
        将 device、dtype 传递给各个子层，属性名需与 forward 和测试接口一致。
        """
        super().__init__()
        #  raise NotImplementedError("TODO T5-init: SwiGLU.__init__ — 创建 w1、w2、w3")
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        """T5：SwiGLU(x) = (SiLU(x W₁ᵀ) ⊙ (x W₃ᵀ)) W₂ᵀ。

        W₁、W₃ 的形状为 [d_ff,d_model]，W₂ 为 [d_model,d_ff]；⊙ 为逐元素乘法。
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))
        #        raise NotImplementedError("TODO T5: SwiGLU.forward")


def softmax(x, dim=-1):
    """T6：p_i = exp(x_i - m) / Σ_j exp(x_j - m)，m = max_j x_j。

    i、j 沿 dim 所指定的维度；允许某些项为 -inf。
    本课程保证每行至少有一个未被屏蔽的有限值，不要求处理全 -inf 行。
    不能调用 torch.softmax / nn.functional.softmax。
    """
    m = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - m)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x
    # raise NotImplementedError("TODO T6: softmax")
