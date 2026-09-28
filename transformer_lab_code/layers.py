"""M2 基础层。框架已提供，填写 T1–T6；公式和验收见 docs/m2_transformer.md。"""

import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """TODO T1-init：初始化线性层权重，不使用 bias。

        weight 的创建已提供。请使用截断正态分布初始化：均值为 0，
        标准差为 sqrt(2 / (in_features + out_features))，截断范围为 ±3 倍标准差。
        可使用 math.sqrt 和 nn.init.trunc_normal_。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        raise NotImplementedError("TODO T1-init: Linear.__init__ — 初始化 self.weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """T1：x [..., Din]，weight [Dout, Din] → [..., Dout]，没有 bias。"""
        raise NotImplementedError("TODO T1: Linear.forward")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """TODO T2-init：初始化嵌入矩阵。

        weight 的创建已提供。请使用均值为 0、标准差为 1、
        截断范围为 [-3, 3] 的正态分布初始化，可使用 nn.init.trunc_normal_。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        raise NotImplementedError("TODO T2-init: Embedding.__init__ — 初始化 self.weight")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """T2：整数 IDs [...] → 向量 [..., D]。使用索引，不需要构造 one-hot。"""
        raise NotImplementedError("TODO T2: Embedding.forward")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """TODO T3-init：保存归一化所需的配置。

        可学习增益 weight 已创建并初始化为全 1。
        请将传入的 eps 保存为 self.eps，供 forward 使用。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        raise NotImplementedError("TODO T3-init: RMSNorm.__init__ — 保存 eps")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PDF rmsnorm / T3：输入 [..., D]，返回相同形状与原 dtype。

        在下面直接计算 x / sqrt(mean(x²) + eps) * self.weight，赋值给 result。
        只沿最后一维归约，保留归约维度，不减均值。类型转换已经提供。
        """
        original_dtype = x.dtype
        x = x.float()
        result = None
        raise NotImplementedError("TODO T3: RMSNorm.forward — 计算 result")
        return result.to(original_dtype)


def silu(x):
    """T4：SiLU(x) = x * sigmoid(x)，可以使用 torch.sigmoid。"""
    raise NotImplementedError("TODO T4: silu")


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """TODO T5-init：使用本文件的 Linear 组装三个无偏置投影。

        self.w1 和 self.w3：d_model → d_ff；self.w2：d_ff → d_model。
        将 device、dtype 传递给各个子层，属性名需与 forward 和测试接口一致。
        """
        super().__init__()
        raise NotImplementedError("TODO T5-init: SwiGLU.__init__ — 创建 w1、w2、w3")

    def forward(self, x):
        """T5：w2(SiLU(w1(x)) ⊙ w3(x))。⊙ 是逐元素乘法，不是矩阵乘法。"""
        raise NotImplementedError("TODO T5: SwiGLU.forward")


def softmax(x, dim=-1):
    """T6：沿 dim 减最大值、取指数、归一化。允许某些项为 -inf。

    本课程保证每行至少有一个未被屏蔽的有限值，不要求处理全 -inf 行。
    不能调用 torch.softmax / nn.functional.softmax。
    """
    raise NotImplementedError("TODO T6: softmax")
