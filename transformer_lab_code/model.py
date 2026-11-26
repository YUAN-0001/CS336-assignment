"""M2 完整模型。M4/M5 直接使用本文件中你已完成的实现。"""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .layers import Embedding, Linear, RMSNorm, SwiGLU, silu


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 704
    rope_theta: float = 10000.0
    use_rope: bool = True
    use_norm: bool = True
    ffn_type: str = "swiglu"
    tie_embeddings: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须被 num_heads 整除")
        if self.use_rope and (self.d_model // self.num_heads) % 2:
            raise ValueError("使用 RoPE 时每个 head 的维度必须为偶数")
        if self.ffn_type not in ("swiglu", "silu"):
            raise ValueError("ffn_type 只能为 swiglu 或 silu")


class SiLUFeedForward(nn.Module):
    """M5 可选对照：两层 FFN；配置 d_ff=4*d_model 可近似匹配 SwiGLU 参数量。"""

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, *, context_length: int = 256,
                 rope_theta: float = 10000.0, device=None, dtype=None,
                 use_rope: bool = True, use_norm: bool = True, ffn_type: str = "swiglu"):
        """PDF transformer_block：直接接收 d_model、num_heads、d_ff。

        关键字参数配置位置缓存、设备及 M5 选做；无需先构造 ModelConfig。
        """
        super().__init__()
        if ffn_type not in ("swiglu", "silu") or type(d_ff) is not int or d_ff <= 0:
            raise ValueError("ffn_type 应为 swiglu/silu，d_ff 应为正整数")
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len=context_length, theta=rope_theta,
            use_rope=use_rope, device=device, dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if use_norm else nn.Identity()
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if use_norm else nn.Identity()
        ffn_cls = SwiGLU if ffn_type == "swiglu" else SiLUFeedForward
        self.ffn = ffn_cls(d_model, d_ff, device, dtype)

    def forward(self, x, token_positions=None):
        """T10：两段 pre-norm 残差。输入和输出均为 [...,T,D]。

        y = x + Attention(RMSNorm₁(x))
        z = y + FFN(RMSNorm₂(y))
        """
        y = x + self.attn(self.ln1(x), token_positions=token_positions)
        z = y + self.ffn(self.ln2(y))
        return z
        #        raise NotImplementedError("TODO T10: TransformerBlock.forward")


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, *, rope_theta: float = 10000.0,
                 device=None, dtype=None, use_rope: bool = True, use_norm: bool = True,
                 ffn_type: str = "swiglu", tie_embeddings: bool = False):
        """PDF transformer_lm：显式接收词表、上下文、层数及 block 参数。"""
        super().__init__()
        self.config = ModelConfig(vocab_size=vocab_size, context_length=context_length, d_model=d_model,
                                  num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta,
                                  use_rope=use_rope, use_norm=use_norm, ffn_type=ffn_type,
                                  tie_embeddings=tie_embeddings)
        c = self.config
        self.context_length = context_length
        self.token_embeddings = Embedding(c.vocab_size, c.d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length=context_length, rope_theta=rope_theta,
                             device=device, dtype=dtype, use_rope=use_rope, use_norm=use_norm, ffn_type=ffn_type)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(c.d_model, device=device, dtype=dtype) if c.use_norm else nn.Identity()
        self.lm_head = Linear(c.d_model, c.vocab_size, device, dtype)
        if c.tie_embeddings:
            # 共享参数的初始尺度会影响优化；M5 对照中必须记录此初始化差异。
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """T11：h₀ = Embedding(token_ids)，h_l = Block_l(h_{l-1})，l = 1,…,L。

        logits = RMSNorm(h_L) W_outᵀ。
        token_ids [B,T] → logits [B,T,V]；输出为未归一化分数。
        """
        if token_ids.shape[-1] > self.context_length or token_ids.shape[-1] == 0:
            raise ValueError("输入长度必须在 [1, context_length] 内")
        h = self.token_embeddings(token_ids)
        for layer in self.layers:
            h = layer(h, token_positions=None)
        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits
        # raise NotImplementedError("TODO T11: TransformerLM.forward")

    def config_dict(self):
        return asdict(self.config)
