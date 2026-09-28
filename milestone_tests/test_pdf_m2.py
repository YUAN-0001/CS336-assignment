"""PDF 公开构造接口；直接使用各层及模型，避免 adapter 掩盖接口差异。"""
import torch

from transformer_lab_code.attention import RotaryPositionalEmbedding, MultiHeadSelfAttention
from transformer_lab_code.layers import Linear, Embedding, RMSNorm
from transformer_lab_code.model import TransformerBlock, TransformerLM


def test_pdf_modules_direct_calls():
    ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    x = Embedding(num_embeddings=16, embedding_dim=8, device="cpu", dtype=torch.float32)(ids)
    assert x.shape == (2, 3, 8)
    assert Linear(in_features=8, out_features=4, device="cpu", dtype=torch.float32)(x).shape == (2, 3, 4)
    assert RMSNorm(d_model=8, eps=1e-5)(x).shape == x.shape
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=8, device="cpu")
    assert rope(x=x, token_positions=torch.arange(3)).shape == x.shape
    assert MultiHeadSelfAttention(d_model=8, num_heads=2)(x).shape == x.shape
    assert TransformerBlock(d_model=8, num_heads=2, d_ff=16)(x).shape == x.shape
    model = TransformerLM(vocab_size=16, context_length=8, d_model=8, num_layers=2, num_heads=2, d_ff=16)
    scores = model(ids)
    assert scores.shape == (2, 3, 16)
    scores.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
