"""PDF data_loading：S1 位于 get_batch；内存映射、随机采样和设备搬运已提供。"""

import json
from pathlib import Path

import numpy as np
import torch

from tokenizer_lab_code.tokenizer_lab.storage import file_sha256

def get_batch(data, batch_size, context_length, device="cpu", *, starts=None):
    """numpy 一维 token 数组 → (x,y)，均为指定设备上的 long [B,T]。

    S1：对每个起点 s，x=data[s:s+T]、y=data[s+1:s+T+1]，堆叠为两个 numpy 数组。
    在 TODO 中赋值 x、y；索引合法性、随机起点和设备搬运已提供。
    starts 是教师验证入口使用的可选固定起点；按 PDF 调用时无需传入。
    """
    if data.ndim != 1 or len(data) <= context_length or batch_size <= 0 or context_length <= 0:
        raise ValueError("数据长度必须大于 context_length，且 batch_size/context_length 为正")
    if starts is None:
        starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    else:
        starts = np.asarray(starts)
        if (starts.shape != (batch_size,) or starts.dtype.kind not in "iu"
                or np.any(starts < 0) or np.any(starts >= len(data) - context_length)):
            raise ValueError("starts 必须包含 batch_size 个合法整数起点")
    x = y = None
    raise NotImplementedError("TODO S1: get_batch — 构造输入和 next-token 目标窗口")
    return (
        torch.as_tensor(np.array(x, dtype=np.int64), device=device),
        torch.as_tensor(np.array(y, dtype=np.int64), device=device),
    )


def batch_at(data, starts, context_length, device):
    """教师提供：确定性验证仍复用学生的同一个 get_batch 实现。"""
    return get_batch(data, len(starts), context_length, device, starts=starts)


class Dataset:
    def __init__(self, root, verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != "cs336-course-data-v1":
            raise ValueError("未知数据格式，请使用 lab_tools.prepare")
        self.identity = file_sha256(self.root / "manifest.json")
        self.tokenizer_path = self.root / "tokenizer.json"
        if verify and file_sha256(self.tokenizer_path) != self.manifest["tokenizer_sha256"]:
            raise ValueError("tokenizer 与 manifest 不匹配")
        for split in ("train", "valid"):
            path = self.root / f"{split}.npy"
            info = self.manifest[split]
            if verify and file_sha256(path) != info["sha256"]:
                raise ValueError(f"{split} 数据校验和不匹配")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.ndim != 1 or data.dtype.kind not in "ui" or len(data) != info["tokens"]:
                raise ValueError("token 数组形状、dtype 或长度不匹配")
            if len(data) < 2 or int(data.min()) < 0 or int(data.max()) >= self.manifest["vocab_size"]:
                raise ValueError("数据过短或包含超出词表的 ID")
            setattr(self, split, data)

    def check_context(self, context_length):
        if min(len(self.train), len(self.valid)) <= context_length:
            raise ValueError("train/valid 各自至少需要 context_length + 1 个 token")
