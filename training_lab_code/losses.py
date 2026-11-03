"""M3：从 logits 计算稳定的交叉熵。"""

import torch  # noqa: F401 -- O1 可使用 torch 操作


def cross_entropy(logits, targets):
    """PDF cross_entropy / O1：logits [...,V]、targets [...] → 平均损失标量。

    L = (1/N) Σ_n [log(Σ_j exp(z_nj - m_n)) - (z_n,y_n - m_n)]，
    其中 m_n = max_j z_nj，y_n 为目标类别，N 为预测位置总数。
    不调用 F.cross_entropy，不先 softmax 再 log，不 detach。
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets 的形状必须等于 logits 除最后一维外的形状")
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_targets = targets.reshape(-1).long()
    if not flat_targets.numel():
        raise ValueError("不能对空 batch 计算损失")
    mean_loss = None
    raise NotImplementedError("TODO O1: cross_entropy — 计算 mean_loss")
    return mean_loss


def target_scores(scores, targets):
    """已提供：从每行取出该行目标类别的分数，返回 [N]。"""
    return scores.gather(-1, targets[:, None]).squeeze(-1)

