"""M3：状态与参数组管理已提供；学生只填写 AdamW 的数学更新。"""

import math  # noqa: F401 -- O3/O4 使用

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("无效的 AdamW 超参数")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """PDF adamw：在同一个 step 中观察状态更新、偏差修正和参数更新。

        O2：m_t = β₁ m_{t-1} + (1-β₁) g_t，v_t = β₂ v_{t-1} + (1-β₂) g_t²。
        O3：α_t = η sqrt(1-β₂^t)/(1-β₁^t)，
        θ_t = (1-ηλ) θ_{t-1} - α_t m_t/(sqrt(v_t)+ε)。
        t 从 1 开始；η 为学习率，λ 为解耦权重衰减系数。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("本实验仅支持 dense gradients")
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["t"] += 1
                grad, m, v, t = p.grad, state["m"], state["v"], state["t"]
                lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
                # O2：一阶矩 m_t 与二阶矩 v_t 的更新公式见方法说明。
                m_new = v_new = None
                raise NotImplementedError("TODO O2: AdamW.step — 更新一阶矩和二阶矩")
                state["m"].copy_(m_new)
                state["v"].copy_(v_new)
                # O3：PDF Algorithm 1 的参数更新；ε 位于未作偏差修正的 sqrt(v_t) 外。
                parameter_new = None
                raise NotImplementedError("TODO O3: AdamW.step — 计算新参数")
                p.copy_(parameter_new)
        return loss


def cosine_schedule(step, max_lr, min_lr, warmup_steps, total_steps):
    """O4：返回标量学习率，定义见 docs/m3_optimization.md。

    step 是已完成的更新次数，从 0 开始；warmup_steps=0 时直接进入 cosine。
    step > total_steps 时返回 min_lr。
    """
    if step < 0 or not 0 <= warmup_steps < total_steps or not 0 <= min_lr <= max_lr:
        raise ValueError("无效的学习率日程")
    raise NotImplementedError("TODO O4: cosine_schedule")


@torch.no_grad()
def clip_gradients(parameters, max_norm):
    """PDF gradient_clipping：参数集合 + 最大范数 → 原地修改各参数的 .grad。

    所有参数梯度视为同一个向量 g，r = sqrt(Σ_i g_i²)。
    r ≤ M 时 s = 1，否则 s = M/(r+10⁻⁶)；裁剪后的梯度为 s g。
    M 为 max_norm；范数以 FP32 计算。
    扩展返回裁剪前范数供日志使用，普通调用可以忽略返回值。
    """
    if max_norm <= 0:
        raise ValueError("max_norm 必须为正数")
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norm = scale = None
    raise NotImplementedError("TODO O5: clip_gradients — 计算全局 norm 和 scale")
    if not torch.isfinite(torch.as_tensor(norm)):
        raise FloatingPointError("梯度范数非有限；停止更新以便排查")
    for grad in grads:
        grad.mul_(scale)
    return norm
