"""DGCNN 风格图算子 (集中定义, 供自研图模型共享)。

历史上 model/graph_residual.py 与 model/graph_res_GCN.py 各自维护了一份
字节级重复的 knn_gpu / get_graph_feature / weighted_downsample。本模块将其
集中, 保证两个模型 (及未来变体) 使用同一份实现, 避免拷贝漂移。

布局约定: 全程 (B, C, N) channel-first, 与 baseline/DGCNN.py 对齐 (但函数名不同,
DGCNN baseline 作为外部对照基准保留其独立实现, 不依赖本模块)。

主要导出:
- knn_gpu: 特征空间负距离 topk KNN。
- get_graph_feature: EdgeConv 边特征 [x_j - x_i, x_i]。
- weighted_downsample: 按特征 L2 范数无放回加权采样。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# ══════════════════════════════════════════════════
# GPU KNN (DGCNN 风格: 特征空间, 负距离 topk)
# ══════════════════════════════════════════════════

def knn_gpu(x: torch.Tensor, k: int) -> torch.Tensor:
    """在特征空间做 KNN (与 DGCNN 一致)。

    用负平方距离 + topk 实现, 全程 GPU matmul, 无需排序。

    Args:
        x: (B, C, N) — 任意维特征 (坐标 / 学到的特征均可)。
        k: 近邻数 (含自身; 自身距离为 0 必然入选, 但 EdgeConv 的 x_j - x_i = 0
            使其对边特征无贡献, 故无需显式排除)。

    Returns:
        idx: (B, N, k), int64。
    """
    # (B, N, N) 负平方距离: 越大越近
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    neg_dist = -xx - inner - xx.transpose(2, 1)
    _, idx = neg_dist.topk(k=k, dim=-1)
    return idx


# ══════════════════════════════════════════════════
# 图特征构建 (DGCNN 风格: 全局 flatten 索引)
# ══════════════════════════════════════════════════

def get_graph_feature(x: torch.Tensor, k: int,
                      idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """构造 EdgeConv 边特征 [x_j - x_i, x_i] (DGCNN 原版写法)。

    Args:
        x: (B, C, N)。
        k: 近邻数。
        idx: (B, N, k), 若 None 则内部调用 knn_gpu。

    Returns:
        (B, 2C, N, k) — 每点 k 个邻居的边特征。
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn_gpu(x, k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)

    x_t = x.transpose(2, 1).contiguous()                   # (B, N, C)
    nbr = x_t.view(B * N, C)[idx_flat].view(B, N, k, C)    # (B, N, k, C)
    x_i = x_t.unsqueeze(2).expand_as(nbr)                  # (B, N, k, C)

    # [x_j - x_i, x_i] → (B, 2C, N, k)
    return torch.cat([nbr - x_i, x_i], dim=-1).permute(0, 3, 1, 2).contiguous()


# ══════════════════════════════════════════════════
# 加权下采样 (B, C, N) 布局
# ══════════════════════════════════════════════════

def weighted_downsample(
    p: torch.Tensor,
    f: torch.Tensor,
    target_n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按特征 L2 范数做无放回加权采样。

    Args:
        p: (B, 4, N)。
        f: (B, C, N)。
        target_n: 目标点数。

    Returns:
        (p_down, f_down): (B, 4, target_n), (B, C, target_n)。
    """
    B, C, N = f.shape
    if target_n >= N:
        return p, f

    scores = f.norm(p=2, dim=1).clamp(min=1e-8)              # (B, N)
    probs = scores / scores.sum(dim=1, keepdim=True)
    idx = torch.multinomial(probs, target_n, replacement=False)  # (B, target_n)

    # gather 沿 N 维采样
    idx_f = idx.unsqueeze(1).expand(-1, C, -1)                # (B, C, target_n)
    idx_p = idx.unsqueeze(1).expand(-1, 4, -1)                # (B, 4, target_n)
    return torch.gather(p, 2, idx_p), torch.gather(f, 2, idx_f)
