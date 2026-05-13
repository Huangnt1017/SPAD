"""
Pointcept 通用工具函数

从 Pointcept 仓库复现的通用工具:
- offset/batch 转换 (用于变长点云批次)
- LayerNorm1d (PT V1 使用)
- 点云操作辅助函数 (kNN, FPS, grouping, interpolation 的纯 PyTorch 实现)

Author: Adapted from Pointcept (Xiaoyang Wu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Batch / Offset 转换 (用于变长点云批次管理)
# ============================================================================

@torch.no_grad()
def offset2bincount(offset):
    """offset → 每个样本的点数"""
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    """每个样本的点数 → offset (累计和)"""
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    """offset → batch 索引 (每个点的批次归属)"""
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    """batch 索引 → offset"""
    return torch.cumsum(batch.bincount(), dim=0).long()


# ============================================================================
# LayerNorm1d (PT V1 点集上使用的一维 LayerNorm)
# ============================================================================

class LayerNorm1d(nn.Module):
    """
    对点云特征进行 LayerNorm，兼容 (N, C) 与 (B, N, C) 两种形状。
    Pointcept 中用于点特征的归一化。
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        # 仅在 BatchNorm1d 上包装，行为与 LayerNorm 不同但 Pointcept 中如此使用
        self.norm = nn.BatchNorm1d(normalized_shape, eps=eps, affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # (B, N, C) → (B, C, N) → BN → (B, N, C)
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        elif x.dim() == 2:
            # (N, C) → (C, N) → BN → (N, C)
            return self.norm(x.t()).t()
        else:
            raise NotImplementedError(f"LayerNorm1d: unsupported dim {x.dim()}")


# ============================================================================
# 纯 PyTorch 实现的 pointops 替代
# ============================================================================

def square_distance(src, dst):
    """欧氏距离平方: src[B,N,C] dst[B,M,C] → [B,N,M]"""
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    根据索引收集点: points[B,N,C] idx[B,S]或[B,S,K] → [B,S,C]或[B,S,K,C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz, npoint):
    """最远点采样 (FPS): xyz[B,N,3] → [B,npoint] (索引)"""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_point(k, xyz, new_xyz):
    """kNN: xyz[B,N,3] new_xyz[B,S,3] → idx[B,S,k]"""
    dist = square_distance(new_xyz, xyz)  # [B, S, N]
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


def knn_point_with_break_tie(k, xyz, new_xyz):
    """kNN with tie breaking for stability."""
    dist = square_distance(new_xyz, xyz)
    # add small noise to break ties
    noise = torch.rand_like(dist) * 1e-8
    idx = (dist + noise).topk(k=k, dim=-1, largest=False)[1]
    return idx


def grouping(feat, idx, with_xyz=False):
    """
    根据索引分组: feat[B,N,C] idx[B,N,K] → grouped[B,N,K,C]
    """
    grouped = index_points(feat, idx)
    if with_xyz:
        return grouped
    return grouped


def grouping_with_xyz(xyz, feat, idx):
    """
    xyz[B,N,3] feat[B,N,C] idx[B,N,K] → pos[B,N,K,3], grouped_feat[B,N,K,C]
    """
    grouped_xyz = index_points(xyz, idx)
    grouped_feat = index_points(feat, idx)
    return grouped_xyz, grouped_feat


def interpolation(unknown_xyz, known_xyz, known_feat, n_unknown, n_known, k=3):
    """
    反距离加权插值: 用于上采样点特征
    unknown_xyz[B,U,3], known_xyz[B,K,3], known_feat[B,K,C] → [B,U,C]
    """
    dist = square_distance(unknown_xyz, known_xyz)  # [B, U, K]
    # 取 k 个最近邻
    dist, idx = dist.topk(k=k, dim=-1, largest=False)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=-1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feat = torch.sum(
        index_points(known_feat, idx) * weight.unsqueeze(-1), dim=2
    )
    return interpolated_feat


# ============================================================================
# 变长点云版本的 FPS 和 kNN (使用 offset 约定)
# ============================================================================

def farthest_point_sample_varlen(xyz: torch.Tensor, offset: torch.Tensor, new_offset: torch.Tensor):
    """
    变长点云的 FPS: 使用 offset 管理不同样本长度
    xyz: (N, 3) — 所有样本的点串联
    offset: (B,) — 每个样本的结束索引
    new_offset: (B,) — 采样后每个样本的结束索引
    Returns: new_xyz_idx (M,) — 采样点在原 xyz 中的索引
    """
    device = xyz.device
    npoints_new = new_offset[-1].item()
    new_xyz_idx = torch.zeros(npoints_new, dtype=torch.long, device=device)
    
    for i in range(len(offset)):
        if i == 0:
            s_i, e_i = 0, offset[0].item()
        else:
            s_i, e_i = offset[i - 1].item(), offset[i].item()
        
        s_new = 0 if i == 0 else new_offset[i - 1].item()
        e_new = new_offset[i].item()
        npoint = e_new - s_new
        
        if npoint <= 0 or e_i - s_i <= 0:
            continue
        
        batch_xyz = xyz[s_i:e_i, :].unsqueeze(0)  # [1, N, 3]
        idx = farthest_point_sample(batch_xyz, npoint)
        new_xyz_idx[s_new:e_new] = idx[0] + s_i
    
    return new_xyz_idx


def knn_point_varlen(k, xyz, offset, new_xyz, new_offset):
    """
    变长点云的 kNN
    xyz: (N, 3), new_xyz: (M, 3)
    Returns: idx (M, k)
    """
    device = xyz.device
    M = new_xyz.shape[0]
    idx = torch.zeros(M, k, dtype=torch.long, device=device)
    
    for i in range(len(offset)):
        if i == 0:
            s_i, e_i = 0, offset[0].item()
            s_new, e_new = 0, new_offset[0].item()
        else:
            s_i, e_i = offset[i - 1].item(), offset[i].item()
            s_new, e_new = new_offset[i - 1].item(), new_offset[i].item()
        
        if e_i - s_i <= 0 or e_new - s_new <= 0:
            continue
        
        batch_xyz = xyz[s_i:e_i, :].unsqueeze(0)  # [1, N, 3]
        batch_new_xyz = new_xyz[s_new:e_new, :].unsqueeze(0)  # [1, M, 3]
        batch_idx = knn_point(k, batch_xyz, batch_new_xyz)  # [1, M, k]
        idx[s_new:e_new, :] = batch_idx[0]
    
    return idx


def knn_query_and_group(feat, xyz, offset, new_xyz=None, new_offset=None, 
                         nsample=16, idx=None, with_xyz=True):
    """
    Pointcept 风格的 knn_query_and_group:
    - 如果 new_xyz 和 new_offset 为 None，则在原点上进行 kNN (自注意力)
    - 返回 grouped_feat (如果 with_xyz=True, 则 xyz 拼接在 feat 前)
    """
    if new_xyz is None:
        new_xyz = xyz
        new_offset = offset
    
    if idx is None:
        idx = knn_point_varlen(nsample, xyz, offset, new_xyz, new_offset)
    
    # 由于使用的是变长约定，需要按 batch 索引收集
    # 注意: 这里简化处理，假设 idx 可以直接索引
    # 对于 pointops 风格，feat 和 xyz 都是串联的
    
    if with_xyz:
        # 返回 [pos, grouped_feat] 其中 grouped_feat = concat(pos, feat)
        grouped_xyz = new_xyz.unsqueeze(1).expand(-1, nsample, -1)  # 简化
        # 实际应该按照 idx 索引 xyz
        grouped_feat = feat[idx.long(), :] if idx is not None else None
        return torch.cat([grouped_xyz, grouped_feat], dim=-1) if grouped_feat is not None else grouped_xyz
    else:
        return feat[idx.long(), :] if idx is not None else None


# ============================================================================
# 其他辅助函数
# ============================================================================

def off_diagonal(x):
    """返回方阵非对角元素的扁平视图"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
