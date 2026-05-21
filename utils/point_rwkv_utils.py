"""
PointRWKV / PointBERT / PointMAE 共用的非几何辅助工具。

仅保留 RWKV 特有的多尺度封装 + PointNet++ 特征传播 + 训练指标:
- MultiScaleGrouping: 多尺度分组 (基于 PatchGroup + PatchEncoder)
- PointNetFeaturePropagation: 特征传播 (三近邻插值)
- AverageMeter: 平均值统计

通用 patch 原语已迁移到 utils.transformer_blocks (PatchGroup / PatchEncoder),
此处通过别名 `Group / Encoder` re-export 以保持现有调用方兼容。

基础几何函数 (square_distance / knn_point / fps / index_points / fps_points)
来自 utils.pointnet_utils。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pointnet_utils import (
    square_distance, knn_point, fps, index_points, fps_points
)
from utils.transformer_blocks import PatchGroup, PatchEncoder

# 兼容别名 — 旧代码以 Group / Encoder 名导入
Group = PatchGroup
Encoder = PatchEncoder


# ============================================================================
# 多尺度分组 (Multi-Scale Grouping)
# ============================================================================

class MultiScaleGrouping(nn.Module):
    """多尺度点云分组：在不同分辨率下分别分组 + 编码，用于层级特征学习。

    Args:
        num_points_list: 每层 FPS 采样点数，如 [2048, 1024, 512]
        group_sizes: 每层 KNN 分组大小，如 [32, 32, 32]
        embed_dim: 编码后的特征维度
    """
    def __init__(self, num_points_list, group_sizes, embed_dim):
        super().__init__()
        self.num_scales = len(num_points_list)
        self.num_points_list = num_points_list
        self.group_sizes = group_sizes
        self.groupers = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for i in range(self.num_scales):
            self.groupers.append(Group(num_points_list[i], group_sizes[i]))
            self.encoders.append(Encoder(embed_dim))

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            features_list: [(B, G_i, C), ...] 各尺度特征
            centers_list: [(B, G_i, 3), ...] 各尺度中心点
        """
        features_list, centers_list = [], []
        for i in range(self.num_scales):
            neighborhood, center = self.groupers[i](xyz)
            feature = self.encoders[i](neighborhood)
            features_list.append(feature)
            centers_list.append(center)
        return features_list, centers_list


# ============================================================================
# 特征传播 (Feature Propagation) — PointNet++ FP 模块
# ============================================================================

class PointNetFeaturePropagation(nn.Module):
    """PointNet++ 特征传播模块 (三近邻插值上采样 + 跳连接 + MLP)。"""
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, N1, 3) — 目标坐标 (稠密)
            xyz2: (B, N2, 3) — 源坐标 (稀疏, N2 < N1)
            points1: (B, N1, D1) — 目标特征 (跳连接) 或 None
            points2: (B, N2, D2) — 源特征 (待插值)
        Returns:
            new_points: (B, N1, mlp[-1])
        """
        B, N1, _ = xyz1.shape
        N2 = xyz2.shape[1]

        if N2 == 1:
            interpolated_points = points2.repeat(1, N1, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            _, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2
            )

        new_points = interpolated_points if points1 is None else torch.cat([points1, interpolated_points], dim=-1)
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


# ============================================================================
# 辅助: 平均值统计
# ============================================================================

class AverageMeter:
    """计算并存储当前值与平均值。"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
