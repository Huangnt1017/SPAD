"""
统一分类头 + 中心点回归头构建工具。

所有 baseline 和自研模型必须使用相同的头部架构，确保：
1. 分类头结构完全一致（层数、维度、激活、Dropout）
2. 中心点回归头结构完全一致
3. 唯一变量仅为 backbone 架构

统一标准：
- cls_head: 3 层 MLP (pooled → 256 → 128 → num_classes)
- box_head: 3 层 MLP (pooled → 256 → 128 → 3)
- 所有层使用 LeakyReLU(0.2) + BatchNorm1d + Dropout(0.3)

例外 (仅自研模型可用, baseline 严禁使用以保证对比公平):
- SegmentationCentroidHead: 分割引导质心头, 见下方类 docstring。
"""

from typing import Tuple

import torch
import torch
import torch.nn as nn


def build_standard_cls_head(
    pooled_dim: int,
    num_classes: int,
    dropout: float = 0.3,
) -> nn.Sequential:
    """构建统一分类头。

    Args:
        pooled_dim: backbone 池化后的特征维度。
        num_classes: 分类类别数。
        dropout: Dropout 概率，默认 0.3。

    Returns:
        nn.Sequential: 3 层 MLP 分类头。

    Architecture:
        Linear(pooled → 256, bias=False) → BN1d → LeakyReLU(0.2) → Dropout
        Linear(256 → 128) → BN1d → LeakyReLU(0.2) → Dropout
        Linear(128 → num_classes)
    """
    return nn.Sequential(
        # 第一层: pooled → 256
        nn.Linear(pooled_dim, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
        # 第二层: 256 → 128
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
        # 输出层: 128 → num_classes
        nn.Linear(128, num_classes),
    )


def build_standard_box_head(
    pooled_dim: int,
    box_dim: int = 3,
    dropout: float = 0.3,
) -> nn.Sequential:
    """构建统一中心点回归头。

    Args:
        pooled_dim: backbone 池化后的特征维度。
        box_dim: 输出维度，默认 3 (中心点 [cx, cy, cz])。
        dropout: Dropout 概率，默认 0.3。

    Returns:
        nn.Sequential: 3 层 MLP 回归头。

    Architecture:
        Linear(pooled → 256, bias=False) → BN1d → LeakyReLU(0.2) → Dropout
        Linear(256 → 128) → BN1d → LeakyReLU(0.2)
        Linear(128 → box_dim)
    """
    return nn.Sequential(
        # 第一层: pooled → 256
        nn.Linear(pooled_dim, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
        # 第二层: 256 → 128
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.2),
        # 输出层: 128 → box_dim (默认 3)
        nn.Linear(128, box_dim),
    )


class SegmentationCentroidHead(nn.Module):
    """分割引导质心回归头 (Segmentation-Guided Centroid, 仅自研模型使用)。

    动机:
        标准 box_head 把全局池化向量 (B, pooled) 直接 MLP 回归出 [cx, cy, cz]。
        全局池化抹掉了逐点空间结构, MLP 只能从统计量里 "盲猜" 绝对坐标,
        深度 (z) 定位精度受限。本头改为:
            逐点目标性打分 → softmax 归一化为权重 → 用真实点坐标加权求质心。
        质心是输入点坐标的凸组合, 天然落在点云凸包内 (归一化空间 [0,1]),
        无需 "自由回归"; z 质心由 "哪些点属于目标" 直接决定, 直击 z_mae。

    与标准 box_head 的关键区别 (这是创新点, 不破坏 baseline 对比公平):
        - baseline 仍用 build_standard_box_head (全局池化 → MLP → [B,3])
        - 本头是自研 backbone 专属, 利用逐点特征 + 原始点坐标, baseline 无此结构

    数据流:
        point_feats (B, C, N) + points_xyz (B, 3, N)
            ↓ seg_mlp (1x1 Conv): C → C/4 → 1
        logits (B, 1, N)
            ↓ softmax over N (排除 padding 点可选)
        weights (B, 1, N)        # Σ_N w = 1, 每点目标性概率
            ↓ 加权求和: Σ_N w_i · xyz_i
        centroid (B, 3)          # 凸组合质心, 落在 [0,1]

    轻量化:
        seg_mlp 为两层 1x1 Conv (C→C/4→1), C=512 时约 66K 参数,
        比标准 box_head (pooled=1024 → 256 → 128 → 3, 约 295K) 轻 ~4.5x。

    Args:
        in_channels: 逐点特征通道数 C (backbone 末层输出, 非池化维度)。
        mid_ratio: 中间层通道压缩比, 中间维度 = max(in_channels // mid_ratio, 16)。
        coord_dim: 坐标维度 (默认 3, 即 xyz; intensity 不参与质心)。

    Returns (forward):
        dict:
            - centroid: (B, coord_dim) 加权质心, 即中心点预测。
            - seg_logits: (B, N) 逐点目标性 logits (可选用于辅助分割监督)。
            - seg_weights: (B, N) softmax 后的逐点权重 (可视化/分析用)。
    """

    def __init__(
        self,
        in_channels: int,
        mid_ratio: int = 4,
        coord_dim: int = 3,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.coord_dim = int(coord_dim)
        mid_channels = max(in_channels // mid_ratio, 16)

        # 逐点目标性打分: 两层 1x1 Conv (等价逐点 MLP), 输出每点 1 个 logit
        self.seg_mlp = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(mid_channels, 1, 1),
        )

    def forward(
        self,
        point_feats: torch.Tensor,
        points_xyz: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        """前向: 逐点打分 → softmax 权重 → 加权质心。

        Args:
            point_feats: (B, C, N) backbone 末层逐点特征。
            points_xyz: (B, coord_dim, N) 原始 (归一化) 点坐标, 与 point_feats 逐点对齐。
            valid_mask: (B, N) 可选, True 表示有效点; padding 点会被屏蔽 (logits→-inf)。

        Returns:
            dict: centroid (B, coord_dim), seg_logits (B, N), seg_weights (B, N)。

        Raises:
            ValueError: point_feats 与 points_xyz 的 batch / N 不一致,
                或 points_xyz 通道数与 coord_dim 不符。
        """
        if point_feats.dim() != 3:
            raise ValueError(f"point_feats must be (B, C, N), got {tuple(point_feats.shape)}")
        if points_xyz.dim() != 3:
            raise ValueError(f"points_xyz must be (B, coord_dim, N), got {tuple(points_xyz.shape)}")
        B, _, N = point_feats.shape
        if points_xyz.shape[0] != B or points_xyz.shape[2] != N:
            raise ValueError(
                f"point_feats {tuple(point_feats.shape)} 与 points_xyz "
                f"{tuple(points_xyz.shape)} 的 batch/N 不一致"
            )
        if points_xyz.shape[1] < self.coord_dim:
            raise ValueError(
                f"points_xyz 通道数 {points_xyz.shape[1]} 小于 coord_dim {self.coord_dim}"
            )

        # 逐点目标性 logits: (B, C, N) → (B, 1, N) → (B, N)
        seg_logits = self.seg_mlp(point_feats).squeeze(1)        # (B, N)

        # 屏蔽 padding 点: 无效点 logits 置 -inf, softmax 后权重为 0
        if valid_mask is not None:
            seg_logits = seg_logits.masked_fill(~valid_mask.bool(), float("-inf"))

        # softmax over N: 每点目标性概率, Σ_N w = 1
        seg_weights = torch.softmax(seg_logits, dim=-1)          # (B, N)

        # 加权质心: Σ_N w_i · xyz_i
        # xyz: (B, coord_dim, N), weights: (B, 1, N) → 广播相乘 → 对 N 求和 → (B, coord_dim)
        xyz = points_xyz[:, : self.coord_dim, :]                 # (B, coord_dim, N)
        centroid = (xyz * seg_weights.unsqueeze(1)).sum(dim=-1)  # (B, coord_dim)

        return {
            "centroid": centroid,
            "seg_logits": seg_logits,
            "seg_weights": seg_weights,
        }
