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
"""

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
