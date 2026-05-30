"""
共享 Transformer / Patch 原语 — Point-BERT / Point-MAE / PointRWKV 等模型的共用底层。

集中此文件的动机:
- Point-BERT 与 Point-MAE 的 `Group / Encoder / Mlp / Attention / Block / TransformerEncoder / DropPath`
  完全同体；PointRWKV 也复用同套 `Group / Encoder`。把这些原语集中后, 各 baseline
  仅保留与自家论文绑定的差异化模块, 避免"同名异源"。

不放此文件的 (按职责归属其他模块):
- 几何算子 (square_distance / farthest_point_sample / index_points / knn_point / query_ball_point /
  grouping / interpolation 等) → utils.pointnet_utils
- 3DETR 专用的 GenericMLP / PositionEmbeddingCoordsSine 等 → utils.detr3_util
- PTv3 的序列化 (z-order / hilbert) → utils.serialization
- 损失分发 → utils.loss

注意:
- `PatchGroup` / `PatchEncoder` 是 `Group` / `Encoder` 的明确化命名 (避免与
  PTv2/V3 内的 `Block / Encoder` 混淆); `utils.point_rwkv_utils` 仍以原名 `Group / Encoder`
  re-export 以保持下游兼容。
"""

from typing import Optional

import torch
import torch.nn as nn

from utils.pointnet_utils import (
    farthest_point_sample_fast,
    index_points_fast,
    knn_point,
)


# ============================================================================
# DropPath — 逐样本随机深度 (Stochastic Depth)
# ============================================================================

class DropPath(nn.Module):
    """与 timm.layers.DropPath 行为一致的轻量本地实现。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ============================================================================
# MLP (FFN) — Transformer 内的前馈网络
# ============================================================================

class Mlp(nn.Module):
    """两层 MLP: FC → Act → Drop → FC → Drop (ViT 风格)。"""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# Attention — 标准多头自注意力 (ViT 风格)
# ============================================================================

class Attention(nn.Module):
    """标准多头自注意力 (Multi-Head Self-Attention)。"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# TransformerBlock — Pre-LN 标准块 (Attention + MLP + 残差)
# ============================================================================

class TransformerBlock(nn.Module):
    """标准 ViT Block: LN → Attention → DropPath → LN → MLP → DropPath。"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# TransformerEncoder — 堆叠 Block + 位置编码相加
# ============================================================================

class TransformerEncoder(nn.Module):
    """Point-BERT / Point-MAE 的 Transformer 编码器: tokens + pos 相加后送入 blocks。"""
    def __init__(self, embed_dim: int = 768, depth: int = 4, num_heads: int = 12,
                 mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


# ============================================================================
# PatchGroup — FPS + KNN 分组, 构建局部 patches
# ============================================================================

class PatchGroup(nn.Module):
    """Point-BERT / Point-MAE / PointRWKV 共用的 patch 分组器。

    流程:
        1. FPS 采样 `num_group` 个中心点
        2. KNN 取每个中心的 `group_size` 个邻居
        3. 邻域以中心为原点做归一化
    """
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) — 输入点云坐标
        Returns:
            neighborhood: (B, G, K, 3) — 中心归一化后的局部 patches
            center: (B, G, 3) — FPS 采样的中心点坐标
        """
        center_idx = farthest_point_sample_fast(xyz, self.num_group)  # (B, G)
        center = index_points_fast(xyz, center_idx)                    # (B, G, 3)
        idx = knn_point(self.group_size, xyz, center)             # (B, G, K)
        neighborhood = index_points_fast(xyz, idx)                # (B, G, K, 3)
        neighborhood = neighborhood - center.unsqueeze(2)         # 中心归一化
        return neighborhood, center


# ============================================================================
# PatchEncoder — Mini-PointNet 编码 patches → patch tokens
# ============================================================================

class PatchEncoder(nn.Module):
    """Mini-PointNet 编码器, 将局部点 patches 编码为 patch tokens。

    架构: Conv1d(3→128→256) → MaxPool 拼接 → Conv1d(512→512→C)
    与 Point-BERT / Point-MAE / PointRWKV 官方实现一致。
    """
    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, K, 3) — 归一化后的点 patches
        Returns:
            feature_global: (B, G, C) — 编码后的 patch tokens
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3).permute(0, 2, 1)  # (BG, 3, K)
        feature = self.first_conv(point_groups)                              # (BG, 256, K)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]          # (BG, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # (BG, 512, K)
        feature = self.second_conv(feature)                                  # (BG, C, K)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]         # (BG, C)
        return feature_global.reshape(bs, g, self.encoder_channel)


# ============================================================================
# trunc_normal_ — 与 timm 一致的截断正态分布初始化
# ============================================================================

def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0):
    """截断正态分布初始化 (alias of torch.nn.init.trunc_normal_)。"""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


__all__ = [
    "DropPath",
    "Mlp",
    "Attention",
    "TransformerBlock",
    "TransformerEncoder",
    "PatchGroup",
    "PatchEncoder",
    "trunc_normal_",
]
