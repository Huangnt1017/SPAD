"""
Point Transformer V2 (Wu et al., NeurIPS 2022) - 严格对齐 Pointcept 复现。

参考:
- D:\\essay\\3d目标检测复现仓库\\Pointcept-main\\Pointcept-main\\pointcept\\models\\point_transformer_v2\\
    point_transformer_v2m1_origin.py

复现说明:
- ``PointBatchNorm / GroupedVectorAttention / PTv2Block(=Block) / BlockSequence / GridPool /
  GVAPatchEmbed / PTv2Encoder(=Encoder)`` 与 Pointcept ``v2m1`` 一一对应; 默认超参 (patch_embed_*,
  enc_depths=(2,2,6,2), enc_channels=(96,192,384,512), grid_sizes=(0.06,0.12,0.24,0.48) 等)
  全部沿用官方值。
- ``pointops.knn_query / pointops.grouping`` 替换为本文件 ``knn_query / grouping`` 纯 PyTorch 实现
  (变长 offset 约定), 行为对应 ``pointops`` 的"返回 idx+dist"与"按索引 gather"。
- ``torch_scatter.segment_csr`` 与 ``torch_geometric.nn.pool.voxel_grid`` 都尝试动态 import,
  缺失时走 ``GridPool`` 的纯 PyTorch 回退路径 (在 ``GridPool.forward`` 注释里说明)。
- Pointcept 参考是 **语义分割** (per-point ``seg_logits``); SPAD 用作 **分类**, 因此只保留 encoder 部分,
  encoder 末层后做按 offset 的全局平均池化, 再接 ``cls_head`` (logits) + ``center_head`` (3D 中心点),
  **这是与 Pointcept 唯一的差异**, 其余层一律保留。

数据契约 (与项目其他 baseline 一致):
- 输入: (B, N, 4) xyzi 点云。
- 输出: tuple (logits [B, num_classes], center_pred [B, 3])。

Reference:
@article{wu2022point,
  title={Point transformer v2: Grouped vector attention and partition-based pooling},
  author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={33330--33342},
  year={2022}
}
"""

import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.layers import DropPath

# 可用依赖
try:
    from torch_scatter import segment_csr
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False
    print("WARNING: torch_scatter not available, using fallback implementation")

try:
    from torch_geometric.nn.pool import voxel_grid
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("WARNING: torch_geometric not available, using fallback GridPool")

from utils.pointnet_utils import (
    offset2batch, batch2offset, offset2bincount,
    farthest_point_sample, knn_point, index_points, square_distance
)


# ============================================================================
# Point Batch Norm (兼容 [N, C] 与 [N, L, C] 格式)
# ============================================================================

class PointBatchNorm(nn.Module):
    """点云数据的 Batch Normalization"""
    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError(f"Unsupported dim {input.dim()}")


# ============================================================================
# pointops 替代: kNN 查询 + 分组 (使用 offset 约定)
# ============================================================================

def knn_query(k: int, coord: torch.Tensor, offset: torch.Tensor,
              new_coord: Optional[torch.Tensor] = None,
              new_offset: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    kNN 查询，支持变长点云。
    类似于 pointops.knn_query。

    Args:
        k: 邻居数
        coord: (N, 3) 所有点坐标
        offset: (B,) 每个样本的结束索引
        new_coord: (M, 3) 查询点坐标，默认为 coord
        new_offset: (B,) 查询点 offset，默认为 offset

    Returns:
        idx: (M, k) 邻居索引
        dist: (M, k) 邻居距离
    """
    if new_coord is None:
        new_coord = coord
        new_offset = offset

    device = coord.device
    M = new_coord.shape[0]
    B = len(offset)

    idx = torch.zeros(M, k, dtype=torch.long, device=device)
    dist_out = torch.zeros(M, k, dtype=torch.float32, device=device)

    for i in range(B):
        s_i = 0 if i == 0 else offset[i - 1].item()
        e_i = offset[i].item()
        s_new = 0 if i == 0 else new_offset[i - 1].item()
        e_new = new_offset[i].item()

        if e_i - s_i <= 0 or e_new - s_new <= 0:
            continue

        # 计算距离矩阵 (n_new, n)
        dist = torch.cdist(new_coord[s_new:e_new], coord[s_i:e_i])
        k_actual = min(k, dist.shape[1])
        topk_dist, topk_idx = dist.topk(k=k_actual, dim=-1, largest=False)

        if k_actual < k:
            # 如果点数不足 k，用最后一个邻居填充
            pad_idx = topk_idx[:, -1:].expand(-1, k - k_actual)
            pad_dist = topk_dist[:, -1:].expand(-1, k - k_actual)
            topk_idx = torch.cat([topk_idx, pad_idx], dim=1)
            topk_dist = torch.cat([topk_dist, pad_dist], dim=1)

        idx[s_new:e_new] = topk_idx + s_i
        dist_out[s_new:e_new] = topk_dist

    return idx, dist_out


def grouping(feat: torch.Tensor, idx: torch.Tensor, with_xyz: bool = False,
             coord: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    根据索引分组。类似 pointops.grouping。

    Args:
        feat: (N, C) 特征
        idx: (M, K) 邻居索引
        with_xyz: 是否拼接坐标
        coord: (N, 3) 坐标，with_xyz=True 时需要

    Returns:
        grouped: (M, K, C') 分组后的特征 (如果 with_xyz=True, C'=C+3)
    """
    if with_xyz:
        assert coord is not None
        feat = torch.cat([coord, feat], dim=-1)  # (N, 3+C)

    M, K = idx.shape
    C = feat.shape[1]
    idx_flat = idx.reshape(-1).unsqueeze(-1).expand(-1, C)  # (M*K, C)
    grouped = feat.gather(0, idx_flat)  # (M*K, C)
    return grouped.reshape(M, K, C)


# ============================================================================
# Grouped Vector Attention (分组向量注意力)
# ============================================================================

class GroupedVectorAttention(nn.Module):
    """
    Point Transformer V2 的核心: 分组向量注意力

    将特征分成 groups 个组，每个组独立计算注意力，
    然后用注意力权重对 value 进行加权求和。
    """
    def __init__(
        self,
        embed_channels: int,
        groups: int,
        attn_drop_rate: float = 0.0,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
    ):
        super().__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat: torch.Tensor, coord: torch.Tensor,
                reference_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (N, C) 点特征
            coord: (N, 3) 点坐标
            reference_index: (N, K) 邻居索引 (kNN 结果)

        Returns:
            feat: (N, C) 更新后的特征
        """
        query = self.linear_q(feat)   # (N, C)
        key = self.linear_k(feat)     # (N, C)
        value = self.linear_v(feat)   # (N, C)

        # 分组邻居
        key_grouped = grouping(key, reference_index, with_xyz=True, coord=coord)
        # key_grouped: (N, K, 3+C)
        value_grouped = grouping(value, reference_index, with_xyz=False)

        pos, key_grouped = key_grouped[:, :, 0:3], key_grouped[:, :, 3:]

        # 相对位置编码
        relation_qk = key_grouped - query.unsqueeze(1)  # (N, K, C)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value_grouped = value_grouped + peb

        # 注意力权重
        weight = self.weight_encoding(relation_qk)  # (N, K, groups)
        weight = self.attn_drop(self.softmax(weight))

        # 掩码: 对无效邻居 mask 为 0
        mask = torch.sign(reference_index + 1)  # 非零索引 (+1 后为正)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)

        # 分组加权求和
        # value: (N, K, C) → (N, K, groups, C//groups)
        value_grouped = einops.rearrange(
            value_grouped, "n ns (g i) -> n ns g i", g=self.groups
        )
        # weight: (N, K, groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value_grouped, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


# ============================================================================
# Block (注意力 + FFN 残差块)
# ============================================================================

class PTv2Block(nn.Module):
    def __init__(
        self,
        embed_channels: int,
        groups: int,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        enable_checkpoint: bool = False,
    ):
        super().__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


# ============================================================================
# Block Sequence (共享 kNN 索引的 Block 序列)
# ============================================================================

class BlockSequence(nn.Module):
    def __init__(
        self,
        depth: int,
        embed_channels: int,
        groups: int,
        neighbours: int = 16,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        enable_checkpoint: bool = False,
    ):
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PTv2Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # kNN 查询获取邻居索引 (共享给所有 block)
        reference_index, _ = knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


# ============================================================================
# Grid Pool (基于体素划分的下采样)
# ============================================================================

class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)

    将点云划分为网格体素，在每个体素内做池化。
    """
    def __init__(self, in_channels: int, out_channels: int, grid_size: float, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)

        feat = self.act(self.norm(self.fc(feat)))

        # 尝试使用 torch_geometric 的 voxel_grid, 失败则回退
        _use_voxel_grid = False
        if TORCH_GEOMETRIC_AVAILABLE:
            try:
                if start is None:
                    start = segment_csr(
                        coord,
                        torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                        reduce="min",
                    )
                cluster = voxel_grid(
                    pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
                )
                _use_voxel_grid = True
            except ImportError:
                pass

        if not _use_voxel_grid:
            # 纯 PyTorch 替代: 基于坐标量化的网格划分
            # 将坐标除以 grid_size 并 floor, 得到体素索引
            cluster_coord = torch.div(coord, self.grid_size, rounding_mode="floor").int()
            # 为每个 batch 单独分配 cluster
            unique_clusters = []
            for b in range(batch.max().item() + 1):
                mask = batch == b
                bc = cluster_coord[mask]
                # 将3D坐标编码为唯一整数
                bc_offset = bc - bc.min(0)[0]
                bc_codes = bc_offset[:, 0] * 1000000 + bc_offset[:, 1] * 1000 + bc_offset[:, 2]
                _, bc_inverse = torch.unique(bc_codes, sorted=True, return_inverse=True)
                # 偏移以避免不同 batch 的 cluster 号冲突
                offset_shift = 0 if len(unique_clusters) == 0 else unique_clusters[-1].max().item() + 1
                unique_clusters.append(bc_inverse + offset_shift)
            cluster = torch.cat(unique_clusters)

        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        # 池化
        if TORCH_SCATTER_AVAILABLE:
            new_coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
            new_feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        else:
            # 纯 PyTorch 替代
            new_coord = torch.zeros(len(unique), coord.shape[1], device=coord.device)
            new_feat = torch.zeros(len(unique), feat.shape[1], device=feat.device)
            for j in range(len(unique)):
                mask = cluster == j
                new_coord[j] = coord[mask].mean(0)
                new_feat[j] = feat[mask].max(0)[0]
            _, sorted_cluster_indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        new_batch = batch[idx_ptr[:-1]]
        new_offset = batch2offset(new_batch)
        return [new_coord, new_feat, new_offset], cluster


# ============================================================================
# GVAPatchEmbed (初始 Patch 嵌入)
# ============================================================================

class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        embed_channels: int,
        groups: int,
        neighbours: int = 16,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        enable_checkpoint: bool = False,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


# ============================================================================
# Encoder (下采样 + 特征提取)
# ============================================================================

class PTv2Encoder(nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        embed_channels: int,
        groups: int,
        grid_size: float = None,
        neighbours: int = 16,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        enable_checkpoint: bool = False,
    ):
        super().__init__()
        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


# ============================================================================
# Point Transformer V2 Classification + BBox
# ============================================================================

class PointTransformerV2Cls(nn.Module):
    """
    Point Transformer V2 for Classification + 3D BBox

    使用编码器部分提取多尺度特征，全局池化后接分类和框回归头。
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 40,
        box_dim: int = 3,
        patch_embed_depth: int = 1,
        patch_embed_channels: int = 48,
        patch_embed_groups: int = 6,
        patch_embed_neighbours: int = 8,
        enc_depths: Tuple[int, ...] = (2, 2, 6, 2),
        enc_channels: Tuple[int, ...] = (96, 192, 384, 512),
        enc_groups: Tuple[int, ...] = (12, 24, 48, 64),
        enc_neighbours: Tuple[int, ...] = (16, 16, 16, 16),
        grid_sizes: Tuple[float, ...] = (0.06, 0.12, 0.24, 0.48),
        attn_qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        enable_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)

        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(grid_sizes)

        # Patch Embed
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
        )

        # Drop path rates
        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        enc_channels_list = [patch_embed_channels] + list(enc_channels)

        # Encoder stages
        self.enc_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = PTv2Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels_list[i],
                embed_channels=enc_channels_list[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]): sum(enc_depths[:i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            self.enc_stages.append(enc)

        # 分类头: 与 Pointcept 的 seg_head 结构一致 (Linear → BN → ReLU → Linear), 仅在最后一层
        # 接 ``num_classes`` 输出; 增加两层 Dropout(0.5) 与 SPAD 项目其他 baseline 的分类头保持一致。
        final_channels = enc_channels_list[-1]
        self.cls_head = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

        # 中心点回归头 (SPAD 特有 center-only 约定): 输出 [B, 3], 由全局特征预测归一化中心。
        self.center_head = nn.Sequential(
            nn.Linear(final_channels, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, box_dim),
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict: {
                ``coord``: (N, 3) 变长批拼接坐标
                ``feat``: (N, C_in) 变长批拼接特征 (Pointcept 约定 feat 含 coord, 见 SPAD 适配封装)
                ``offset``: (B,) cumsum 形式的 batch 划分
            }

        Returns:
            logits: (B, num_classes)
            center_pred: (B, 3) 归一化空间中心点
        """
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # 数据流: patch_embed 把 in_channels → patch_embed_channels=48; 然后 4 个 encoder 阶段
        # 每阶段 = GridPool 下采样 + BlockSequence 注意力块, 末端 feat 维度 = enc_channels[-1] (默认 512)。
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        for i in range(self.num_stages):
            points, _ = self.enc_stages[i](points)
        coord, feat, offset = points

        # 全局平均池化: 按 offset 把每个样本对应的点特征取均值, 得到 (B, final_channels)。
        # 数据流: feat 形状 (N_final, C_final) → 逐 batch slice [s_i:e_i] → mean(0) → (1, C) → cat → (B, C)。
        pooled = []
        for i in range(len(offset)):
            s_i = 0 if i == 0 else int(offset[i - 1].item())
            e_i = int(offset[i].item())
            cnt = e_i - s_i
            if cnt > 0:
                pooled.append(feat[s_i:e_i].mean(0, keepdim=True))
            else:
                # 极端情况: GridPool 把整个 batch 压成空 — 用零向量占位避免崩溃。
                pooled.append(torch.zeros(1, feat.shape[1], device=feat.device, dtype=feat.dtype))
        feat_global = torch.cat(pooled, dim=0)  # (B, final_channels)

        logits = self.cls_head(feat_global)
        center_pred = self.center_head(feat_global)
        return logits, center_pred


# ============================================================================
# 适配 SPAD 管道的包装类
# ============================================================================

class PointTransV2Classification(PointTransformerV2Cls):
    """SPAD 训练管道适配封装。

    与项目其它 baseline 一致, 接收 (B, N, 4) xyzi 点云, 内部转换为 Pointcept 期望的
    ``data_dict = {coord, feat, offset}``: coord 取前 3 维, feat 直接用整个 4 维向量
    (Pointcept ``in_channels=4`` 时把 xyz+intensity 一并送入 patch_embed 的 Linear)。
    """

    def __init__(self, num_classes: int = 26, **kwargs):
        super().__init__(in_channels=4, num_classes=num_classes, **kwargs)

    @staticmethod
    def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
        """统一输入为 (B, N, 4); 见 PointTransformer.py 中同名函数的实现说明。"""
        if x.ndim != 3:
            raise ValueError(f"PointTransV2Classification 仅接受 3D 输入, 收到 {tuple(x.shape)}")
        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()   # (B, C, N) → (B, N, C)
        else:
            raise ValueError(f"PointTransV2Classification 不支持的形状 {tuple(x.shape)}")
        if points.shape[-1] == 3:
            pad_intensity = torch.zeros(
                points.shape[0], points.shape[1], 1, dtype=points.dtype, device=points.device,
            )
            points = torch.cat([points, pad_intensity], dim=-1)
        return points

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, 4) xyzi 或可被 ``_normalize_input_points`` 识别的等价布局。

        Returns:
            (logits [B, num_classes], center_pred [B, 3])。
        """
        points = self._normalize_input_points(x)
        batch_size, num_points, _ = points.shape

        # 数据流: (B, N, 4) → coord (B*N, 3) + feat (B*N, 4) + offset (B,) = arange step N。
        # Pointcept v2 习惯把全量特征 (含 xyz) 喂入 patch_embed, 故 feat 用整张 (B, N, 4)。
        coord = points[:, :, :3].reshape(batch_size * num_points, 3).contiguous()
        feat = points.reshape(batch_size * num_points, -1).contiguous()
        offset = torch.arange(
            num_points, (batch_size + 1) * num_points, step=num_points,
            dtype=torch.long, device=points.device,
        )

        data_dict = {"coord": coord, "feat": feat, "offset": offset}
        return super().forward(data_dict)


# ============================================================================
# 快速测试 + GPU 显存测试
# ============================================================================

def _quick_test() -> None:
    """简单 forward 形状检查; CPU/CUDA 通用。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing PointTransformer V2 on {device}")

    model = PointTransV2Classification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, center_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Center: {tuple(center_pred.shape)}")
    print("OK PointTransformer V2 works correctly")


def _gpu_memory_test() -> None:
    """逐 batch size 显存扫查 (4/8/16/32)。CPU 环境下直接跳过。"""
    import gc
    if not torch.cuda.is_available():
        print("无 CUDA, 跳过 GPU 显存测试。")
        return

    print("\n=== GPU 显存测试 ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        if total_mem:
            print(f"总显存: {total_mem / 1024 ** 3:.1f} GB")
    except Exception:
        pass
    print()

    num_points = 1024
    for batch_size in [4, 8, 16, 32]:
        try:
            m = PointTransV2Classification(num_classes=26).cuda()
            pts = torch.randn(batch_size, num_points, 4).cuda()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            m.train()
            out = m(pts)
            loss = out[0].sum() + out[1].sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"  B={batch_size:2d}: peak {peak:6.0f} MB")
            del m, pts, out, loss
            torch.cuda.empty_cache()
            gc.collect()
        except torch.cuda.OutOfMemoryError:
            print(f"  B={batch_size:2d}: OOM!")
            torch.cuda.empty_cache()
            gc.collect()
            break


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()

