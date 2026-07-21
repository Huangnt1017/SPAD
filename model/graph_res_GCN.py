"""
单光子点云图残差多任务网络 — PyG 图卷积版 (Graph Residual Multi-Task Network, GCN variant)

与 graph_residual.py (v7) 的核心区别:
    原版使用 DGCNN 风格 Conv2d + BN2d 做 EdgeConv ([x_j - x_i, x_i] → MLP);
    本版使用 PyTorch Geometric 的 SAGEConv 做 **真正的消息传递图卷积**:

        h_i' = W_1 · h_i + W_2 · AGG_{j ∈ N(i)} h_j

    默认使用 max 聚合以突出稀疏目标响应，并从 KNN 中排除中心点自身；PyG 边按
    “邻居 source → 中心 target”构建，保证中心点聚合自己的 KNN 邻居。

    两路均替换:
    - GCN_f (特征流): SAGEConv on 特征空间动态 KNN 图 (DGCNN "动态图" 范式保留, 卷积核换成真 GNN)
    - GCN_p (位置流): SAGEConv on 坐标空间静态 KNN 图 (预计算复用, 同原版)

    Block 内改进 (相比原版 GraphResidualBlock):
    - SE-style channel gate 替代原版全局 Q/K/V 注意力 (本版已无 softmax 注意力)
    - 双流 SAGEConv 输出经 SE 加权后直接 cat + Conv1d 融合, 替代 Q/K/V 投影
    - 显式 feature residual 保留中心点语义，稳定深层优化
    - 2 层坐标编码器与 coord_res 组成门控坐标增量，并由可学习小尺度系数控制注入强度，
      避免原版强制坐标融合压过分类语义

Block 数据流 (全程 (B, C, N) 布局, 无下采样, N=1024):
    f(B,C,N) + p(B,4,N) + p_edge_index(E_total,2) [预计算缓存]
        ↓
    Dynamic KNN from f → idx(B,N,k)   ← 仅特征 KNN 每层重算
    → f_edge_index(E_total,2)
        ↓
    ┌ GCN_f: SAGEConv(f) + BN1d + LReLU → f_gcn(B,C_out,N)   ← SE 加权后融合
    └ GCN_p: SAGEConv(p) + BN1d + LReLU → p_gcn(B,C_out,N)   ← 复用预计算边
        ↓
    SE gate on f_gcn, 然后 cat(f_gcn_gated, p_gcn) → Conv1d → fused
        ↓
    feature_skip(f), coord_gate(p), coord_res(p), coord_encoder(p)
        ↓
    f_out = act(fused + feature_skip + coord_scale·gate·(coord_res + coord_encoder))

References:
    - model/readme.md 任务 1
    - model/graph_residual.py (v7 原版)
    - Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017 (GraphSAGE)
    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from utils.graph_ops import knn_gpu, weighted_downsample
from utils.heads import (
    SegmentationCentroidHead,
    build_standard_box_head,
    build_standard_cls_head,
)

try:
    from torch_geometric.nn import SAGEConv
except ImportError as exc:
    raise ImportError(
        "graph_res_GCN 依赖 torch_geometric (PyG)。\n"
        "安装指南: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n"
        "典型命令 (conda): conda install pyg -c pyg\n"
        "或 (pip):       pip install torch_geometric"
    ) from exc

try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _HAS_CKPT = True
except (ImportError, AttributeError):
    _HAS_CKPT = False

    def _ckpt(fn, *args, **kwargs):
        return fn(*args)


# ══════════════════════════════════════════════════
# 批量 edge_index 构建 (KNN 索引 → PyG 格式)
# ══════════════════════════════════════════════════

def knn_without_self(x: torch.Tensor, k: int) -> torch.Tensor:
    """返回不含中心点自身的 KNN 索引，避免 SAGEConv 根节点被重复聚合。"""
    if x.ndim != 3:
        raise ValueError(f"knn_without_self expects (B, C, N), got {tuple(x.shape)}")
    num_nodes = int(x.shape[-1])
    if num_nodes < 2:
        raise ValueError("GraphResidual GCN requires at least 2 points per sample.")
    effective_k = min(int(k), num_nodes - 1)
    if effective_k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    candidate_count = min(num_nodes, effective_k + 1)
    candidates = knn_gpu(x, candidate_count)
    centers = torch.arange(num_nodes, device=x.device).view(1, num_nodes, 1)
    positions = torch.arange(candidate_count, device=x.device).view(1, 1, candidate_count)
    priorities = positions + candidates.eq(centers).to(positions.dtype) * candidate_count
    keep_order = priorities.argsort(dim=-1)[..., :effective_k]
    return candidates.gather(dim=-1, index=keep_order)

def batched_knn_edge_index(
    knn_idx: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """将 KNN 索引转换为 PyG 格式的 batched edge_index。

    对 batch 内每个样本，为中心点 i 的每个邻居 j 创建 j→i 边。PyG 默认采用
    source_to_target 消息流，因此该方向保证中心点 i 聚合自己的 KNN 邻居。

    Args:
        knn_idx: (B, N, k) — KNN 索引, 每个元素是 [0, N) 范围内的邻居编号。
        batch_size: 批次大小 B。
        num_nodes: 每样本点数 N。

    Returns:
        edge_index: (2, B*N*k), int64 — PyG 格式的全局边索引。
            edge_index[0] = 邻居源节点 j，edge_index[1] = 中心目标节点 i。

    Shape 推导:
        输入 (B, N, k) → 每样本 N*k 条边 → 全局 B*N*k 条边 → (2, B*N*k)
    """
    if knn_idx.ndim != 3:
        raise ValueError(f"knn_idx must have shape (B, N, k), got {tuple(knn_idx.shape)}")
    if knn_idx.shape[0] != batch_size or knn_idx.shape[1] != num_nodes:
        raise ValueError(
            f"knn_idx shape {tuple(knn_idx.shape)} does not match "
            f"batch_size={batch_size}, num_nodes={num_nodes}"
        )
    if knn_idx.numel() == 0:
        raise ValueError("knn_idx must contain at least one edge")

    device = knn_idx.device
    # batch 偏移: 第 b 个样本的全局节点编号从 b*N 开始
    # (B, 1, 1) 广播到 (B, N, k)
    offset = torch.arange(batch_size, device=device).view(-1, 1, 1) * num_nodes

    # 中心节点 (每点重复 k 次) → 全局编号
    # (B, N, 1) → (B, N, k) → (B*N*k,)
    centers = (torch.arange(num_nodes, device=device).view(1, -1, 1) + offset)
    centers = centers.expand_as(knn_idx).reshape(-1)

    # 邻居节点 → 全局编号
    neighbors = (knn_idx + offset).reshape(-1)

    # PyG 默认 source_to_target：邻居 j 发送消息，中心 i 负责聚合。
    return torch.stack([neighbors, centers], dim=0)         # (2, B*N*k)


# ══════════════════════════════════════════════════
# Batched SAGEConv 包装层
# ══════════════════════════════════════════════════

class BatchedSAGEConv(nn.Module):
    """基于 PyG SAGEConv 的批量图卷积包装。

    将 (B, C, N) 格式的点云特征展平为 (B*N, C), 在预计算的 batched edge_index 上
    执行一次 GraphSAGE 消息传递, 再恢复为 (B, C_out, N)。

    SAGEConv 核心公式 (Hamilton et al., NeurIPS 2017):
        h_i' = W_1 · h_i + W_2 · AGG_{j ∈ N(i)} h_j

    相比 DGCNN 的 Conv2d EdgeConv:
        h_i' = MLP( max_{j ∈ N(i)} [h_j - h_i, h_i] )

    关键差异:
        - SAGEConv 是归纳式 (inductive), 可处理未见过的图拓扑;
          DGCNN EdgeConv 依赖 [h_j - h_i, h_i] 拼接, 本质是 edge-level MLP
        - 聚合器可配置；SPAD 稀疏目标默认用 max，减少烟雾背景的均值稀释
        - SAGEConv 有中心-邻居分离权重 (W_1 vs W_2), 表达能力更强

    Args:
        in_channels: 输入特征维度。
        out_channels: 输出特征维度。
        aggregation: PyG SAGEConv 聚合方式，支持 mean/max。
    """

    def __init__(self, in_channels: int, out_channels: int, aggregation: str = "max"):
        super().__init__()
        if aggregation not in {"mean", "max"}:
            raise ValueError(f"aggregation must be 'mean' or 'max', got {aggregation}")
        self.conv = SAGEConv(in_channels, out_channels, aggr=aggregation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """批量 SAGEConv 前向。

        Args:
            x: (B, C_in, N) — channel-first 格式。
            edge_index: (2, E_total) — batched PyG 边索引, E_total = B*N*k。

        Returns:
            (B, C_out, N) — 图卷积输出。
        """
        B, C_in, N = x.shape

        # (B, C_in, N) → (B, N, C_in) → (B*N, C_in): PyG 要求 (num_nodes, features)
        x_flat = x.permute(0, 2, 1).reshape(B * N, C_in)

        # SAGEConv 消息传递: 一次调用处理整个 batched graph
        out = self.conv(x_flat, edge_index)                    # (B*N, C_out)

        # (B*N, C_out) → (B, N, C_out) → (B, C_out, N): 恢复 channel-first
        return out.view(B, N, -1).permute(0, 2, 1).contiguous()


# ══════════════════════════════════════════════════
# Batched EdgeCNN 包装层（GraphSAGE 的参数匹配对照）
# ══════════════════════════════════════════════════

class BatchedEdgeCNNConv(nn.Module):
    """在同一 KNN 图上执行 DGCNN 风格的边卷积对照。

    每条邻居边 ``j -> i`` 构造 ``[x_j - x_i, x_i]``，使用共享线性层
    （等价于 edge feature 上的 1×1 CNN）生成消息，再按目标节点聚合：

        m_ji = W [x_j - x_i, x_i] + b
        h_i' = AGG_{j in N(i)} m_ji

    ``Linear(2*C_in, C_out)`` 的参数量与默认 ``SAGEConv(C_in, C_out)``
    完全一致，因此可在保持 KNN、双分支、残差和融合结构不变的情况下，比较
    EdgeCNN 与 GraphSAGE 消息传递算子的差异。
    """

    def __init__(self, in_channels: int, out_channels: int, aggregation: str = "max"):
        super().__init__()
        if aggregation not in {"mean", "max"}:
            raise ValueError(f"aggregation must be 'mean' or 'max', got {aggregation}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.aggregation = aggregation
        self.edge_cnn = nn.Linear(2 * self.in_channels, self.out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """执行参数匹配的批量 EdgeCNN，输入输出布局与 BatchedSAGEConv 一致。"""
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, C, N), got {tuple(x.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
            )
        batch_size, channels, num_points = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {channels}"
            )

        x_flat = x.permute(0, 2, 1).reshape(batch_size * num_points, channels)
        sources, targets = edge_index[0], edge_index[1]
        source_features = x_flat.index_select(0, sources)
        target_features = x_flat.index_select(0, targets)
        edge_features = torch.cat(
            [source_features - target_features, target_features],
            dim=1,
        )
        messages = self.edge_cnn(edge_features)
        node_count = batch_size * num_points

        if self.aggregation == "max":
            output = messages.new_full(
                (node_count, self.out_channels),
                -torch.inf,
            )
            scatter_index = targets.unsqueeze(1).expand_as(messages)
            output.scatter_reduce_(
                0,
                scatter_index,
                messages,
                reduce="amax",
                include_self=True,
            )
            output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        else:
            output = messages.new_zeros((node_count, self.out_channels))
            output.index_add_(0, targets, messages)
            counts = messages.new_zeros((node_count, 1))
            counts.index_add_(
                0,
                targets,
                messages.new_ones((messages.shape[0], 1)),
            )
            output = output / counts.clamp_min(1.0)

        return output.view(batch_size, num_points, self.out_channels).permute(
            0, 2, 1
        ).contiguous()


# ══════════════════════════════════════════════════
# Graph Residual Block — PyG 图卷积版
# ══════════════════════════════════════════════════

class GraphResidualBlockGCN(nn.Module):
    """图残差模块 — PyG SAGEConv 版 (真消息传递 + SE 门控 + 双残差)。

    与原版 GraphResidualBlock 的区别:
        原版 GCN_f / GCN_p 使用 Conv2d 对 [x_j - x_i, x_i] 做 edge MLP;
        本版使用 SAGEConv 做真正的邻居聚合消息传递:
            GCN_f: SAGEConv on 特征空间动态 KNN 图 → f_gcn
            GCN_p: SAGEConv on 坐标空间静态 KNN 图 → p_gcn
        之后, f_gcn 经 SE 通道门控后与 p_gcn 融合。

    数据流 (全程 (B, C, N) 布局, 无下采样, N 不变):
        f(B,C_in,N) + p(B,4,N) + p_edge_index(2,E) [外部预计算缓存]
            ↓
        Dynamic KNN from f → idx(B,N,k)    ← 仅特征 KNN 每层重算
        → f_edge_index(2,E)
            ↓
        ┌ GCN_f: BatchedSAGEConv(f) + BN1d + LReLU → f_gcn(B,C_out,N)
        └ GCN_p: BatchedSAGEConv(p) + BN1d + LReLU → p_gcn(B,C_out,N)
            ↓
        SE gate: f_gap = GAP(f_gcn) → MLP → sigmoid → se_weight
        f_gcn_gated = f_gcn * se_weight
        fused = Conv1d(cat(f_gcn_gated, p_gcn))
            ↓
        feature_skip = projection(f)
        coord_delta = sigmoid(coord_gate(p)) * (coord_res(p) + coord_encoder(p))
        f_out = act(fused + feature_skip + coord_scale * coord_delta)
            ↓
        f_out(B,C_out,N), p 原样传出

    设计说明:
        - p 全程不变 → p_edge_index 在 Net 层预计算一次, 4 个 Block 共享
        - SAGEConv 默认 max 聚合，减少大量烟雾背景点对稀疏目标响应的均值稀释
        - KNN 默认排除中心点自身，避免与 SAGEConv 的 root transform 重复计入
        - SE channel gate 替代全局注意力, 消除 1024×1024 过拟合风险
        - 显式 feature residual 保留中心点语义并改善梯度传播
        - 坐标增量由可学习 coord_scale 从较小值开始注入，兼顾分类与定位

    Args:
        in_channels: C_in。
        out_channels: C_out。
        k: 近邻数。
        downsample: 是否启用 N→N/2 下采样 (当前配置为 False)。
        aggregation: SAGEConv 聚合方式，默认 max。
        exclude_self: 是否从 KNN 中排除中心点自身。
        feature_residual: 是否启用显式特征残差。
        coord_scale_init: 可学习坐标残差缩放的初值。
        legacy_mode: 是否复现旧边方向、坐标融合与参数结构。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 20,
        downsample: bool = True,
        se_ratio: int = 4,
        fuse_bottleneck_ratio: int = 1,
        coord_mid_ratio: int = 1,
        aggregation: str = "max",
        exclude_self: bool = True,
        feature_residual: bool = True,
        coord_scale_init: float = 0.1,
        legacy_mode: bool = False,
    ):
        super().__init__()
        self.k = k
        self.downsample = downsample
        self.se_ratio = int(se_ratio)
        self.fuse_bottleneck_ratio = int(fuse_bottleneck_ratio)
        self.coord_mid_ratio = int(coord_mid_ratio)
        self.exclude_self = bool(exclude_self)
        self.use_feature_residual = bool(feature_residual)
        self.legacy_mode = bool(legacy_mode)
        if self.se_ratio <= 0:
            raise ValueError(f"se_ratio must be positive, got {se_ratio}")
        if self.fuse_bottleneck_ratio <= 0:
            raise ValueError(f"fuse_bottleneck_ratio must be positive, got {fuse_bottleneck_ratio}")
        if self.coord_mid_ratio <= 0:
            raise ValueError(f"coord_mid_ratio must be positive, got {coord_mid_ratio}")

        # ── GCN_f: 特征流 SAGEConv ──
        # 对特征空间动态 KNN 图做消息传递 (DGCNN "动态图" 范式, 卷积核换为真 GNN)
        self.gcn_f = BatchedSAGEConv(in_channels, out_channels, aggregation=aggregation)
        self.bn_f = nn.BatchNorm1d(out_channels)

        # ── GCN_p: 位置流 SAGEConv ──
        # 对坐标空间静态 KNN 图做消息传递 (输入为 4D 原始坐标 x,y,z,i)
        # SAGEConv 从固定的坐标拓扑中学习几何特征, 替代原版 Conv2d EdgeConv
        self.gcn_p = BatchedSAGEConv(4, out_channels, aggregation=aggregation)
        self.bn_p = nn.BatchNorm1d(out_channels)

        # ── SE-style channel gate (替代原版全局注意力) ──
        # SAGEConv 已将邻域聚合为单向量, 全局 (B,N,N) 注意力既嘈杂又过拟合;
        # SE 模块通过通道注意力自适应加权 f_gcn, 计算量可忽略。
        # 结构: GAP → Linear(C→C/se_ratio) → ReLU → Linear(C/se_ratio→C) → Sigmoid
        se_hidden = max(out_channels // self.se_ratio, 1)
        self.se_gate = nn.Sequential(
            nn.Linear(out_channels, se_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(se_hidden, out_channels, bias=False),
            nn.Sigmoid(),
        )

        # ── 双流融合 Conv1d ──
        # 将 SE 加权后的 f_gcn 与 p_gcn 拼接后, 用 Conv1d 融合回 out_channels。
        # 替代原版的 Q/K/V 投影 + 注意力 + out_conv, 参数量大幅缩减。
        if self.fuse_bottleneck_ratio == 1:
            self.fuse_conv = nn.Sequential(
                nn.Conv1d(2 * out_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            fuse_mid = max(out_channels // self.fuse_bottleneck_ratio, 16)
            self.fuse_conv = nn.Sequential(
                nn.Conv1d(2 * out_channels, fuse_mid, 1, bias=False),
                nn.BatchNorm1d(fuse_mid),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(fuse_mid, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        # 坐标门控 + 坐标残差 (4D: x,y,z,i)
        self.coord_gate = nn.Sequential(
            nn.Conv1d(4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.coord_res = nn.Sequential(
            nn.Conv1d(4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # 坐标编码器 (2 层 MLP): 对原始 4D 坐标提取更丰富的位置特征
        # 通过残差加法注入到最终特征中, 使每个点特征显式包含其空间位置编码
        # 全局池化后, 位置统计量仍被保留, 改善深度回归 (避免 "盲猜" 中心点)
        coord_mid = out_channels if self.coord_mid_ratio == 1 else max(out_channels // self.coord_mid_ratio, 16)
        self.coord_encoder = nn.Sequential(
            nn.Conv1d(4, coord_mid, 1, bias=False),
            nn.BatchNorm1d(coord_mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(coord_mid, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        if self.use_feature_residual and not self.legacy_mode:
            if in_channels == out_channels:
                self.feature_residual = nn.Identity()
            else:
                self.feature_residual = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
        else:
            self.feature_residual = None
        if coord_scale_init < 0:
            raise ValueError(f"coord_scale_init must be non-negative, got {coord_scale_init}")
        if not self.legacy_mode:
            self.coord_scale = nn.Parameter(torch.tensor(float(coord_scale_init)))

        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        p_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block 前向。

        Args:
            p: (B, 4, N) 原始坐标+intensity。
            f: (B, C_in, N) 当前层特征。
            p_edge_index: (2, B*N*k) 坐标图的 batched PyG 边索引
                (Net 层预计算, 4 个 Block 共享)。

        Returns:
            (p, f_out): p 原样传出 (当前配置不下采样),
                f_out 为升维后特征 (B, C_out, N)。
        """
        B, C_in, N = f.shape
        k = min(self.k, N - 1)

        # ── 特征空间 KNN (每层重算, 语义驱动动态图) ──
        knn_f = knn_without_self(f, k) if self.exclude_self else knn_gpu(f, k)
        f_edge_index = batched_knn_edge_index(knn_f, B, N)          # (2, B*N*k)
        if self.legacy_mode:
            f_edge_index = f_edge_index.flip(0)

        # ── GCN_f: 特征流 SAGEConv → f_gcn ──
        # SAGEConv: h_i' = W_1·f_i + W_2·AGG_{j∈N(i)} f_j
        f_gcn = self.gcn_f(f, f_edge_index)                         # (B, C_out, N)
        f_gcn = self.act(self.bn_f(f_gcn))                          # (B, C_out, N)

        # ── GCN_p: 位置流 SAGEConv → p_gcn (复用预计算的坐标边) ──
        # 输入为 4D 原始坐标, SAGEConv 从固定拓扑中学习几何结构特征
        p_gcn = self.gcn_p(p, p_edge_index)                         # (B, C_out, N)
        p_gcn = self.act(self.bn_p(p_gcn))                          # (B, C_out, N)

        # ── SE-style channel gate (替代全局注意力) ──
        # 对 f_gcn 做全局平均池化 → MLP → sigmoid → 通道加权
        # (B, C_out, N) → (B, C_out) → (B, C_out) → (B, C_out, 1) → broadcast
        f_gap = f_gcn.mean(dim=-1)                                  # (B, C_out)
        se_weight = self.se_gate(f_gap).unsqueeze(-1)               # (B, C_out, 1)
        f_gcn_gated = f_gcn * se_weight                             # (B, C_out, N)

        # ── 双流融合 ──
        # cat([f_gcn_gated, p_gcn]) → Conv1d → fused
        fused = self.fuse_conv(torch.cat([f_gcn_gated, p_gcn], dim=1))  # (B, C_out, N)

        # ── 受控坐标残差 + 显式特征残差 ──
        gate = torch.sigmoid(self.coord_gate(p))                    # (B, C_out, N)
        coord_info = self.coord_res(p)                              # (B, C_out, N)
        pos_feat = self.coord_encoder(p)                            # (B, C_out, N)
        if self.legacy_mode:
            out = gate * fused + (1.0 - gate) * coord_info
            f_out = self.act(out + pos_feat)
            if self.downsample:
                p, f_out = weighted_downsample(p, f_out, N // 2)
            return p, f_out

        coord_delta = gate * (coord_info + pos_feat)
        feature_skip = self.feature_residual(f) if self.feature_residual is not None else 0.0
        f_out = self.act(fused + feature_skip + self.coord_scale * coord_delta)

        # ── 层间下采样 ──
        if self.downsample:
            p, f_out = weighted_downsample(p, f_out, N // 2)
        return p, f_out


# ══════════════════════════════════════════════════
# 外层多任务网络 — PyG 图卷积版
# ══════════════════════════════════════════════════

class GraphResidualMultiTaskNetGCN(nn.Module):
    """图残差多任务网络 — PyG SAGEConv 版。

    与原版 GraphResidualMultiTaskNet (v7) 的区别:
        1. Block 内 GCN_f / GCN_p 使用 PyG SAGEConv (真消息传递)
           替代 DGCNN Conv2d EdgeConv
        2. p_graph 缓存从 (B, 8, N, k) 边特征变为 (2, B*N*k) PyG 边索引
        3. SAGEConv 已在邻域内完成可配置聚合, 故 Block 内用 SE channel gate 替代
           原版 Q/K/V 全局注意力 (无 softmax), 计算量更省且避免过拟合

    全程 (B, C, N) 布局, Conv+BN+LeakyReLU。
    通道: 4→32→64→64→128→256→512, 点数: 全程 1024 不下采样。

    Args:
        num_classes: 分类数。
        k: 近邻数 (默认 20, 对齐 DGCNN)。
        use_checkpoint: 梯度检查点 (默认 True)。
        dropout: 头部 Dropout。
        box_dim: bbox 维度。
        seg_centroid_box: 是否启用分割引导质心头替代 MLP 回归头 (自研创新点,
            默认 True)。置 False 时退回与 baseline 一致的统一 MLP 回归头,
            便于消融对比。
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 20,
        use_checkpoint: bool = True,
        dropout: float = 0.3,
        box_dim: int = 3,
        seg_centroid_box: bool = True,
        aggregation: str = "max",
        exclude_self: bool = True,
        feature_residual: bool = True,
        coord_scale_init: float = 0.1,
        legacy_mode: bool = False,
        block_channels: Tuple[int, int, int, int] = (64, 64, 128, 256),
        agg_channels: int = 512,
        block_se_ratio: int = 4,
        fuse_bottleneck_ratio: int = 1,
        coord_mid_ratio: int = 1,
    ):
        super().__init__()
        self.k = int(k)
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.box_dim = box_dim
        self.use_checkpoint = use_checkpoint
        self.aggregation = aggregation
        self.exclude_self = bool(exclude_self)
        self.use_feature_residual = bool(feature_residual)
        self.coord_scale_init = float(coord_scale_init)
        self.legacy_mode = bool(legacy_mode)
        self.seg_centroid_box = bool(seg_centroid_box)
        self.block_channels = tuple(int(c) for c in block_channels)
        self.agg_channels = int(agg_channels)
        if len(self.block_channels) != 4:
            raise ValueError(f"block_channels must contain 4 values, got {self.block_channels}")
        if any(c <= 0 for c in self.block_channels):
            raise ValueError(f"block_channels must be positive, got {self.block_channels}")
        if self.agg_channels <= 0:
            raise ValueError(f"agg_channels must be positive, got {agg_channels}")
        c1, c2, c3, c4 = self.block_channels

        # Stem: (B, 4, N) → (B, 32, N)
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )

        # 无下采样: 全部 1024 点贯穿 4 层, p 不变只升维 f
        block_cfg = dict(
            k=k,
            downsample=False,
            se_ratio=block_se_ratio,
            fuse_bottleneck_ratio=fuse_bottleneck_ratio,
            coord_mid_ratio=coord_mid_ratio,
            aggregation=aggregation,
            exclude_self=exclude_self,
            feature_residual=feature_residual,
            coord_scale_init=coord_scale_init,
            legacy_mode=legacy_mode,
        )
        self.block1 = GraphResidualBlockGCN(32, c1, **block_cfg)
        self.block2 = GraphResidualBlockGCN(c1, c2, **block_cfg)
        self.block3 = GraphResidualBlockGCN(c2, c3, **block_cfg)
        self.block4 = GraphResidualBlockGCN(c3, c4, **block_cfg)

        # 多尺度拼接: cat(b1, b2, b3, b4) → Conv1d 聚合
        cat_dim = sum(self.block_channels)
        self.agg_conv = nn.Sequential(
            nn.Conv1d(cat_dim, self.agg_channels, 1, bias=False),
            nn.BatchNorm1d(self.agg_channels),
            nn.LeakyReLU(0.2),
        )

        pooled_dim = self.agg_channels * 2  # max + avg

        # 统一分类头: 3 层 MLP (1024 → 256 → 128 → num_classes)
        self.cls_head = build_standard_cls_head(pooled_dim, num_classes, dropout=dropout)

        # Box 头二选一:
        #   seg_centroid_box=True  → 分割引导质心头 (自研创新点, 逐点打分 → 加权质心)
        #   seg_centroid_box=False → 统一 MLP 回归头 (与 baseline 一致, 消融对照)
        if self.seg_centroid_box:
            # 逐点特征通道 = agg_conv 输出通道 (非池化维度)
            self.box_head = SegmentationCentroidHead(in_channels=self.agg_channels, coord_dim=box_dim)
        else:
            self.box_head = build_standard_box_head(pooled_dim, box_dim=box_dim, dropout=dropout)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 4) — x, y, z, intensity。

        Returns:
            dict: 'logits' (B, C), 'box_pred' (B, box_dim)。
        """
        # (B, N, 4) → (B, 4, N) 全程 channel-first
        p = points.transpose(1, 2).contiguous()                # (B, 4, N)
        B, _, N = p.shape
        if N < 2:
            raise ValueError(f"GraphResidual GCN requires at least 2 points, got N={N}")
        f = self.stem(p)                                        # (B, 32, N)

        # ── 坐标 KNN + edge_index 预计算 (p 全程不变, 4 个 Block 共享) ──
        k = min(self.k, N - 1)
        knn_p = knn_without_self(p, k) if self.exclude_self else knn_gpu(p, k)
        p_edge_index = batched_knn_edge_index(knn_p, B, N)     # (2, B*N*k)
        if self.legacy_mode:
            p_edge_index = p_edge_index.flip(0)

        # 4 层无下采样, 全程 N=1024; 仅特征 KNN 每层重算
        use_ckpt = self.use_checkpoint and self.training and _HAS_CKPT

        def _run_block(block, _p, _f, _pe):
            return block(_p, _f, _pe)

        if use_ckpt:
            p, f1 = _ckpt(_run_block, self.block1, p, f, p_edge_index, use_reentrant=False)
            p, f2 = _ckpt(_run_block, self.block2, p, f1, p_edge_index, use_reentrant=False)
            p, f3 = _ckpt(_run_block, self.block3, p, f2, p_edge_index, use_reentrant=False)
            p, f4 = _ckpt(_run_block, self.block4, p, f3, p_edge_index, use_reentrant=False)
        else:
            p, f1 = self.block1(p, f, p_edge_index)             # 32→64
            p, f2 = self.block2(p, f1, p_edge_index)            # 64→64
            p, f3 = self.block3(p, f2, p_edge_index)            # 64→128
            p, f4 = self.block4(p, f3, p_edge_index)            # 128→256

        # 多尺度拼接 + 聚合
        f = self.agg_conv(torch.cat([f1, f2, f3, f4], dim=1))  # (B, 512, N)

        # 全局池化: max + avg → (B, 1024)
        f_max = f.max(dim=-1)[0]
        f_avg = f.mean(dim=-1)
        f_pooled = torch.cat([f_max, f_avg], dim=1)              # (B, 1024)

        logits = self.cls_head(f_pooled)

        if self.seg_centroid_box:
            # 分割引导质心头 (自研创新点): 逐点目标性打分 → softmax 权重
            # → 用真实点 xyz 凸组合求质心。SAGEConv 聚合 + 受控坐标残差
            # 注入使逐点特征对 "目标 vs 雾" 有强判别力, 质心头据此聚焦目标点。
            # 输入点坐标 xyz: (B, N, 4) → (B, 3, N) channel-first
            point_xyz = points[..., :3].transpose(1, 2).contiguous()  # (B, 3, N)
            centroid_out = self.box_head(f, point_xyz)
            return {
                "logits": logits,
                "box_pred": centroid_out["centroid"],             # (B, 3)
                "seg_logits": centroid_out["seg_logits"],         # (B, N) 可选辅助监督
                "seg_weights": centroid_out["seg_weights"],       # (B, N) 可视化
            }

        # 退回统一 MLP 回归头 (消融用, 与 baseline 一致)
        box_preds = self.box_head(f_pooled)                       # (B, 3)
        return {"logits": logits, "box_pred": box_preds}


class GraphResidualMultiTaskNetGCNLite(GraphResidualMultiTaskNetGCN):
    """轻量版 GraphResidual GCN。

    设计目标是压缩 block4、聚合通道和图邻域计算, 同时保留四层多尺度结构与
    分割引导质心头。默认配置:
        - k: 20 → 16, 减少消息传递边数
        - block_channels: (64, 64, 128, 256) → (64, 64, 128, 192)
        - agg_channels: 512 → 384
        - SE ratio: 4 → 8
        - fuse_conv: 2C→C 改为 2C→C/4→C bottleneck
        - coord_encoder: 4→C→C 改为 4→C/2→C
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 16,
        use_checkpoint: bool = True,
        dropout: float = 0.3,
        box_dim: int = 3,
        seg_centroid_box: bool = True,
        aggregation: str = "max",
        exclude_self: bool = True,
        feature_residual: bool = True,
        coord_scale_init: float = 0.1,
        legacy_mode: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            k=k,
            use_checkpoint=use_checkpoint,
            dropout=dropout,
            box_dim=box_dim,
            seg_centroid_box=seg_centroid_box,
            aggregation=aggregation,
            exclude_self=exclude_self,
            feature_residual=feature_residual,
            coord_scale_init=coord_scale_init,
            legacy_mode=legacy_mode,
            block_channels=(64, 64, 128, 192),
            agg_channels=384,
            block_se_ratio=8,
            fuse_bottleneck_ratio=4,
            coord_mid_ratio=2,
        )


# ══════════════════════════════════════════════════
# 验证 + GPU 显存测试
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    print("=== GraphResidualMultiTaskNetGCN (PyG SAGEConv 版) ===\n")

    B, N = 2, 1024
    dummy = torch.randn(B, N, 4)

    model = GraphResidualMultiTaskNetGCN(
        num_classes=26, k=20, use_checkpoint=True, box_dim=3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f} M")

    model.eval()
    with torch.no_grad():
        out = model(dummy)

    print(f"输入:   {dummy.shape}")
    print(f"logits: {out['logits'].shape}")
    print(f"box:    {out['box_pred'].shape}")
    assert out["logits"].shape == (B, 26)
    assert out["box_pred"].shape == (B, 3)

    for i, blk in enumerate(
        [model.block1, model.block2, model.block3, model.block4], 1,
    ):
        c_out = blk.gcn_f.conv.out_channels
        ds = "Y" if blk.downsample else "N"
        print(f"  block{i}: C_out={c_out:3d} downsample={ds}")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.loss import split_cls_and_box_predictions
    logits_o, box_o = split_cls_and_box_predictions(out)
    assert logits_o is not None and box_o is not None
    print(f"\n  split_cls_and_box_predictions: logits {logits_o.shape}, box {box_o.shape}")
    print("\nAll checks pass.")

    # ── GPU 显存测试 ──
    print("\n=== GPU 显存测试 ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        try:
            props = torch.cuda.get_device_properties(0)
            total_gb = getattr(props, "total_memory", 0) / 1024 ** 3
            if total_gb > 0:
                print(f"总显存: {total_gb:.1f} GB")
        except Exception:
            pass
        print()

        import gc
        for bs in [4, 8, 16, 32]:
            try:
                m = GraphResidualMultiTaskNetGCN(
                    num_classes=26, k=20, use_checkpoint=True, box_dim=3,
                ).cuda()
                pts = torch.randn(bs, N, 4).cuda()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                m.train()
                o = m(pts)
                loss = o["logits"].sum() + o["box_pred"].sum()
                loss.backward()
                peak = torch.cuda.max_memory_allocated() / 1024 ** 2
                print(f"  B={bs:2d}: peak {peak:6.0f} MB")
                del m, pts, o, loss
                torch.cuda.empty_cache()
                gc.collect()
            except torch.cuda.OutOfMemoryError:
                print(f"  B={bs:2d}: OOM!")
                torch.cuda.empty_cache()
                gc.collect()
                break
    else:
        print("无 CUDA，跳过。")
