"""
单光子点云图残差多任务网络 — PyG 图卷积版 (Graph Residual Multi-Task Network, GCN variant)

与 graph_residual.py (v7) 的核心区别:
    原版使用 DGCNN 风格 Conv2d + BN2d 做 EdgeConv ([x_j - x_i, x_i] → MLP);
    本版使用 PyTorch Geometric 的 SAGEConv 做 **真正的消息传递图卷积**:

        h_i' = W_1 · h_i + W_2 · mean_{j ∈ N(i)} h_j     (SAGEConv, mean 聚合)

    两路均替换:
    - GCN_f (特征流): SAGEConv on 特征空间动态 KNN 图 (DGCNN "动态图" 范式保留, 卷积核换成真 GNN)
    - GCN_p (位置流): SAGEConv on 坐标空间静态 KNN 图 (预计算复用, 同原版)

    Block 内改进 (相比原版 GraphResidualBlock):
    - SE-style channel gate 替代原版全局 Q/K/V 注意力 (本版已无 softmax 注意力)
    - 双流 SAGEConv 输出经 SE 加权后直接 cat + Conv1d 融合, 替代 Q/K/V 投影
    - **坐标编码器残差注入**: 2 层 MLP 对 4D 坐标提取位置特征, 通过残差加法注入到最终特征,
      使每个点特征显式包含其空间位置编码, 改善深度回归 (避免全局池化后位置信息丢失)

    坐标门控残差逻辑沿用原版 (详见 readme.md 任务 1)。

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
    coord_gate(p), coord_res(p): gate·fused + (1-gate)·coord_res → out
        ↓
    coord_encoder(p): pos_feat  ← 新增: 2 层 MLP 提取位置特征
        ↓
    f_out = act(out + pos_feat)  ← 坐标编码残差注入

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
from utils.heads import build_standard_cls_head, build_standard_box_head

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

def batched_knn_edge_index(
    knn_idx: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """将 KNN 索引转换为 PyG 格式的 batched edge_index。

    对 batch 内每个样本, 为每点及其 k 个邻居创建有向边 (i→j 表示 j 是 i 的邻居),
    并通过 batch 偏移拼接为全局 edge_index。由于 KNN 结果中每点的最近邻通常包含
    自身 (距离=0), 所得 edge_index 天然含有自环。

    Args:
        knn_idx: (B, N, k) — KNN 索引, 每个元素是 [0, N) 范围内的邻居编号。
        batch_size: 批次大小 B。
        num_nodes: 每样本点数 N。

    Returns:
        edge_index: (2, B*N*k), int64 — PyG 格式的全局边索引。
            row = 中心节点 (i), col = 邻居节点 (j), 均已加上 batch 偏移。

    Shape 推导:
        输入 (B, N, k) → 每样本 N*k 条边 → 全局 B*N*k 条边 → (2, B*N*k)
    """
    device = knn_idx.device
    # batch 偏移: 第 b 个样本的全局节点编号从 b*N 开始
    # (B, 1, 1) 广播到 (B, N, k)
    offset = torch.arange(batch_size, device=device).view(-1, 1, 1) * num_nodes

    # 中心节点 (每点重复 k 次) → 全局编号
    # (B, N, 1) → (B, N, k) → (B*N*k,)
    row = (torch.arange(num_nodes, device=device).view(1, -1, 1) + offset)
    row = row.expand_as(knn_idx).reshape(-1)

    # 邻居节点 → 全局编号
    col = (knn_idx + offset).reshape(-1)

    # PyG 格式: row = source (中心), col = target (邻居)
    return torch.stack([row, col], dim=0)                   # (2, B*N*k)


# ══════════════════════════════════════════════════
# Batched SAGEConv 包装层
# ══════════════════════════════════════════════════

class BatchedSAGEConv(nn.Module):
    """基于 PyG SAGEConv 的批量图卷积包装。

    将 (B, C, N) 格式的点云特征展平为 (B*N, C), 在预计算的 batched edge_index 上
    执行一次 GraphSAGE 消息传递, 再恢复为 (B, C_out, N)。

    SAGEConv 核心公式 (Hamilton et al., NeurIPS 2017, mean 聚合):
        h_i' = W_1 · h_i + W_2 · mean_{j ∈ N(i)} h_j

    相比 DGCNN 的 Conv2d EdgeConv:
        h_i' = MLP( max_{j ∈ N(i)} [h_j - h_i, h_i] )

    关键差异:
        - SAGEConv 是归纳式 (inductive), 可处理未见过的图拓扑;
          DGCNN EdgeConv 依赖 [h_j - h_i, h_i] 拼接, 本质是 edge-level MLP
        - SAGEConv 用 mean 聚合替代 max pool, 保留更多邻域统计信息
        - SAGEConv 有中心-邻居分离权重 (W_1 vs W_2), 表达能力更强

    Args:
        in_channels: 输入特征维度。
        out_channels: 输出特征维度。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)

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
# Graph Residual Block — PyG 图卷积版
# ══════════════════════════════════════════════════

class GraphResidualBlockGCN(nn.Module):
    """图残差模块 — PyG SAGEConv 版 (真消息传递 + Q/K/V 注意力 + 坐标门控)。

    与原版 GraphResidualBlock 的区别:
        原版 GCN_f / GCN_p 使用 Conv2d 对 [x_j - x_i, x_i] 做 edge MLP;
        本版使用 SAGEConv 做真正的邻居聚合消息传递:
            GCN_f: SAGEConv on 特征空间动态 KNN 图 → f_gcn
            GCN_p: SAGEConv on 坐标空间静态 KNN 图 → p_gcn
        之后, f_gcn / p_gcn 分别作为 V / K 的来源, 进入 Q/K/V 注意力。

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
        gate = sigmoid(coord_gate(p)), coord_info = coord_res(p)
        out = gate * fused + (1-gate) * coord_info
            ↓
        pos_feat = coord_encoder(p)  ← 2 层 MLP, 位置编码
        f_out = act(out + pos_feat)   ← 坐标编码残差注入
            ↓
        f_out(B,C_out,N), p 原样传出

    设计说明:
        - p 全程不变 → p_edge_index 在 Net 层预计算一次, 4 个 Block 共享
        - SAGEConv 的 mean 聚合天然保留邻域分布信息, 适合 SPAD 点云的噪声建模
        - SE channel gate 替代全局注意力, 消除 1024×1024 过拟合风险
        - **coord_encoder 残差注入**: 2 层 MLP 对 4D 坐标提取位置特征,
          通过残差加法注入到最终特征, 使每个点特征显式包含其空间位置编码,
          改善深度回归 (避免全局池化后位置信息丢失)

    Args:
        in_channels: C_in。
        out_channels: C_out。
        k: 近邻数。
        downsample: 是否启用 N→N/2 下采样 (当前配置为 False)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 20,
        downsample: bool = True,
    ):
        super().__init__()
        self.k = k
        self.downsample = downsample

        # ── GCN_f: 特征流 SAGEConv ──
        # 对特征空间动态 KNN 图做消息传递 (DGCNN "动态图" 范式, 卷积核换为真 GNN)
        self.gcn_f = BatchedSAGEConv(in_channels, out_channels)
        self.bn_f = nn.BatchNorm1d(out_channels)

        # ── GCN_p: 位置流 SAGEConv ──
        # 对坐标空间静态 KNN 图做消息传递 (输入为 4D 原始坐标 x,y,z,i)
        # SAGEConv 从固定的坐标拓扑中学习几何特征, 替代原版 Conv2d EdgeConv
        self.gcn_p = BatchedSAGEConv(4, out_channels)
        self.bn_p = nn.BatchNorm1d(out_channels)

        # ── SE-style channel gate (替代原版全局注意力) ──
        # SAGEConv 已将邻域聚合为单向量, 全局 (B,N,N) 注意力既嘈杂又过拟合;
        # SE 模块通过通道注意力自适应加权 f_gcn, 计算量可忽略。
        # 结构: GAP → Linear(C→C/4) → ReLU → Linear(C/4→C) → Sigmoid
        se_ratio = 4
        self.se_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels // se_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // se_ratio, out_channels, bias=False),
            nn.Sigmoid(),
        )

        # ── 双流融合 Conv1d ──
        # 将 SE 加权后的 f_gcn 与 p_gcn 拼接后, 用 Conv1d 融合回 out_channels。
        # 替代原版的 Q/K/V 投影 + 注意力 + out_conv, 参数量大幅缩减。
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, 1, bias=False),
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
        self.coord_encoder = nn.Sequential(
            nn.Conv1d(4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

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
        knn_f = knn_gpu(f, k)                                       # (B, N, k)
        f_edge_index = batched_knn_edge_index(knn_f, B, N)          # (2, B*N*k)

        # ── GCN_f: 特征流 SAGEConv → f_gcn ──
        # SAGEConv: h_i' = W_1·f_i + W_2·mean_{j∈N(i)} f_j
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

        # ── 坐标门控跳跃连接 ──
        gate = torch.sigmoid(self.coord_gate(p))                    # (B, C_out, N)
        coord_info = self.coord_res(p)                              # (B, C_out, N)
        out = gate * fused + (1.0 - gate) * coord_info

        # ── 坐标编码残差注入 (改善深度回归) ──
        # 2 层 MLP 对 4D 坐标提取丰富位置特征, 通过残差加法注入到最终特征
        # 使每个点的 f_out[:, :, i] 显式包含 p[:, :, i] 的空间编码
        # 全局池化后位置统计量仍被保留, 避免深度回归 "盲猜" 中心点
        pos_feat = self.coord_encoder(p)                            # (B, C_out, N)
        f_out = self.act(out + pos_feat)

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
        3. SAGEConv 已在邻域内做 mean 聚合, 故 Block 内用 SE channel gate 替代
           原版 Q/K/V 全局注意力 (无 softmax), 计算量更省且避免过拟合

    全程 (B, C, N) 布局, Conv+BN+LeakyReLU。
    通道: 4→32→64→64→128→256→512, 点数: 全程 1024 不下采样。

    Args:
        num_classes: 分类数。
        k: 近邻数 (默认 20, 对齐 DGCNN)。
        use_checkpoint: 梯度检查点 (默认 True)。
        dropout: 头部 Dropout。
        box_dim: bbox 维度。
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 20,
        use_checkpoint: bool = True,
        dropout: float = 0.3,
        box_dim: int = 3,
    ):
        super().__init__()
        self.k = k
        self.box_dim = box_dim
        self.use_checkpoint = use_checkpoint

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
        block_cfg = dict(k=k, downsample=False)
        self.block1 = GraphResidualBlockGCN(32, 64, **block_cfg)
        self.block2 = GraphResidualBlockGCN(64, 64, **block_cfg)
        self.block3 = GraphResidualBlockGCN(64, 128, **block_cfg)
        self.block4 = GraphResidualBlockGCN(128, 256, **block_cfg)

        # 多尺度拼接: cat(b1, b2, b3, b4) → Conv1d 聚合
        cat_dim = 64 + 64 + 128 + 256  # 512
        self.agg_conv = nn.Sequential(
            nn.Conv1d(cat_dim, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        pooled_dim = 1024  # 512 * 2 (max + avg)

        # 统一分类头: 3 层 MLP (1024 → 256 → 128 → num_classes)
        self.cls_head = build_standard_cls_head(pooled_dim, num_classes, dropout=dropout)

        # 统一中心点回归头: 3 层 MLP (1024 → 256 → 128 → box_dim)
        # 直接回归, 与 baseline 一致, 确保 backbone 为唯一变量。
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
        f = self.stem(p)                                        # (B, 32, N)

        # ── 坐标 KNN + edge_index 预计算 (p 全程不变, 4 个 Block 共享) ──
        k = min(self.k, N - 1)
        knn_p = knn_gpu(p, k)                                   # (B, N, k)
        p_edge_index = batched_knn_edge_index(knn_p, B, N)     # (2, B*N*k)

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

        # Box head: 直接回归 (与 baseline 一致)
        # backbone 内 Block 的 coord_encoder 已将位置信息注入到每个点特征中,
        # 全局池化后位置统计量仍被保留, 深度回归不再 "盲猜" 中心点
        box_preds = self.box_head(f_pooled)                       # (B, 3)

        return {"logits": logits, "box_pred": box_preds}


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
