"""
单光子点云图残差多任务网络 (Graph Residual Multi-Task Network for SPAD)

该模块实现了基于"坐标注意力图残差模块"的多任务深度学习模型，
专门针对单光子雪崩二极管 (SPAD) 激光雷达的稀疏点云数据。

物理动机:
    单光子激光雷达在浓雾等复杂场景中采集的点云信噪比极低。
    通用点云网络缺乏对几何位置中心的显式约束，深层特征易因
    噪声干扰发生"位置漂移"。本网络通过"坐标残差"机制使
    每一层图卷积都能直接访问原始三维空间坐标，从而抑制
    噪声引起的特征形变，提高目标分类与 3D 定位精度。

显存优化 (v2):
    - EdgeFeature 分块处理 (chunked): 峰值显存降至 1/chunk
    - Gradient Checkpoint: 训练时以计算换空间
    - 默认 k=16, max_chunk_size=4

核心架构 (参考 model/readme.md 任务1):
    - KNN 动态建图 (纯 PyTorch 实现)
    - 图残差模块: NGF/GCN + 自注意力门控 + 坐标残差
    - 全局双池化 (Max + Avg)
    - 双预测头: 分类 logits + 3D Box 回归

References:
    - model/readme.md 任务1
    - utils/loss.py split_cls_and_box_predictions
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容不同 PyTorch 版本的 checkpoint 导入
try:
    from torch.utils.checkpoint import checkpoint as _torch_checkpoint
    _HAS_CHECKPOINT = True
except (ImportError, AttributeError):
    _HAS_CHECKPOINT = False
    def _torch_checkpoint(fn, *args, **kwargs):
        return fn(*args)


# ═══════════════════════════════════════════════════════════
# KNN 工具函数
# ═══════════════════════════════════════════════════════════

def get_knn_indices(
    points_xyz: torch.Tensor,
    k: int,
    exclude_self: bool = True,
) -> torch.Tensor:
    """计算每个点的 K 近邻索引（纯 PyTorch 实现，无 C++ 扩展依赖）。

    通过批量矩阵乘法计算成对欧氏平方距离，排除自身后取 top-k。
    时间复杂度 O(B·N²·D)，在 N=1024、D=3 时高效运行。

    Args:
        points_xyz: 点云坐标张量，形状 (B, N, 3)。
        k: 近邻数量，应满足 1 ≤ k < N。
        exclude_self: 是否排除自身点。KNN 构图时通常为 True。

    Returns:
        knn_idx: 近邻索引，形状 (B, N, k)，值域 [0, N-1]。
    """
    B, N, _ = points_xyz.shape

    if exclude_self and k >= N:
        raise ValueError(
            f"k ({k}) must be less than N ({N}) when exclude_self=True."
        )

    # ||p_i - p_j||² = ||p_i||² + ||p_j||² - 2·p_i·p_j
    xx = torch.sum(points_xyz ** 2, dim=2, keepdim=True)          # (B, N, 1)
    dist = xx + xx.transpose(2, 1) \
           - 2.0 * torch.bmm(points_xyz, points_xyz.transpose(2, 1))  # (B, N, N)

    if exclude_self:
        diag_mask = torch.eye(
            N, device=points_xyz.device, dtype=torch.bool
        ).unsqueeze(0).expand(B, -1, -1)
        dist.masked_fill_(diag_mask, float("inf"))

    _, knn_idx = torch.topk(dist, k, dim=2, largest=False)  # (B, N, k)
    return knn_idx


# ═══════════════════════════════════════════════════════════
# 边特征提取层 (内存高效版)
# ═══════════════════════════════════════════════════════════

class EdgeFeature(nn.Module):
    """局部邻域图边特征提取层（内存高效分块版），等价于 EdgeConv。

    对每个中心点 i 取 K 个近邻 j，构建边特征 e_{ij} = [f_i, f_j - f_i]，
    通过共享 MLP 变换后最大池化聚合。

    **显存优化**: 沿近邻维度分块循环处理，峰值从 (B,N,k,2C) 降至
    (B,N,chunk,2C)。C=1024 时 ~2.5 GB → ~0.5 GB (chunk=4)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 16,
        max_chunk_size: int = 4,
    ):
        """初始化。

        Args:
            in_channels: 输入特征维度 C_in。
            out_channels: 输出特征维度 C_out。
            k: 近邻数量。
            max_chunk_size: 分块大小。越小越省显存但越慢。
                设为 0 或 ≥ k 则禁用分块。
        """
        super().__init__()
        self.k = k
        self.max_chunk_size = max_chunk_size if max_chunk_size > 0 else k

        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, knn_idx: torch.Tensor
    ) -> torch.Tensor:
        """前向传播：分块提取边特征并跨块取最大值。

        Args:
            x: 点特征，形状 (B, N, C_in)。
            knn_idx: KNN 索引，形状 (B, N, k)。

        Returns:
            out: 聚合后的局部特征，形状 (B, N, C_out)。
        """
        B, N, C = x.shape
        k = knn_idx.shape[-1]
        chunk = min(self.max_chunk_size, k)

        out: Optional[torch.Tensor] = None

        for chunk_start in range(0, k, chunk):
            chunk_end = min(chunk_start + chunk, k)
            kc = chunk_end - chunk_start
            knn_chunk = knn_idx[:, :, chunk_start:chunk_end]  # (B, N, kc)

            # 批量索引 gather 邻居特征
            batch_idx = (
                torch.arange(B, device=x.device)
                .view(B, 1, 1)
                .expand(-1, N, kc)
            )
            x_neighbors = x[batch_idx, knn_chunk, :]          # (B, N, kc, C)
            x_center = x.unsqueeze(2).expand(-1, -1, kc, -1)  # (B, N, kc, C)

            # 边特征 + MLP
            edge_feat = torch.cat(
                [x_center, x_neighbors - x_center], dim=-1
            )  # (B, N, kc, 2C)
            edge_feat = self.mlp(edge_feat)  # (B, N, kc, C_out)

            # 当前 chunk 内 max-pool
            chunk_max = edge_feat.max(dim=2)[0]  # (B, N, C_out)

            # 跨 chunk 累积最大值
            if out is None:
                out = chunk_max
            else:
                out = torch.maximum(out, chunk_max)

            # 及时释放中间张量
            del x_neighbors, x_center, edge_feat, chunk_max, knn_chunk, batch_idx

        if out is None:
            raise RuntimeError("EdgeFeature: k must be >= 1")
        return out


# ═══════════════════════════════════════════════════════════
# 图残差模块
# ═══════════════════════════════════════════════════════════

class GraphResidualBlock(nn.Module):
    r"""图残差模块 —— 网络的核心构建块。

    针对性解决 SPAD 点云两大痛点:
    1. 噪声几何形变 → 坐标残差连接锚定位置
    2. 通道信息不平衡 → 自注意力门控自适应调制

    数据流::

        feat ──→ [LN] ──┬──→ [EdgeFeature] ──→ [ReLU] ──┐
                         │                                  │
                         └──→ [Linear → ReLU → Sigmoid] ───┤
                                                            │
                                   [逐元素点乘 Attention] ←─┘
                                            │
                                       [Linear]
                                            │
                                            ├──→ (+) ←── P_proj
                                            │
                                         [ReLU]
                                            │
                                         Output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 16,
        max_chunk_size: int = 4,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.use_checkpoint = use_checkpoint

        # 坐标投影: 3D → C_out，供残差连接
        self.point_proj = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        # 层归一化
        self.ln = nn.LayerNorm(in_channels)

        # Flow_A: 图特征提取 (NGF → GCN)
        self.edge_feat = EdgeFeature(in_channels, out_channels, k, max_chunk_size)

        # Flow_B: 自注意力权重
        self.attn_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )

        # 融合后精炼
        self.fuse_linear = nn.Linear(out_channels, out_channels)

        # 输出激活
        self.act = nn.ReLU(inplace=True)

    def _forward_impl(
        self,
        feat: torch.Tensor,
        xyz: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """核心前向逻辑（可被 checkpoint 包装）。"""
        # 1. 坐标投影
        P_proj = self.point_proj(xyz)                       # (B, N, C_out)

        # 2. 层归一化
        F_norm = self.ln(feat)                              # (B, N, C_in)

        # 3. Flow_A: 图特征提取 + ReLU
        A = self.edge_feat(F_norm, knn_idx)                 # (B, N, C_out)
        A = torch.relu(A)

        # 4. Flow_B: 自注意力权重 + Sigmoid
        B = self.attn_proj(F_norm)                          # (B, N, C_out)
        B = torch.sigmoid(B)

        # 5. 注意力融合
        fused = A * B                                       # (B, N, C_out)

        # 6. 特征映射
        fused = self.fuse_linear(fused)                     # (B, N, C_out)

        # 7. 坐标残差连接
        out = self.act(fused + P_proj)                      # (B, N, C_out)
        return out

    def forward(
        self,
        feat: torch.Tensor,
        xyz: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播（训练时启用梯度检查点以节省显存）。

        梯度检查点以 ~20% 额外计算为代价，将中间激活存储降至接近 0，
        使 batch_size=8-16 可在 12GB 显卡上训练。
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, feat, xyz, knn_idx, use_reentrant=False
            )
        return self._forward_impl(feat, xyz, knn_idx)


# ═══════════════════════════════════════════════════════════
# 外层模型: 图残差多任务网络
# ═══════════════════════════════════════════════════════════

class GraphResidualMultiTaskNet(nn.Module):
    """基于图残差网络的多任务 SPAD 点云目标检测模型。

    同时执行:
    1. 分类: 预测点云所属类别 (logits)
    2. 3D Box 回归: 预测轴对齐包围盒 [xmin, xmax, ymin, ymax, zmin, zmax]

    通道升维: 64 → 128 → 256 → 512 → 1024
    全局池化: MaxPool + AvgPool → 2048 维
    双预测头: 各自独立的 3 层 MLP

    Args:
        num_classes: 分类类别数。
        k: KNN 近邻数量。
        max_chunk_size: EdgeFeature 分块大小 (0=不分块)。
        use_checkpoint: 是否使用梯度检查点。
        dropout: 预测头 dropout 比例。
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 16,
        max_chunk_size: int = 4,
        use_checkpoint: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.k = k

        # 初始特征嵌入: (B, N, 4) → (B, N, 64)
        self.stem = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # 图残差模块堆叠: 64 → 128 → 256 → 512 → 1024
        block_cfg = dict(
            k=k, max_chunk_size=max_chunk_size, use_checkpoint=use_checkpoint
        )
        self.block1 = GraphResidualBlock(64, 128, **block_cfg)
        self.block2 = GraphResidualBlock(128, 256, **block_cfg)
        self.block3 = GraphResidualBlock(256, 512, **block_cfg)
        self.block4 = GraphResidualBlock(512, 1024, **block_cfg)

        # 全局池化后维度: 1024 + 1024 = 2048
        pooled_dim = 2048

        # 分类预测头
        self.cls_head = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # 3D Box 回归预测头
        self.box_head = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 6),
        )

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """模型前向传播。

        兼容 utils/loss.py split_cls_and_box_predictions 字典解析:
        - logits 通过 key 'logits'
        - box_pred 通过 key 'box_pred'

        Args:
            points: 输入点云，形状 (B, N, 4)，通道 x, y, z, intensity。

        Returns:
            dict: {'logits': (B, num_classes), 'box_pred': (B, 6)}
        """
        B, N, _ = points.shape

        # 分离坐标
        xyz = points[:, :, :3]  # (B, N, 3)

        # 初始特征嵌入
        feat = self.stem(points)  # (B, N, 64)

        # 预计算 KNN 图（所有块复用）
        knn_idx = get_knn_indices(xyz, self.k)  # (B, N, k)

        # 堆叠图残差模块
        feat = self.block1(feat, xyz, knn_idx)  # (B, N, 128)
        feat = self.block2(feat, xyz, knn_idx)  # (B, N, 256)
        feat = self.block3(feat, xyz, knn_idx)  # (B, N, 512)
        feat = self.block4(feat, xyz, knn_idx)  # (B, N, 1024)

        # 全局池化: Max + Avg
        feat_max = feat.max(dim=1)[0]   # (B, 1024)
        feat_avg = feat.mean(dim=1)     # (B, 1024)
        feat_pooled = torch.cat([feat_max, feat_avg], dim=1)  # (B, 2048)

        # 双头预测
        logits = self.cls_head(feat_pooled)     # (B, num_classes)
        box_preds = self.box_head(feat_pooled)  # (B, 6)

        return {"logits": logits, "box_pred": box_preds}


# ═══════════════════════════════════════════════════════════
# 快速验证
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== GraphResidualMultiTaskNet 快速验证 ===\n")

    B, N = 2, 1024
    dummy = torch.randn(B, N, 4)

    # 默认配置 (显存优化)
    model = GraphResidualMultiTaskNet(
        num_classes=26, k=16, max_chunk_size=4, use_checkpoint=True
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f} M")

    # 前向传播
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    print(f"输入:   {dummy.shape}")
    print(f"logits: {out['logits'].shape}")
    print(f"box:    {out['box_pred'].shape}")
    assert out["logits"].shape == (B, 26)
    assert out["box_pred"].shape == (B, 6)

    # 通道验证
    for i, block in enumerate([model.block1, model.block2, model.block3, model.block4], 1):
        c_in = block.edge_feat.mlp[0].in_features // 2
        c_out = block.edge_feat.mlp[-1].out_features
        print(f"  block{i}: {c_in} → {c_out}")

    # 兼容性验证
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
    from utils.loss import split_cls_and_box_predictions
    l, b = split_cls_and_box_predictions(out)
    assert l is not None and b is not None
    assert torch.equal(l, out["logits"])
    assert torch.equal(b, out["box_pred"])
    print("\n全部验证通过! split_cls_and_box_predictions 兼容 OK")
