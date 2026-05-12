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

核心架构 (参考 model/readme.md 任务1):
    - KNN 动态建图 (纯 PyTorch 实现, 无需 C++ 扩展)
    - 图残差模块 (GraphResidualBlock): NGF/GCN + 自注意力门控 + 坐标残差
    - 全局双池化 (Max + Avg)
    - 双预测头: 分类 logits + 3D Box 回归

References:
    - model/readme.md 任务1
    - utils/loss.py split_cls_and_box_predictions
    - .github/skills/pointcloud-3d-workflows/SKILL.md
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# KNN 工具函数
# ══════════════════════════════════════════════════════════════════════

def get_knn_indices(
    points_xyz: torch.Tensor,
    k: int,
    exclude_self: bool = True,
) -> torch.Tensor:
    """计算每个点的 K 近邻索引（纯 PyTorch 实现，无 C++ 扩展依赖）。

    通过批量矩阵乘法和广播计算所有点对的欧氏平方距离，
    排除自身后选取 top-k 最近邻。时间复杂度 O(B·N²·D)，
    在 N=1024、D=3 时可高效运行。

    Args:
        points_xyz: 点云坐标张量，形状 (B, N, 3)。
        k: 近邻数量，应满足 1 ≤ k < N。
        exclude_self: 是否排除自身点（自身距离为 0）。KNN 构图时通常为 True。

    Returns:
        knn_idx: 近邻索引，形状 (B, N, k)，dtype=torch.int64，值域 [0, N-1]。

    Raises:
        ValueError: 若 k >= N 且 exclude_self=True，则无法选取足够数量的邻居。

    Example:
        >>> xyz = torch.randn(2, 1024, 3)
        >>> idx = get_knn_indices(xyz, k=20)
        >>> idx.shape
        torch.Size([2, 1024, 20])
    """
    B, N, _ = points_xyz.shape

    if exclude_self and k >= N:
        raise ValueError(
            f"k ({k}) must be less than N ({N}) when exclude_self=True, "
            f"otherwise no valid neighbors remain."
        )

    # ||p_i - p_j||² = ||p_i||² + ||p_j||² - 2·p_iᵀ·p_j
    xx = torch.sum(points_xyz ** 2, dim=2, keepdim=True)          # (B, N, 1)
    dist = xx + xx.transpose(2, 1) \
           - 2.0 * torch.bmm(points_xyz, points_xyz.transpose(2, 1))  # (B, N, N)

    if exclude_self:
        # 将对角线（自身距离）置为无穷大，确保不被选为近邻
        diag_mask = torch.eye(
            N, device=points_xyz.device, dtype=torch.bool
        ).unsqueeze(0).expand(B, -1, -1)
        dist.masked_fill_(diag_mask, float("inf"))

    # topk 取最小 k 个距离
    _, knn_idx = torch.topk(dist, k, dim=2, largest=False)  # (B, N, k)
    return knn_idx


# ══════════════════════════════════════════════════════════════════════
# 边特征提取层 (Neighborhood Graph Feature)
# ══════════════════════════════════════════════════════════════════════

class EdgeFeature(nn.Module):
    """局部邻域图边特征提取层，等价于 EdgeConv 操作。

    对每个中心点 i，取 K 个近邻 j，构建边特征 e_{ij} = [f_i, f_j - f_i]，
    通过共享 MLP 变换后做最大池化聚合。差分项 f_j - f_i 捕获局部几何变化，
    拼接项 f_i 保留全局语义锚点。

    在 SPAD 场景下，k 不宜过小（避免被噪声点主导）也不宜过大
    （避免跨目标混合），默认 k=20 经验表现稳定。
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        """初始化边特征提取层。

        Args:
            in_channels: 输入特征维度 C_in。
            out_channels: 输出特征维度 C_out。
            k: 近邻数量。
        """
        super().__init__()
        self.k = k

        # MLP 作用于拼接后的边特征 [f_i, f_j - f_i]（2·C_in → C_out）
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, knn_idx: torch.Tensor
    ) -> torch.Tensor:
        """前向传播：提取并聚合边特征。

        Args:
            x: 点特征，形状 (B, N, C_in)。
            knn_idx: KNN 索引，形状 (B, N, k)。

        Returns:
            out: 聚合后的局部特征，形状 (B, N, C_out)。
        """
        B, N, C = x.shape
        k = knn_idx.shape[-1]

        # ── 使用高级索引 gather 邻居特征 ──
        # 构造批量索引: (B, N, k)
        batch_idx = (
            torch.arange(B, device=x.device)
            .view(B, 1, 1)
            .expand(-1, N, k)
        )
        x_neighbors = x[batch_idx, knn_idx, :]  # (B, N, k, C)

        # 中心点特征广播: (B, N, k, C)
        x_center = x.unsqueeze(2).expand(-1, -1, k, -1)

        # 边特征: 拼接 [f_i, f_j - f_i]
        # 动机: f_i 提供绝对语义, f_j - f_i 编码局部相对几何变化
        edge_feat = torch.cat(
            [x_center, x_neighbors - x_center], dim=-1
        )  # (B, N, k, 2·C)

        # MLP 变换
        edge_feat = self.mlp(edge_feat)  # (B, N, k, C_out)

        # 最大池化聚合 —— 提取每个点邻域内最显著的特征响应
        out = edge_feat.max(dim=2)[0]  # (B, N, C_out)
        return out


# ══════════════════════════════════════════════════════════════════════
# 图残差模块 (Graph Residual Module)
# ══════════════════════════════════════════════════════════════════════

class GraphResidualBlock(nn.Module):
    r"""图残差模块 —— 网络的核心构建块。

    该模块的设计针对性解决了 SPAD 点云的两大痛点:
    1. **噪声导致的几何形变**: 通过将原始坐标显式注入深层特征
       （坐标残差连接），防止网络在特征抽象中"忘记"目标位置。
    2. **特征通道间的信息不平衡**: 通过自注意力门控 (Flow_B)
       自适应地调制图特征 (Flow_A)，增强有效通道、抑制噪声通道。

    数据流 (与 model/readme.md 图一致)::

            F ──→ [LayerNorm] ──┬──→ [EdgeFeature] ──→ [ReLU] ──┐
                                │                                  │
                                └──→ [Linear → ReLU → Sigmoid] ───┤
                                                                   │
                                          [逐元素点乘 Attention] ←─┘
                                                   │
                                              [Linear]
                                                   │
                                                   ├──→ (+) ←── P_proj (坐标投影)
                                                   │
                                                [ReLU]
                                                   │
                                                Output

    Args:
        in_channels: 输入特征维度 C_in。
        out_channels: 输出特征维度 C_out。
        k: KNN 近邻数。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 20,
    ) -> None:
        super().__init__()
        self.k = k

        # ── 坐标投影层: 将 3D 坐标映射到特征空间，供残差连接使用 ──
        # 使用两层 MLP 以允许非线性空间映射（如坐标拉伸/旋转的不变性编码）
        self.point_proj = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        # ── 层归一化: 稳定深层训练，减少内部协变量偏移 ──
        self.ln = nn.LayerNorm(in_channels)

        # ── Flow_A: 图特征提取 (NGF → GCN → ReLU) ──
        self.edge_feat = EdgeFeature(in_channels, out_channels, k)

        # ── Flow_B: 自注意力权重生成 ──
        self.attn_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )

        # ── 融合后特征精炼 ──
        self.fuse_linear = nn.Linear(out_channels, out_channels)

        # ── 输出激活 ──
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        feat: torch.Tensor,
        xyz: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """图残差模块前向传播。

        Args:
            feat: 输入特征流，形状 (B, N, C_in)。
            xyz: 原始三维坐标，形状 (B, N, 3)。
                用于坐标残差投影，所有块使用同一份原始坐标。
            knn_idx: 预计算的 KNN 索引，形状 (B, N, k)。
                基于原始坐标构建，块间复用。

        Returns:
            out: 精炼后的特征，形状 (B, N, C_out)。
        """
        # ── 1. 坐标投影：为残差连接做准备 ──
        # 将 3D 空间坐标非线性映射到 C_out 维特征空间
        P_proj = self.point_proj(xyz)  # (B, N, C_out)

        # ── 2. 层归一化 ──
        F_norm = self.ln(feat)  # (B, N, C_in)

        # ── 3. Flow_A: 图特征提取 ──
        # NGF 提取局部邻域边特征 → GCN(MLP) 变换 → ReLU
        A = self.edge_feat(F_norm, knn_idx)  # (B, N, C_out)
        A = torch.relu(A)

        # ── 4. Flow_B: 自注意力权重 ──
        # 基于归一化特征生成逐通道注意力权重，通过 sigmoid 压缩到 (0,1)
        B = self.attn_proj(F_norm)  # (B, N, C_out)
        B = torch.sigmoid(B)

        # ── 5. 注意力融合 ──
        # 逐元素点乘：注意力权重 B 对图特征 A 进行自适应门控
        # 高权重通道被放大（有效信号），低权重通道被抑制（噪声）
        fused = A * B  # (B, N, C_out)

        # ── 6. 特征映射 ──
        fused = self.fuse_linear(fused)  # (B, N, C_out)

        # ── 7. 坐标残差连接 ──
        # 核心创新：将精炼特征与原始坐标投影相加
        # 确保每一层输出都"锚定"在真实的三维几何位置上
        out = self.act(fused + P_proj)  # (B, N, C_out)

        return out


# ══════════════════════════════════════════════════════════════════════
# 外层模型: 图残差多任务网络
# ══════════════════════════════════════════════════════════════════════

class GraphResidualMultiTaskNet(nn.Module):
    """基于图残差网络的多任务 SPAD 点云目标检测模型。

    该模型同时执行两项任务:
    1. **单类别分类**: 预测输入点云所属的目标类别 (logits)。
    2. **3D 边界框回归**: 预测目标的轴对齐包围盒
       [xmin, xmax, ymin, ymax, zmin, zmax]。

    通道升维策略: 64 → 128 → 256 → 512 → 1024
    全局池化: MaxPool + AvgPool 拼接 → 2048 维
    双预测头: 各自独立的 3 层 MLP

    Args:
        num_classes: 分类类别数，默认 26 (A-Z)。
        k: KNN 近邻数量。
        dropout: 预测头中的 dropout 比例。
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 20,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.k = k

        # ── 初始特征嵌入: (B, N, 4) → (B, N, 64) ──
        # 将 (x, y, z, intensity) 映射到初始特征空间
        self.stem = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # ── 图残差模块堆叠: 逐级升维提取多尺度几何特征 ──
        self.block1 = GraphResidualBlock(64, 128, k)
        self.block2 = GraphResidualBlock(128, 256, k)
        self.block3 = GraphResidualBlock(256, 256, k)
        self.block4 = GraphResidualBlock(256, 512, k)

        # ── 全局池化后维度: 1024 + 1024 = 2048 ──
        pooled_dim = 2048

        # ── 分类预测头 ──
        # 三层 MLP，逐步压缩: 2048 → 512 → 256 → num_classes
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

        # ── 3D Box 回归预测头 ──
        # 与分类头结构对称，最后输出 6 个坐标值
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

        兼容 utils/loss.py 中 split_cls_and_box_predictions 的字典解析:
        - logits 通过 key 'logits' 提取
        - box 预测通过 key 'box_pred' 提取

        Args:
            points: 输入点云张量，形状 (B, N, 4)。
                    通道依次为: x, y, z, intensity。

        Returns:
            outputs: 字典，包含:
                - 'logits': 分类 logits，(B, num_classes)。
                - 'box_pred': 3D 框预测，(B, 6)，格式
                  [xmin, xmax, ymin, ymax, zmin, zmax]。
        """
        B, N, _ = points.shape

        # ── 分离坐标与强度 ──
        xyz = points[:, :, :3]  # (B, N, 3) — 用于 KNN 和坐标残差

        # ── 初始特征嵌入 ──
        F = self.stem(points)  # (B, N, 64)

        # ── 预计算 KNN 图 ──
        # 所有图残差块复用同一组空间近邻（基于原始 xyz 坐标构建）
        # 因为坐标残差设计保证了空间结构在各层间保持一致
        knn_idx = get_knn_indices(xyz, self.k)  # (B, N, k)

        # ── 堆叠图残差模块: 逐级抽象 ──
        F = self.block1(F, xyz, knn_idx)  # (B, N, 128)
        F = self.block2(F, xyz, knn_idx)  # (B, N, 256)
        F = self.block3(F, xyz, knn_idx)  # (B, N, 512)
        F = self.block4(F, xyz, knn_idx)  # (B, N, 1024)

        # ── 全局池化 ──
        # 最大池化捕获最显著特征，平均池化保留全局分布信息
        # 两者拼接提供互补的全局描述
        F_max = F.max(dim=1)[0]   # (B, 1024)
        F_avg = F.mean(dim=1)     # (B, 1024)
        F_pooled = torch.cat([F_max, F_avg], dim=1)  # (B, 2048)

        # ── 双头预测 ──
        logits = self.cls_head(F_pooled)     # (B, num_classes)
        box_preds = self.box_head(F_pooled)  # (B, 6)

        return {
            "logits": logits,
            "box_pred": box_preds,
        }


# ══════════════════════════════════════════════════════════════════════
# 模块入口: 快速测试
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== GraphResidualMultiTaskNet 快速验证 ===\n")

    # 模拟 SPAD 点云输入
    B, N = 2, 1024
    dummy_points = torch.randn(B, N, 4)

    # 构建模型
    model = GraphResidualMultiTaskNet(num_classes=26, k=20)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f} M")

    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_points)

    print(f"输入形状:  {dummy_points.shape}")
    print(f"logits:    {outputs['logits'].shape}")
    print(f"box_pred:  {outputs['box_pred'].shape}")

    # 验证输出范围
    logits = outputs["logits"]
    box = outputs["box_pred"]
    assert logits.shape == (B, 26), f"logits shape mismatch: {logits.shape}"
    assert box.shape == (B, 6), f"box_pred shape mismatch: {box.shape}"
    print(f"\nlogits 范围:  [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"box_pred 范围: [{box.min().item():.3f}, {box.max().item():.3f}]")

    # 兼容性验证: 模拟 split_cls_and_box_predictions 解析
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    
    from utils.loss import split_cls_and_box_predictions
    parsed_logits, parsed_boxes = split_cls_and_box_predictions(outputs)
    assert parsed_logits is not None, "logits 解析失败"
    assert parsed_boxes is not None, "box_pred 解析失败"
    assert torch.equal(parsed_logits, logits), "logits 值不一致"
    assert torch.equal(parsed_boxes, box), "box_pred 值不一致"

    print("\n全部验证通过! split_cls_and_box_predictions 兼容 OK")
    print(f"parsed logits:  {parsed_logits.shape}")
    print(f"parsed boxes:   {parsed_boxes.shape}")

    print("\n=== 显存占用测试 (B=32, N=1024, C=4) ===")
    if torch.cuda.is_available():
        test_model = GraphResidualMultiTaskNet(num_classes=26, k=20).cuda()
        test_points = torch.randn(32, 1024, 4).cuda()
        
        # 预热并清空缓存
        _ = test_model(test_points)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 模拟训练时的前向和反向传播
        train_out = test_model(test_points)
        loss = train_out["logits"].sum() + train_out["box_pred"].sum()
        loss.backward()
        
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"训练阶段峰值显存占用 (Forward + Backward): {peak_mem_mb:.2f} MB")
        
        # 清理
        del test_model, test_points, train_out, loss
        torch.cuda.empty_cache()
    else:
        print("未检测到 GPU (CUDA)，跳过实际峰值显存测试。")

