"""
单光子点云图残差多任务网络 (Graph Residual Multi-Task Network for SPAD)  v7

v7 (v6 速度优化)
==========================================================
v6 → v7 关键改进 (架构不变, 仅优化计算效率):
    1. 坐标 KNN + graph_feature 预计算: p 全程不变, 在 Net.forward 入口
       一次性计算 knn_p 和 p_graph(B,8,N,k), 4 个 Block 共享复用,
       省掉 3 次 O(N²) KNN + 3 次 gather/permute (约提速 30%)。
    2. 梯度检查点可配置 (use_checkpoint, 默认 True): 开启时反向重算前向,
       省显存适配 12GB 卡; 显存充裕时传 use_checkpoint=False 关闭以提速约 15%。

v6 基础架构 (保留):
    - Conv2d+BN2d EdgeConv, 动态特征空间 KNN, DGCNN 风格 GPU KNN
    - 双流 GCN → Q/K/V 注意力 → 坐标门控残差
    - 全程 (B, C, N) 布局, 无下采样 (N=1024 贯穿 4 层)

Block 数据流 (无下采样, 全程 N=1024):
    f(B,C,N) + p(B,4,N) + p_graph(B,8,N,k) [预计算缓存]
        ↓
    Dynamic KNN from f → idx(B,N,k)    ← 仅特征 KNN 每层重算
        ↓
    ┌─ GCN_f: get_graph_feature(f) → Conv2d+BN2d → Fk(B,C_out,N,k)  ← V source
    └─ GCN_p: p_graph → Conv2d+BN2d → Pk(B,C_out,N,k)               ← K source (复用)
        ↓
    Q = Conv1d(f‖p), K = Conv2d(Pk), V = Conv2d(Fk)
    attn = softmax(Q·K/√C) @ V
        ↓
    gate·Conv1d(attn) + (1-gate)·Conv1d(p)    ← 坐标门控
        ↓
    LeakyReLU → f_out(B,C_out,N), p 不变

References:
    - model/readme.md 任务 1
    - baseline/DGCNN.py (knn, get_graph_feature, Conv2d+BN2d)
    - utils/loss.py split_cls_and_box_predictions
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from utils.heads import build_standard_cls_head, build_standard_box_head
# knn_gpu / get_graph_feature / weighted_downsample 集中在 utils.graph_ops,
# 与 graph_res_GCN.py 共享同一份实现 (避免拷贝漂移)。
from utils.graph_ops import get_graph_feature, knn_gpu, weighted_downsample

try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _HAS_CKPT = True
except (ImportError, AttributeError):
    _HAS_CKPT = False

    def _ckpt(fn, *args, **kwargs):
        return fn(*args)


# ══════════════════════════════════════════════════
# Graph Residual Block v6
# ══════════════════════════════════════════════════

class GraphResidualBlock(nn.Module):
    """图残差模块 v6 (Conv2d+BN2d EdgeConv + Q/K/V 注意力 + 坐标门控)。

    数据流 (全程 (B, C, N) 布局, 无下采样, N 不变):
        f(B,C_in,N) + p(B,4,N) + p_graph(B,8,N,k) [外部预计算缓存]
            ↓
        Dynamic KNN from f → idx(B,N,k)    ← 仅特征 KNN 每层重算
            ↓
        ┌ GCN_f: get_graph_feature(f) → Conv2d+BN2d+LReLU → Fk(B,C_out,N,k)
        └ GCN_p: p_graph → Conv2d+BN2d+LReLU → Pk(B,C_out,N,k)  ← 复用预计算
            ↓
        Q = Conv1d+BN1d(f‖p)     (B, C_out, N)
        K = Conv2d(Pk)            (B, C_out, N, k)
        V = Conv2d(Fk)            (B, C_out, N, k)
        attn = softmax(Q·K/√C) @ V  → (B, C_out, N)
            ↓
        mapped = Conv1d+BN1d(attn)
        gate = σ(Conv1d+BN1d(p))
        out = gate * mapped + (1-gate) * Conv1d+BN1d(p)
            ↓
        LeakyReLU → f_out(B,C_out,N), p 原样传出

    优化说明:
        p 全程不变 → knn_gpu(p) 和 get_graph_feature(p) 在 Net 层预计算一次,
        4 个 Block 共享同一份 p_graph (省掉 3 次 KNN + 3 次 graph_feature 构建)。

    Args:
        in_channels: C_in。
        out_channels: C_out。
        k: 近邻数。
        downsample: 是否启用 N→N/2 下采样 (当前配置为 False, 全程保留所有点)。
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
        self._scale = out_channels ** -0.5

        # GCN_f: 特征 EdgeConv — [f_j - f_i, f_i] → Conv2d+BN2d
        self.conv_f = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

        # GCN_p: 位置 EdgeConv — [p_j - p_i, p_i] (4D) → Conv2d+BN2d
        self.conv_p = nn.Sequential(
            nn.Conv2d(8, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

        # Q: 中心点 (f ‖ p) → Conv1d+BN1d
        self.W_q = nn.Sequential(
            nn.Conv1d(in_channels + 4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        # K: 从位置图卷积 Pk
        self.W_k = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        # V: 从特征图卷积 Fk
        self.W_v = nn.Conv2d(out_channels, out_channels, 1, bias=False)

        # 注意力输出映射
        self.out_conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
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

        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        p_graph: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block 前向。

        Args:
            p: (B, 4, N) 原始坐标+intensity。
            f: (B, C_in, N) 当前层特征。
            p_graph: (B, 8, N, k) 坐标图特征 (Net 层预计算, 4 个 Block 共享)。

        Returns:
            (p, f_out): p 原样���出 (当前配置不下采样), f_out 为升维后特征 (B, C_out, N)。
        """
        B, C_in, N = f.shape
        k = min(self.k, N - 1)

        # ── 特征空间 KNN (每层重算, 语义驱动动态图) ──
        knn_f = knn_gpu(f, k)                                   # (B, N, k)

        # ── GCN_f: 特征 EdgeConv → Fk (V 来源) ──
        f_graph = get_graph_feature(f, k, knn_f)               # (B, 2*C_in, N, k)
        Fk = self.conv_f(f_graph)                               # (B, C_out, N, k)

        # ── GCN_p: 位置 EdgeConv → Pk (K 来源, 复用预计算的坐标图特征) ──
        Pk = self.conv_p(p_graph)                               # (B, C_out, N, k)

        # ── Q/K/V 投影 ──
        # Q: 中心点联合查询 (f ‖ p)
        Q = self.W_q(torch.cat([f, p], dim=1))                 # (B, C_out, N)
        # K: 位置图卷积 (几何驱动注意力权重)
        K = self.W_k(Pk)                                        # (B, C_out, N, k)
        # V: 特征图卷积 (语义内容被聚合)
        V = self.W_v(Fk)                                        # (B, C_out, N, k)

        # ── Scaled Dot-Product Attention ──
        # 转到 (B, N, ...) 做 softmax, 再转回
        Q_t = Q.permute(0, 2, 1)                                # (B, N, C_out)
        K_t = K.permute(0, 2, 3, 1)                             # (B, N, k, C_out)
        V_t = V.permute(0, 2, 3, 1)                             # (B, N, k, C_out)

        score = (Q_t.unsqueeze(2) * K_t).sum(dim=-1) * self._scale  # (B, N, k)
        weights = torch.softmax(score, dim=-1)                       # (B, N, k)
        attn = (weights.unsqueeze(-1) * V_t).sum(dim=2)             # (B, N, C_out)
        attn = attn.permute(0, 2, 1).contiguous()                   # (B, C_out, N)

        # ── 输出映射 + 坐标门控 ──
        mapped = self.out_conv(attn)                             # (B, C_out, N)
        gate = torch.sigmoid(self.coord_gate(p))                 # (B, C_out, N)
        coord_info = self.coord_res(p)                           # (B, C_out, N)
        out = gate * mapped + (1.0 - gate) * coord_info
        f_out = self.act(out)

        # ── 层间下采样 ──
        if self.downsample:
            p, f_out = weighted_downsample(p, f_out, N // 2)
        return p, f_out


# ══════════════════════════════════════════════════
# 外层多任务网络
# ══════════════════════════════════════════════════

class GraphResidualMultiTaskNet(nn.Module):
    """图残差多任务网络 v7。

    v6 → v7 速度优化:
        1. 坐标 KNN + graph_feature 预计算: p 全程不变, 在 forward 入口一次性计算
           knn_p 和 p_graph, 4 个 Block 共享 (省掉 3 次 KNN + 3 次 graph_feature)。
        2. 梯度检查点可选: 默认开启 (B=32 需约 7.4GB); 关闭时反向免重算
           但显存翻倍 (约 13.7GB), 仅 B<=16 或大显存卡适用。

    全程 (B, C, N) 布局, Conv+BN+LeakyReLU。
    通道: 4->32->64->64->128->256->512, 点数: 全程 1024 不下采样。

    Args:
        num_classes: 分类数。
        k: 近邻数 (默认 20, 对齐 DGCNN)。
        use_checkpoint: 梯度检查点 (默认 True, B=32 必须开启以适配 12GB 显存)。
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

        # 无下采样: 全部 1024 点贯穿 4 层, p 不变只升维 f; 通道对齐 DGCNN 避免 OOM
        block_cfg = dict(k=k, downsample=False)
        self.block1 = GraphResidualBlock(32, 64, **block_cfg)
        self.block2 = GraphResidualBlock(64, 64, **block_cfg)
        self.block3 = GraphResidualBlock(64, 128, **block_cfg)
        self.block4 = GraphResidualBlock(128, 256, **block_cfg)

        # 多尺度拼接 (类似 DGCNN): cat(b1, b2, b3, b4) → Conv1d 聚合
        cat_dim = 64 + 64 + 128 + 256  # 512
        self.agg_conv = nn.Sequential(
            nn.Conv1d(cat_dim, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        pooled_dim = 1024  # 512 * 2

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
        f = self.stem(p)                                        # (B, 32, N)

        # 坐标 KNN + graph_feature 预计算 (p 全程不变, 4 个 Block 共享)
        k = min(self.k, p.size(2) - 1)
        knn_p = knn_gpu(p, k)                                   # (B, N, k)
        p_graph = get_graph_feature(p, k, knn_p)               # (B, 8, N, k)

        # 4 层无下采样, 全程 N=1024; 仅特征 KNN 每层重算
        use_ckpt = self.use_checkpoint and self.training and _HAS_CKPT
        def _run_block(block, _p, _f, _pg):
            return block(_p, _f, _pg)

        if use_ckpt:
            p, f1 = _ckpt(_run_block, self.block1, p, f, p_graph, use_reentrant=False)
            p, f2 = _ckpt(_run_block, self.block2, p, f1, p_graph, use_reentrant=False)
            p, f3 = _ckpt(_run_block, self.block3, p, f2, p_graph, use_reentrant=False)
            p, f4 = _ckpt(_run_block, self.block4, p, f3, p_graph, use_reentrant=False)
        else:
            p, f1 = self.block1(p, f, p_graph)                  # 32->64
            p, f2 = self.block2(p, f1, p_graph)                 # 64->64
            p, f3 = self.block3(p, f2, p_graph)                 # 64->128
            p, f4 = self.block4(p, f3, p_graph)                 # 128->256

        # 多尺度拼接 + 聚合 (保留各层分辨率特征, 类似 DGCNN)
        f = self.agg_conv(torch.cat([f1, f2, f3, f4], dim=1))  # (B, 512, N)

        # 全局池化: max + avg → (B, 1024)
        f_max = f.max(dim=-1)[0]
        f_avg = f.mean(dim=-1)
        f_pooled = torch.cat([f_max, f_avg], dim=1)              # (B, 1024)

        logits = self.cls_head(f_pooled)

        # Box head: 直接回归 (与 baseline 一致)
        # 从全局特征直接预测中心点坐标, 不依赖质心先验
        box_preds = self.box_head(f_pooled)                       # (B, 3)

        return {"logits": logits, "box_pred": box_preds}


# ══════════════════════════════════════════════════
# 验证 + GPU 显存测试
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    print("=== GraphResidualMultiTaskNet v7 (v6 + p_graph cache + checkpoint opt) ===\n")

    B, N = 2, 1024
    dummy = torch.randn(B, N, 4)

    model = GraphResidualMultiTaskNet(num_classes=26, k=20, use_checkpoint=True, box_dim=3)
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

    for i, blk in enumerate([model.block1, model.block2, model.block3, model.block4], 1):
        c_out = blk.out_conv[0].out_channels
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
                m = GraphResidualMultiTaskNet(num_classes=26, k=20, use_checkpoint=True, box_dim=3).cuda()
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
