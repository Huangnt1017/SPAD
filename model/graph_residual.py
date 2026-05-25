"""
单光子点云图残差多任务网络 (Graph Residual Multi-Task Network for SPAD)  v5

v5 (标准 Q/K/V 图注意力 + 坐标门控残差)
==========================================================
Block 数据流:

    Input (P[4D], Feature)
       │           \      /
       │            [ LN ]
       │            /    \
       │        [NGF]   [Linear]
       │          |        |
       │        [GCN]      |
       │          ↓        ↓
       │      Fk(B,N,k,C) F(B,N,C)       ← Flow_A / Flow_B
       │          |        |
       │        [W_k]    [W_q]
       │        [W_v]      |
       │          ↓        ↓
       │         K,V       Q              ← 标准 Q/K/V
       │           \      /
       │     [Scaled Dot-Product Attn]    ← softmax(Q·K / √C) @ V
       │              │
       │           [Linear]
       │              │
       └──> [Coord Gate] ( ⊗ / ⊕ ) ───> Output
                P(4D) → σ(W_gate)         ← gate·attn + (1-gate)·W_res(P)

核心设计:
    1. Flow_A: 保留 per-neighbor 边特征 Fk (不做 max-pool),
       边特征包含完整 4D 坐标差 (含 intensity)。
    2. Flow_B: 中心点特征投影 F (全通道, 含 intensity 信息)。
    3. 标准自注意力: Q=W_q(F), K=W_k(Fk), V=W_v(Fk),
       score=Q·K/√C, softmax over k neighbors。
    4. 坐标门控跳跃: P(4D) 生成 sigmoid gate,
       out = gate * attn_mapped + (1-gate) * coord_residual。
       门控含义: gate→1 信任语义特征; gate→0 信任几何坐标。
    5. 每层: 通道翻倍 + 点数减半 (1024→512→256→128→64)。

通道阶梯: 4 → 32 → 64 → 128 → 256 → 512
点数阶梯: 1024 → 512 → 256 → 128 → 64
全局池化: max + avg → 1024 维
双头: cls + box(3, center-only)

References:
    - model/readme.md 任务 1
    - utils/loss.py split_cls_and_box_predictions
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _HAS_CKPT = True
except (ImportError, AttributeError):
    _HAS_CKPT = False

    def _ckpt(fn, *args, **kwargs):
        return fn(*args)


# ══════════════════════════════════════════════════
# Intensity 加权 4D KNN (双向门控)
# ══════════════════════════════════════════════════

def knn_4d(points: torch.Tensor, k: int, alpha: float = 0.3) -> torch.Tensor:
    """intensity 加权的几何近邻搜索 (双向门控)。

    距离公式: dist = dist_xyz * (1 - alpha + 2 * alpha * dist_i_norm)
        dist_i_norm = 0 (强度相同) → 乘数 = 1 - alpha → 拉近
        dist_i_norm = 1 (差异最大) → 乘数 = 1 + alpha → 推远

    Args:
        points: (B, N, 4) — x, y, z, intensity。
        k: 近邻数 (不含自身)。
        alpha: 门控系数, 建议 ∈ [0.1, 0.5]。

    Returns:
        knn_idx: (B, N, k), int64。
    """
    B, N, _ = points.shape
    xyz = points[..., :3]
    intensity = points[..., 3:4]

    xx = torch.sum(xyz ** 2, dim=2, keepdim=True)
    dist_xyz = xx + xx.transpose(2, 1) - 2.0 * torch.bmm(xyz, xyz.transpose(2, 1))
    dist_xyz = dist_xyz.clamp(min=0.0)

    ii = torch.sum(intensity ** 2, dim=2, keepdim=True)
    dist_i = ii + ii.transpose(2, 1) - 2.0 * torch.bmm(intensity, intensity.transpose(2, 1))
    dist_i = dist_i.clamp(min=0.0)

    dist_i_max = dist_i.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    dist_i_norm = dist_i / dist_i_max

    dist = dist_xyz * (1.0 - alpha + 2.0 * alpha * dist_i_norm)

    diag = torch.eye(N, device=points.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    dist.masked_fill_(diag, float("inf"))

    _, knn_idx = torch.topk(dist, k, dim=2, largest=False)
    return knn_idx


# ══════════════════════════════════════════════════
# 加权随机下采样
# ══════════════════════════════════════════════════

def weighted_downsample(
    xyz: torch.Tensor,
    feats: torch.Tensor,
    target_n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按特征 L2 范数作为重要性的无放回随机采样。

    Args:
        xyz: (B, N, 4) 点坐标 + intensity。
        feats: (B, N, C) 每点特征。
        target_n: 目标采样点数。

    Returns:
        xyz_down: (B, target_n, 4)
        feats_down: (B, target_n, C)
    """
    B, N, _ = feats.shape
    if target_n >= N:
        return xyz, feats

    scores = feats.norm(p=2, dim=-1).clamp(min=1e-8)
    probs = scores / scores.sum(dim=1, keepdim=True)
    idx = torch.multinomial(probs, target_n, replacement=False)
    batch_idx = torch.arange(B, device=feats.device).view(B, 1).expand(-1, target_n)
    return xyz[batch_idx, idx, :], feats[batch_idx, idx, :]


# ══════════════════════════════════════════════════
# Graph Residual Block v5 (Q/K/V 图注意力 + 坐标门控)
# ══════════════════════════════════════════════════

class GraphResidualBlock(nn.Module):
    """图残差模块 v5 (标准 Q/K/V 注意力 + 坐标门控残差)。

    数据流:
        Input (P[4D], f)
            ↓
          LN(f) → f_norm
          /          \\
       Flow_A       Flow_B
     (NGF+GCN)     (Linear)
         ↓            ↓
       Fk(B,N,k,C)  F(B,N,C)
         |            |
       W_k, W_v     W_q                   ← Q/K/V 投影
         ↓            ↓
        K, V          Q
         \\          //
      softmax(Q·K/√C) @ V                 ← Scaled Dot-Product Attention
             ↓
          [Linear]  (特征映射)
             ↓
      gate·(mapped) + (1-gate)·W_res(P)   ← 坐标门控跳跃连接
             ↓
           ReLU → [downsample]

    坐标门控: gate = σ(W_gate(P[4D]))
        - gate → 1: 信任注意力聚合的语义特征
        - gate → 0: 信任原始坐标 + 强度信息
        物理含义: 在噪声区域模型可学习更多依赖语义;
                  在几何清晰区域保留原始位置约束。

    Args:
        in_channels: 输入特征维度 C_in。
        out_channels: 输出特征维度 C_out。
        k: KNN 近邻数。
        downsample: 是否 N → N/2。
        use_checkpoint: 训练时梯度检查点。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 12,
        downsample: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.k = k
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint
        self._scale = out_channels ** -0.5

        # 入口归一化
        self.ln = nn.LayerNorm(in_channels)

        # Flow_A: 边特征 MLP — 输入含完整 4D 坐标差 (xyzi)
        # [f_i, f_j - f_i, p_j - p_i(4D)] → (2*C_in + 4)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 4, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

        # Flow_B: 中心点特征投影
        self.flow_b = nn.Linear(in_channels, out_channels)

        # Q/K/V 投影 (标准注意力)
        self.W_q = nn.Linear(out_channels, out_channels)
        self.W_k = nn.Linear(out_channels, out_channels)
        self.W_v = nn.Linear(out_channels, out_channels)

        # 注意力输出映射
        self.linear_out = nn.Linear(out_channels, out_channels)

        # 坐标门控 + 坐标残差 (全程使用 4D: x, y, z, intensity)
        self.coord_gate = nn.Linear(4, out_channels)
        self.coord_res = nn.Linear(4, out_channels)

        self.act = nn.ReLU(inplace=True)

    def _forward_impl(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """核心前向。"""
        B, N, _ = f.shape
        k = knn_idx.shape[-1]

        # ── 入口归一化 ──
        f_norm = self.ln(f)                                      # (B, N, C_in)

        # ── Flow_A: 构图 → per-neighbor 边特征 Fk ──
        batch_idx = torch.arange(B, device=f.device).view(B, 1, 1).expand(B, N, k)
        # 收集邻居特征和 4D 坐标
        f_nbr = f_norm[batch_idx, knn_idx]                       # (B, N, k, C_in)
        p_nbr = p[batch_idx, knn_idx]                            # (B, N, k, 4) 含 intensity

        # 边特征: [f_i, f_j - f_i, p_j - p_i (4D含intensity)]
        f_center = f_norm.unsqueeze(2).expand_as(f_nbr)          # (B, N, k, C_in)
        p_center = p.unsqueeze(2).expand_as(p_nbr)               # (B, N, k, 4)
        edge_feat = torch.cat(
            [f_center, f_nbr - f_center, p_nbr - p_center],
            dim=-1,
        )                                                         # (B, N, k, 2*C_in+4)
        Fk = self.edge_mlp(edge_feat)                             # (B, N, k, C_out)

        # ── Flow_B: 中心点特征 ──
        F = self.flow_b(f_norm)                                   # (B, N, C_out)

        # ── 标准 Q/K/V 注意力 ──
        Q = self.W_q(F)                                           # (B, N, C_out)
        K = self.W_k(Fk)                                          # (B, N, k, C_out)
        V = self.W_v(Fk)                                          # (B, N, k, C_out)

        # Scaled dot-product: Q_i · K_ij / √C → softmax over k
        score = (Q.unsqueeze(2) * K).sum(dim=-1) * self._scale    # (B, N, k)
        weights = torch.softmax(score, dim=-1)                    # (B, N, k)

        # 加权聚合 Value
        attn_out = (weights.unsqueeze(-1) * V).sum(dim=2)         # (B, N, C_out)

        # ── 注意力输出映射 ──
        mapped = self.linear_out(attn_out)                        # (B, N, C_out)

        # ── 坐标门控跳跃连接 (P 为完整 4D: x, y, z, intensity) ──
        # gate ∈ (0,1): 控制语义特征 vs 坐标信息的混合比
        gate = torch.sigmoid(self.coord_gate(p))                  # (B, N, C_out)
        coord_info = self.coord_res(p)                            # (B, N, C_out)
        # gate·语义 + (1-gate)·坐标
        out = gate * mapped + (1.0 - gate) * coord_info           # (B, N, C_out)
        f_out = self.act(out)

        # ── 层间下采样 (N → N/2) ──
        if self.downsample:
            target_n = N // 2
            p_down, f_out = weighted_downsample(p, f_out, target_n)
            knn_new = knn_4d(p_down, min(self.k, target_n - 1))
            return f_out, p_down, knn_new
        return f_out, p, knn_idx

    def forward(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对外接口 (训练启用梯度检查点)。"""
        if self.use_checkpoint and self.training and _HAS_CKPT:
            return _ckpt(self._forward_impl, p, f, knn_idx, use_reentrant=False)
        return self._forward_impl(p, f, knn_idx)


# ══════════════════════════════════════════════════
# 外层多任务网络
# ══════════════════════════════════════════════════

class GraphResidualMultiTaskNet(nn.Module):
    """图残差多任务网络 v5。

    通道阶梯: 4 → 32 → 64 → 128 → 256 → 512
    点数阶梯: 1024 → 512 → 256 → 128 → 64
    全局池化: max + avg → 1024 维
    双头: cls (26) + box (3, center-only)

    Args:
        num_classes: 分类数。
        k: KNN 近邻数。
        use_checkpoint: 梯度检查点。
        dropout: 预测头 Dropout。
        box_dim: bbox 维度 (默认 3)。
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 12,
        use_checkpoint: bool = True,
        dropout: float = 0.3,
        box_dim: int = 3,
    ):
        super().__init__()
        self.k = k
        self.box_dim = box_dim

        # Stem: (B, N, 4) → (B, N, 32)
        self.stem = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )

        # 4 层 Graph Residual Block (通道翻倍 + 点数减半)
        block_cfg = dict(k=k, downsample=True, use_checkpoint=use_checkpoint)
        self.block1 = GraphResidualBlock(32, 64, **block_cfg)
        self.block2 = GraphResidualBlock(64, 128, **block_cfg)
        self.block3 = GraphResidualBlock(128, 256, **block_cfg)
        self.block4 = GraphResidualBlock(256, 512, **block_cfg)

        pooled_dim = 1024  # 512 * 2 (max + avg)

        self.cls_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.box_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, box_dim),
        )

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 4) — x, y, z, intensity。

        Returns:
            dict: 'logits' (B, num_classes), 'box_pred' (B, box_dim)。
        """
        B, N, _ = points.shape

        p = points.clone()                                        # (B, N, 4)
        f = self.stem(points)                                     # (B, N, 32)
        knn_idx = knn_4d(p, min(self.k, N - 1))                   # (B, N, k)

        f, p, knn_idx = self.block1(p, f, knn_idx)                # 1024→512,  32→64
        f, p, knn_idx = self.block2(p, f, knn_idx)                # 512→256,   64→128
        f, p, knn_idx = self.block3(p, f, knn_idx)                # 256→128,  128→256
        f, p, knn_idx = self.block4(p, f, knn_idx)                # 128→64,   256→512

        # 全局池化: max + avg → (B, 1024)
        f_max = f.max(dim=1)[0]
        f_avg = f.mean(dim=1)
        f_pooled = torch.cat([f_max, f_avg], dim=1)

        logits = self.cls_head(f_pooled)
        box_preds = self.box_head(f_pooled)

        return {"logits": logits, "box_pred": box_preds}


# ══════════════════════════════════════════════════
# 验证 + GPU 显存测试
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    print("=== GraphResidualMultiTaskNet v5 (Q/K/V 图注意力 + 坐标门控) ===\n")

    B, N = 2, 1024
    dummy = torch.randn(B, N, 4)

    model = GraphResidualMultiTaskNet(num_classes=26, k=12, use_checkpoint=False, box_dim=3)
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

    for i, block in enumerate([model.block1, model.block2, model.block3, model.block4], 1):
        c_in = block.flow_b.in_features
        c_out = block.linear_out.out_features
        ds = "Y" if block.downsample else "N"
        print(f"  block{i}: C_in={c_in:3d} C_out={c_out:3d} downsample={ds}")

    # 与 utils.loss 接口兼容
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.loss import split_cls_and_box_predictions
    logits_o, box_o = split_cls_and_box_predictions(out)
    assert logits_o is not None and box_o is not None
    print(f"\n  split_cls_and_box_predictions: logits {logits_o.shape}, box {box_o.shape}")
    print("\nAll checks pass.")

    # GPU 显存测试
    print("\n=== GPU 显存测试 ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        try:
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, "total_memory"):
                total_gb = props.total_memory / 1024 ** 3
            elif hasattr(props, "total_mem"):
                total_gb = props.total_mem / 1024 ** 3
            else:
                total_gb = 0.0
            if total_gb > 0:
                print(f"总显存: {total_gb:.1f} GB")
        except Exception:
            pass
        print()

        import gc
        for bs in [4, 8, 16, 32]:
            try:
                m = GraphResidualMultiTaskNet(num_classes=26, k=12, use_checkpoint=True, box_dim=3).cuda()
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
