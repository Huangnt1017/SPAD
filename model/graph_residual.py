"""
单光子点云图残差多任务网络 (Graph Residual Multi-Task Network for SPAD)  v6

v6 (DGCNN 工程范式 + 双流图注意力 + 坐标门控)
==========================================================
v5 → v6 关键改进:
    1. Conv2d + BN2d 替代 Linear 做 EdgeConv (收敛速度大幅提升)。
    2. 动态特征空间 KNN (每层从学到的特征重建图, 接近 DGCNN 的 "Dynamic Graph")。
    3. DGCNN 风格的高效 GPU KNN 与图特征构建 (全局 flatten 索引, 一次 gather)。
    4. LeakyReLU(0.2) + BN1d 替代 LayerNorm + ReLU。
    5. 全程 (B, C, N) 布局 (Conv1d/Conv2d 原生, 省去逐点 Linear 的开销)。
    6. 保留: 双流 GCN → Q/K/V 注意力 → 坐标门控残差 (v5 的结构设计)。

Block 数据流:
    f(B,C,N) + p(B,4,N)
        ↓
    Dynamic KNN from f → idx(B,N,k)    ← 特征空间构图 (GPU)
        ↓
    ┌─ GCN_f: get_graph_feature(f) → Conv2d+BN2d → Fk(B,C_out,N,k)  ← V source
    └─ GCN_p: get_graph_feature(p) → Conv2d+BN2d → Pk(B,C_out,N,k)  ← K source
        ↓
    Q = Conv1d(f‖p), K = Conv2d(Pk), V = Conv2d(Fk)
    attn = softmax(Q·K/√C) @ V
        ↓
    gate·Conv1d(attn) + (1-gate)·Conv1d(p)    ← 坐标门控
        ↓
    LeakyReLU → weighted_downsample(N→N/2)

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

try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _HAS_CKPT = True
except (ImportError, AttributeError):
    _HAS_CKPT = False

    def _ckpt(fn, *args, **kwargs):
        return fn(*args)


# ══════════════════════════════════════════════════
# GPU KNN (DGCNN 风格: 特征空间, 负距离 topk)
# ══════════════════════════════════════════════════

def knn_gpu(x: torch.Tensor, k: int) -> torch.Tensor:
    """在特征空间做 KNN (与 DGCNN 一致)。

    用负平方距离 + topk 实现, 全程 GPU matmul, 无需排序。

    Args:
        x: (B, C, N) — 任意维特征 (坐标 / 学到的特征均可)。
        k: 近邻数 (不含自身, 由 get_graph_feature 的拼接隐式排除)。

    Returns:
        idx: (B, N, k), int64。
    """
    # (B, N, N) 负平方距离: 越大越近
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    neg_dist = -xx - inner - xx.transpose(2, 1)
    _, idx = neg_dist.topk(k=k, dim=-1)
    return idx


# ══════════════════════════════════════════════════
# 图特征构建 (DGCNN 风格: 全局 flatten 索引)
# ══════════════════════════════════════════════════

def get_graph_feature(x: torch.Tensor, k: int,
                      idx: torch.Tensor | None = None) -> torch.Tensor:
    """构造 EdgeConv 边特征 [x_j - x_i, x_i] (DGCNN 原版写法)。

    Args:
        x: (B, C, N)。
        k: 近邻数。
        idx: (B, N, k), 若 None 则内部调用 knn_gpu。

    Returns:
        (B, 2C, N, k) — 每点 k 个邻居的边特征。
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn_gpu(x, k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)

    x_t = x.transpose(2, 1).contiguous()                   # (B, N, C)
    nbr = x_t.view(B * N, C)[idx_flat].view(B, N, k, C)    # (B, N, k, C)
    x_i = x_t.unsqueeze(2).expand_as(nbr)                  # (B, N, k, C)

    # [x_j - x_i, x_i] → (B, 2C, N, k)
    return torch.cat([nbr - x_i, x_i], dim=-1).permute(0, 3, 1, 2).contiguous()


# ══════════════════════════════════════════════════
# 加权下采样 (B, C, N) 布局
# ══════════════════════════════════════════════════

def weighted_downsample(
    p: torch.Tensor,
    f: torch.Tensor,
    target_n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按特征 L2 范数做无放回加权采样。

    Args:
        p: (B, 4, N)。
        f: (B, C, N)。
        target_n: 目标点数。

    Returns:
        (p_down, f_down): (B, 4, target_n), (B, C, target_n)。
    """
    B, C, N = f.shape
    if target_n >= N:
        return p, f

    scores = f.norm(p=2, dim=1).clamp(min=1e-8)              # (B, N)
    probs = scores / scores.sum(dim=1, keepdim=True)
    idx = torch.multinomial(probs, target_n, replacement=False)  # (B, target_n)

    # gather 沿 N 维采样
    idx_f = idx.unsqueeze(1).expand(-1, C, -1)                # (B, C, target_n)
    idx_p = idx.unsqueeze(1).expand(-1, 4, -1)                # (B, 4, target_n)
    return torch.gather(p, 2, idx_p), torch.gather(f, 2, idx_f)


# ══════════════════════════════════════════════════
# Graph Residual Block v6
# ══════════════════════════════════════════════════

class GraphResidualBlock(nn.Module):
    """图残差模块 v6 (Conv2d+BN2d EdgeConv + Q/K/V 注意力 + 坐标门控)。

    数据流 (全程 (B, C, N) 布局):
        f(B,C_in,N) + p(B,4,N)
            ↓
        Dynamic KNN from f → idx(B,N,k)
            ↓
        ┌ GCN_f: get_graph_feature(f) → Conv2d+BN2d+LReLU → Fk(B,C_out,N,k)
        └ GCN_p: get_graph_feature(p) → Conv2d+BN2d+LReLU → Pk(B,C_out,N,k)
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
        LeakyReLU → downsample(N→N/2)

    Args:
        in_channels: C_in。
        out_channels: C_out。
        k: 近邻数。
        downsample: N→N/2。
        use_checkpoint: 梯度检查点。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 20,
        downsample: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.k = k
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint
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

    def _forward_impl(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """核心前向。

        Args:
            p: (B, 4, N) 原始坐标+intensity。
            f: (B, C_in, N) 当前层特征。

        Returns:
            (p_out, f_out): 下采样后的坐标与特征。
        """
        B, C_in, N = f.shape
        k = min(self.k, N - 1)

        # ── 双路 KNN (GPU) ──
        # GCN_f: 特征空间 KNN (语义驱动, 类似 DGCNN dynamic graph)
        knn_f = knn_gpu(f, k)                                   # (B, N, k)
        # GCN_p: 坐标+intensity 空间 KNN (保留 SPAD 强度空间结构)
        knn_p = knn_gpu(p, k)                                   # (B, N, k)

        # ── GCN_f: 特征 EdgeConv → Fk (V 来源) ──
        f_graph = get_graph_feature(f, k, knn_f)               # (B, 2*C_in, N, k)
        Fk = self.conv_f(f_graph)                               # (B, C_out, N, k)

        # ── GCN_p: 位置 EdgeConv → Pk (K 来源, 基于 xyzi 空间邻居) ──
        p_graph = get_graph_feature(p, k, knn_p)               # (B, 8, N, k)
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

    def forward(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpoint and self.training and _HAS_CKPT:
            return _ckpt(self._forward_impl, p, f, use_reentrant=False)
        return self._forward_impl(p, f)


# ══════════════════════════════════════════════════
# 外层多任务网络
# ══════════════════════════════════════════════════

class GraphResidualMultiTaskNet(nn.Module):
    """图残差多任务网络 v6。

    全程 (B, C, N) 布局, Conv+BN+LeakyReLU。
    通道: 4→32→64→128→256→512, 点数: 1024→512→256→128→64。

    Args:
        num_classes: 分类数。
        k: 近邻数 (默认 20, 对齐 DGCNN)。
        use_checkpoint: 梯度检查点。
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

        # Stem: (B, 4, N) → (B, 32, N)
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )

        # 不下采样: 全部 1024 点跑完 4 层, 通道压缩到 DGCNN 级别避免 OOM
        block_cfg = dict(k=k, downsample=False, use_checkpoint=use_checkpoint)
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
        box_input_dim = pooled_dim + 8

        self.cls_head = nn.Sequential(
            nn.Linear(pooled_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Box head: 拼接坐标均值+标准差 → 提供空间锚点
        # p_mean 近似目标中心, p_std 反映空间尺度
        self.box_head = nn.Sequential(
            nn.Linear(box_input_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, box_dim),
        )

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

        # 4 层无下采样, 全程 N=1024
        p, f1 = self.block1(p, f)                               # 32→64
        p, f2 = self.block2(p, f1)                              # 64→64
        p, f3 = self.block3(p, f2)                              # 64→128
        p, f4 = self.block4(p, f3)                              # 128→256

        # 多尺度拼接 + 聚合 (保留各层分辨率特征, 类似 DGCNN)
        f = self.agg_conv(torch.cat([f1, f2, f3, f4], dim=1))  # (B, 512, N)

        # 全局池化: max + avg → (B, 1024)
        f_max = f.max(dim=-1)[0]
        f_avg = f.mean(dim=-1)
        f_pooled = torch.cat([f_max, f_avg], dim=1)              # (B, 1024)

        logits = self.cls_head(f_pooled)

        # Box head: 拼接最终点集的坐标统计量作为空间锚点
        # p_mean ≈ 目标中心先验, p_std ≈ 空间尺度先验
        p_mean = p.mean(dim=-1)                                   # (B, 4)
        p_std = p.std(dim=-1)                                     # (B, 4)
        box_input = torch.cat([f_pooled, p_mean, p_std], dim=1)  # (B, 1032)
        box_preds = self.box_head(box_input)

        return {"logits": logits, "box_pred": box_preds}


# ══════════════════════════════════════════════════
# 验证 + GPU 显存测试
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    print("=== GraphResidualMultiTaskNet v6 (DGCNN范式 + 图注意力 + 坐标门控) ===\n")

    B, N = 2, 1024
    dummy = torch.randn(B, N, 4)

    model = GraphResidualMultiTaskNet(num_classes=26, k=20, use_checkpoint=False, box_dim=3)
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
