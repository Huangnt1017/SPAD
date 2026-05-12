"""
单光子点云图残差多任务网络 (Graph Residual Multi-Task Network for SPAD)  v3

v3 更新:
    - QKV 注意力：Q(中心坐标+特征), K(邻居特征+Δ坐标), V(邻居特征+Δ特征)
    - 每层加权随机下采样，点数逐层减半 (1024→512→256→128→64)
    - KNN 使用完整 4D 向量 (x,y,z,intensity)
    - 坐标 p 与特征 f 始终同时流入各 block
    - 轻量通道: 32→64→128→256→512，目标 B=32

References:
    - model/readme.md 任务1
    - utils/loss.py split_cls_and_box_predictions
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容 checkpoint API
try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _HAS_CKPT = True
except (ImportError, AttributeError):
    _HAS_CKPT = False
    def _ckpt(fn, *args, **kwargs):
        return fn(*args)


# ══════════════════════════════════════════════════
# 4D KNN (含强度)
# ══════════════════════════════════════════════════

def knn_4d(points: torch.Tensor, k: int) -> torch.Tensor:
    """计算 4D 近邻索引（x, y, z, intensity）。

    强度按经验缩放因子 0.1 归一化到与坐标近似的量级，
    避免强度维主导或完全失效。

    Args:
        points: (B, N, 4) — x, y, z, intensity。
        k: 近邻数。

    Returns:
        knn_idx: (B, N, k)，int64。
    """
    B, N, _ = points.shape
    # 分离坐标与强度，强度缩放防止量纲差异
    xyz = points[..., :3]
    intensity = points[..., 3:4] * 0.1  # 缩放因子

    xx = torch.sum(xyz ** 2, dim=2, keepdim=True)  # (B, N, 1)
    ii = torch.sum(intensity ** 2, dim=2, keepdim=True)
    dist_xyz = xx + xx.transpose(2, 1) - 2.0 * torch.bmm(xyz, xyz.transpose(2, 1))
    dist_i = ii + ii.transpose(2, 1) - 2.0 * torch.bmm(intensity, intensity.transpose(2, 1))
    dist = dist_xyz + dist_i  # 4D 欧氏距离

    # 排除自身
    diag = torch.eye(N, device=points.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    dist.masked_fill_(diag, float('inf'))

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
    """基于特征 L2 范数的加权随机下采样。

    Args:
        xyz: (B, N, 4) 点坐标+强度。
        feats: (B, N, C) 特征。
        target_n: 目标点数，应 ≤ N。

    Returns:
        xyz_down: (B, target_n, 4)
        feats_down: (B, target_n, C)
    """
    B, N, _ = feats.shape
    if target_n >= N:
        return xyz, feats

    # 特征范数作为重要性分数
    scores = feats.norm(p=2, dim=-1)  # (B, N)
    scores = scores.clamp(min=1e-8)
    probs = scores / scores.sum(dim=1, keepdim=True)

    idx = torch.multinomial(probs, target_n, replacement=False)  # (B, target_n)
    batch_idx = torch.arange(B, device=feats.device).view(B, 1).expand(-1, target_n)
    xyz_down = xyz[batch_idx, idx, :]
    feats_down = feats[batch_idx, idx, :]
    return xyz_down, feats_down


# ══════════════════════════════════════════════════
# 图残差模块 (v3: QKV 注意力 + 下采样)
# ══════════════════════════════════════════════════

class GraphResidualBlock(nn.Module):
    r"""图残差模块 v3：QKV 局部注意力 + 坐标残差 + 加权下采样。

    QKV 注意力数据流::

        p_i, f_i (中心点)
           │
        KNN(p) → neighbors p_j, f_j
           │
        Q = Linear([f_i, p_i])              ← 中心查询 (linear(f) + linear(p))
        K = Linear([f_j, p_j - p_i])        ← 邻居键 (特征图 + 坐标图)
        V = Linear([f_j, f_j - f_i])        ← 邻居值 (特征图 + 差分)
           │
        attn = sigmoid( MLP([Q, K, Q⊙K, Δp]) )
           │
        out = Σ(attn · V)                   ← 加权聚合
           │
        out = Linear(out) + Linear_res(p_i) ← 坐标残差
           │
        [weighted_downsample] → 下采样输出
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
        C_in, C_out = in_channels, out_channels

        # Q: 中心特征(4+C_in) → C_out
        self.linear_q = nn.Linear(C_in + 4, C_out)

        # K: 邻居特征+Δ坐标 (C_in + 4) → C_out
        self.linear_k = nn.Linear(C_in + 4, C_out)

        # V: 邻居特征+Δ特征 (C_in + C_in) → C_out
        self.linear_v = nn.Linear(C_in * 2, C_out)

        # 标量注意力权重: [Q, K, Q⊙K, Δp] (3*C_out + 4) → 1
        self.attn_mlp = nn.Sequential(
            nn.Linear(C_out * 3 + 4, C_out // 2),
            nn.ReLU(inplace=True),
            nn.Linear(C_out // 2, 1),
        )

        # 输出投影
        self.linear_out = nn.Linear(C_out, C_out)

        # 坐标投影用于残差: 4D → C_out
        self.linear_res = nn.Linear(4, C_out)

        # 归一化
        self.ln = nn.LayerNorm(C_in)

        # 输出激活
        self.act = nn.ReLU(inplace=True)

    def _forward_impl(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """核心前向逻辑。返回 (f_out, p, knn_idx) 或 (f_out, p_down, knn_idx_new)。"""
        B, N, C_in = f.shape
        k = knn_idx.shape[-1]
        C_out = self.linear_out.out_features

        # ── 层归一化 ──
        f_norm = self.ln(f)  # (B, N, C_in)

        # ── 初始化累积器 ──
        out = f.new_zeros(B, N, C_out)

        # ── 逐邻居聚合（轻量，无需一次性创建 (B,N,k,C) 张量）──
        for j in range(k):
            nbr_idx = knn_idx[:, :, j]  # (B, N)
            batch_idx = torch.arange(B, device=f.device).view(B, 1).expand(-1, N)

            # 邻居坐标和特征
            p_nbr = p[batch_idx, nbr_idx, :]   # (B, N, 4)
            f_nbr = f_norm[batch_idx, nbr_idx, :]  # (B, N, C_in)

            # Δ
            dp = p_nbr - p                     # (B, N, 4)
            df = f_nbr - f_norm                # (B, N, C_in)

            # QKV
            Q = self.linear_q(torch.cat([f_norm, p], dim=-1))      # (B, N, C_out)
            K = self.linear_k(torch.cat([f_nbr, dp], dim=-1))      # (B, N, C_out)
            V = self.linear_v(torch.cat([f_nbr, df], dim=-1))      # (B, N, C_out)

            # 标量注意力
            attn_in = torch.cat([Q, K, Q * K, dp], dim=-1)        # (B, N, 3*C_out+4)
            attn_w = self.attn_mlp(attn_in)                        # (B, N, 1)
            attn_w = torch.sigmoid(attn_w)

            # 加权累积
            out = out + attn_w * V  # (B, N, C_out)

        # ── 取平均（除 k） + 残差 ──
        out = out / k
        out = self.linear_out(out) + self.linear_res(p)  # 坐标残差
        f_out = self.act(out)

        # ── 下采样 ──
        if self.downsample:
            target_n = N // 2
            p_down, f_out = weighted_downsample(p, f_out, target_n)
            # 对新坐标重算 KNN
            knn_new = knn_4d(p_down, min(self.k, target_n - 1))
            return f_out, p_down, knn_new
        return f_out, p, knn_idx

    def forward(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向（训练时启用梯度检查点）。"""
        if self.use_checkpoint and self.training and _HAS_CKPT:
            return _ckpt(self._forward_impl, p, f, knn_idx, use_reentrant=False)
        return self._forward_impl(p, f, knn_idx)


# ══════════════════════════════════════════════════
# 外层多任务网络
# ══════════════════════════════════════════════════

class GraphResidualMultiTaskNet(nn.Module):
    """图残差多任务网络 v3。

    通道: 4 → 32 → 64 → 128 → 256 → 512
    点数: 1024 → 512 → 256 → 128 → 64
    池化: Max+Avg → 1024 维
    双头: cls (num_classes) + box (6)

    Args:
        num_classes: 类别数。
        k: KNN 近邻数。
        use_checkpoint: 梯度检查点。
        dropout: 预测头 dropout。
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 12,
        use_checkpoint: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.k = k

        # Stem: (B,N,4) → (B,N,32)
        self.stem = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )

        # 4 层图残差，每层减半
        block_cfg = dict(k=k, use_checkpoint=use_checkpoint)
        self.block1 = GraphResidualBlock(32, 64, downsample=True, **block_cfg)
        self.block2 = GraphResidualBlock(64, 128, downsample=True, **block_cfg)
        self.block3 = GraphResidualBlock(128, 256, downsample=True, **block_cfg)
        self.block4 = GraphResidualBlock(256, 512, downsample=True, **block_cfg)

        pooled_dim = 1024  # 512 * 2

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Box 头
        self.box_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 6),
        )

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            points: (B, N, 4) — x, y, z, intensity。

        Returns:
            {'logits': (B, num_classes), 'box_pred': (B, 6)}
        """
        B, N, _ = points.shape

        # ── 初始化 ──
        p = points.clone()                         # (B, N, 4) 坐标+强度
        f = self.stem(points)                      # (B, N, 32) 初始特征
        knn_idx = knn_4d(p, min(self.k, N - 1))    # (B, N, k)

        # ── 4 层图残差 ──
        f, p, knn_idx = self.block1(p, f, knn_idx)  # 1024→512,  32→64
        f, p, knn_idx = self.block2(p, f, knn_idx)  # 512→256,   64→128
        f, p, knn_idx = self.block3(p, f, knn_idx)  # 256→128,  128→256
        f, p, knn_idx = self.block4(p, f, knn_idx)  # 128→64,   256→512

        # ── 全局池化 ──
        f_max = f.max(dim=1)[0]    # (B, 512)
        f_avg = f.mean(dim=1)      # (B, 512)
        f_pooled = torch.cat([f_max, f_avg], dim=1)  # (B, 1024)

        # ── 双头 ──
        logits = self.cls_head(f_pooled)
        box_preds = self.box_head(f_pooled)

        return {'logits': logits, 'box_pred': box_preds}


# ══════════════════════════════════════════════════
# 快速验证
# ══════════════════════════════════════════════════

if __name__ == '__main__':
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    print("=== GraphResidualMultiTaskNet v3 验证 ===\n")

    B, N = 2, 1024
    dummy = torch.randn(B, N, 4)

    model = GraphResidualMultiTaskNet(num_classes=26, k=12, use_checkpoint=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f} M")

    model.eval()
    with torch.no_grad():
        out = model(dummy)

    print(f"输入:   {dummy.shape}")
    print(f"logits: {out['logits'].shape}")
    print(f"box:    {out['box_pred'].shape}")
    assert out['logits'].shape == (B, 26)
    assert out['box_pred'].shape == (B, 6)

    # 通道验证
    for i, block in enumerate([model.block1, model.block2, model.block3, model.block4], 1):
        c_in = block.linear_q.in_features - 4
        c_out = block.linear_out.out_features
        ds = 'Y' if block.downsample else 'N'
        print(f"  block{i}: C_in={c_in:3d} C_out={c_out:3d} downsample={ds}")

    # 兼容性
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
    from utils.loss import split_cls_and_box_predictions
    l, b = split_cls_and_box_predictions(out)
    assert l is not None and b is not None

    print("\n全部通过!")

    # ══════════════════════════════════════════════
    # GPU 显存测试
    # ══════════════════════════════════════════════
    print("\n=== GPU 显存测试 ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        try:
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, 'total_memory'):
                total_gb = props.total_memory / 1024**3
            elif hasattr(props, 'total_mem'):
                total_gb = props.total_mem / 1024**3
            else:
                total_gb = 0.0
            if total_gb > 0:
                print(f"总显存: {total_gb:.1f} GB")
        except Exception:
            pass
        print()

        for bs in [4, 8, 16, 32]:
            try:
                m = GraphResidualMultiTaskNet(num_classes=26, k=12, use_checkpoint=True).cuda()
                pts = torch.randn(bs, N, 4).cuda()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                m.train()
                o = m(pts)
                loss = o['logits'].sum() + o['box_pred'].sum()
                loss.backward()
                peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  B={bs:2d}: peak {peak:6.0f} MB")
                del m, pts, o, loss
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  B={bs:2d}: OOM!")
                torch.cuda.empty_cache()
                break
    else:
        print("无 CUDA，跳过。")
