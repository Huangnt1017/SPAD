"""
单光子点云图残差多任务网络 (Graph Residual Multi-Task Network for SPAD)  v4

v4 (严格对齐 model/readme.md 任务 1 设计图)
==========================================================
readme 任务 1 的 block 数据流:

    Input (Point, Feature)
       │           \      /
       │            [ LN ]                  ← LayerNorm
       │            /    \\
       |        [NGF]   [Linear]           ← Flow_A=NGF→GCN  /  Flow_B=Linear→σ
       │          |        |
       │        [GCN]      |
       │          |        |
       │         [σ]      [σ]              ← 两支各自经 sigmoid 激活
       │           \\      /
       │          [Attention]              ← element-wise gating (Flow_A ⊙ Flow_B)
       │              │
       │           [Linear]                ← 特征映射
       │              │
       └──────────> ( ⊕ ) ───> Output      ← 坐标残差: + 原始 Point 经 4→C_out 投影

v3 → v4 关键修正:
    1. 把 QKV self-attention 换回 readme 的"双分支门控" (Flow_A * sigmoid(Flow_B))。
    2. Flow_A 用 EdgeConv 风格的 NGF + GCN: 邻居拼接 ([f_i, f_j - f_i, dp]) → MLP → max-pool。
    3. Flow_B 仅用 Linear 从中心特征预测每点 attention 门 (输出 sigmoid 后 ∈[0,1])。
    4. 输出特征 Linear 后再加 ``Linear_res(p)`` 做坐标残差 (与 v3 一致, readme 强约束)。

v3 保留的非 readme 设计点:
    - 通道阶梯 4 → 32 → 64 → 128 → 256 → 512 (readme 未指定, 维持轻量化设置)
    - 每层 weighted_downsample (1024 → 512 → 256 → 128 → 64, readme 未指定, 保留以便实际可跑)
    - 4D KNN (含 intensity 维, 与 readme "稀疏单光子高噪点云" 处理动机一致)
    - 梯度检查点 (训练时省显存)

References:
    - model/readme.md 任务 1
    - utils/loss.py split_cls_and_box_predictions
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

# 兼容 checkpoint API
try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _HAS_CKPT = True
except (ImportError, AttributeError):
    _HAS_CKPT = False

    def _ckpt(fn, *args, **kwargs):
        return fn(*args)


# ══════════════════════════════════════════════════
# 4D KNN (含 intensity)
# ══════════════════════════════════════════════════

def knn_4d(points: torch.Tensor, k: int) -> torch.Tensor:
    """4D 近邻索引 (x, y, z, intensity)。

    intensity 经 0.1 缩放, 量纲接近坐标维, 避免强度维主导或失效。

    Args:
        points: (B, N, 4) — x, y, z, intensity。
        k: 近邻数 (不含自身)。

    Returns:
        knn_idx: (B, N, k), int64。
    """
    B, N, _ = points.shape
    xyz = points[..., :3]
    intensity = points[..., 3:4] * 0.1

    # (B, N, N) pairwise 距离: ||a||² + ||b||² - 2 a·b
    xx = torch.sum(xyz ** 2, dim=2, keepdim=True)
    ii = torch.sum(intensity ** 2, dim=2, keepdim=True)
    dist_xyz = xx + xx.transpose(2, 1) - 2.0 * torch.bmm(xyz, xyz.transpose(2, 1))
    dist_i = ii + ii.transpose(2, 1) - 2.0 * torch.bmm(intensity, intensity.transpose(2, 1))
    dist = dist_xyz + dist_i

    # 排除自身 (设为 +inf)
    diag = torch.eye(N, device=points.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    dist.masked_fill_(diag, float("inf"))

    _, knn_idx = torch.topk(dist, k, dim=2, largest=False)
    return knn_idx


# ══════════════════════════════════════════════════
# 加权随机下采样 (按特征 L2 范数采样)
# ══════════════════════════════════════════════════

def weighted_downsample(
    xyz: torch.Tensor,
    feats: torch.Tensor,
    target_n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 ``feats.norm`` 作为重要性的无放回随机采样。

    Args:
        xyz: (B, N, 4) 点坐标 + intensity。
        feats: (B, N, C) 每点特征。
        target_n: 目标采样点数, ``target_n >= N`` 时直接返回原值。

    Returns:
        xyz_down: (B, target_n, 4)
        feats_down: (B, target_n, C)
    """
    B, N, _ = feats.shape
    if target_n >= N:
        return xyz, feats

    scores = feats.norm(p=2, dim=-1).clamp(min=1e-8)         # (B, N)
    probs = scores / scores.sum(dim=1, keepdim=True)         # (B, N)
    idx = torch.multinomial(probs, target_n, replacement=False)   # (B, target_n)
    batch_idx = torch.arange(B, device=feats.device).view(B, 1).expand(-1, target_n)
    return xyz[batch_idx, idx, :], feats[batch_idx, idx, :]


# ══════════════════════════════════════════════════
# Flow_A: NGF + GCN  (EdgeConv 风格邻域聚合)
# ══════════════════════════════════════════════════

class _NGF_GCN(nn.Module):
    """NGF (Neighborhood Graph Feature) + GCN 聚合, 对应 readme 任务 1 Flow_A。

    数据流:
        f: (B, N, C_in), p: (B, N, 4), knn_idx: (B, N, k)
            → 收集邻居 f_j, p_j
            → 构造边特征: [f_i, f_j - f_i, p_j - p_i]    (B, N, k, 2C_in + 4)
            → 共享 MLP 升到 C_out                          (B, N, k, C_out)
            → max-pool over k                              (B, N, C_out)

    这是 DGCNN EdgeConv 的经典写法, 邻居特征差分 + 中心特征 + 坐标差三者拼接,
    既保留中心信息又显式利用邻域相对几何 (符合 readme "通过 KNN 构图" +
    "由图卷积聚合局部上下文")。max-pool 等价于 readme 图中的 GCN 聚合算子。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 共享 MLP: (2*C_in + 4) → C_out → C_out
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 4, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        f_norm: torch.Tensor,
        p: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f_norm: (B, N, C_in) — 经 LN 归一化后的特征。
            p: (B, N, 4) — 坐标 + intensity。
            knn_idx: (B, N, k) — 近邻索引。

        Returns:
            agg: (B, N, C_out) — GCN 聚合后的局部上下文。
        """
        B, N, C_in = f_norm.shape
        k = knn_idx.shape[-1]

        # 逐邻居 loop 累积, 避免一次性创建 (B, N, k, C) 的 4D tensor 撑爆显存。
        # 用一个 max-buffer 来做 max-pool over k (running max), 内存 O(B*N*C_out)。
        max_buf: torch.Tensor | None = None
        batch_idx = torch.arange(B, device=f_norm.device).view(B, 1).expand(-1, N)

        for j in range(k):
            nbr_idx = knn_idx[:, :, j]               # (B, N)
            f_nbr = f_norm[batch_idx, nbr_idx, :]    # (B, N, C_in)
            p_nbr = p[batch_idx, nbr_idx, :]         # (B, N, 4)

            # 边特征: 中心 f_i, 邻居差 f_j - f_i, 坐标差 p_j - p_i
            edge = torch.cat([f_norm, f_nbr - f_norm, p_nbr - p], dim=-1)   # (B, N, 2C_in + 4)
            e_feat = self.mlp(edge)                                          # (B, N, C_out)

            # running max over k
            max_buf = e_feat if max_buf is None else torch.maximum(max_buf, e_feat)

        # max_buf 在此处已非 None (k ≥ 1, 由 GraphResidualBlock 保证)
        assert max_buf is not None
        return max_buf


# ══════════════════════════════════════════════════
# Graph Residual Block  (v4: 严格按 readme 双分支门控)
# ══════════════════════════════════════════════════

class GraphResidualBlock(nn.Module):
    """图残差模块 v4 (严格对齐 readme 任务 1 图):

        Input (p, f)
            ↓
            LN(f)                       ← f_norm
           / \\
       Flow_A  Flow_B
        NGF    Linear
        GCN
         ↓      ↓
        σ      σ                       ← 两支各自 sigmoid
          \\  //
        Attention (⊙ element-wise)
            ↓
        Linear (特征映射)
            ↓
            + Linear_res(p)             ← 坐标残差 (与原始 4D Point ⊕)
            ↓
        ReLU
            ↓
        [可选 weighted_downsample]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 16,
        downsample: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.k = k
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint

        # LN: 在 C_in 维做层归一化, 对应 readme 设计图入口的 [LN]
        self.ln = nn.LayerNorm(in_channels)

        # Flow_A: NGF (KNN 构图) + GCN (邻居聚合)
        self.flow_a = _NGF_GCN(in_channels=in_channels, out_channels=out_channels)

        # Flow_B: 单层 Linear, 产生 attention 门控 (每点 C_out 维)
        self.flow_b = nn.Linear(in_channels, out_channels)

        # 输出特征映射 Linear (readme 图里 Attention 之后的那个 Linear)
        self.linear_out = nn.Linear(out_channels, out_channels)

        # 坐标残差路径: 把 4D Point 投影到 C_out 维, 再与主流相加
        # (readme 写"逐元素相加", shape 必须一致, 故走一个投影)
        self.linear_res = nn.Linear(4, out_channels)

        # 输出激活
        self.act = nn.ReLU(inplace=True)

    def _forward_impl(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """核心前向: 返回 (f_out, p, knn_idx) 或下采样后的 (f_down, p_down, knn_new)。"""
        B, N, _ = f.shape

        # ── 入口归一化 ──
        f_norm = self.ln(f)                          # (B, N, C_in)

        # ── Flow_A: NGF → GCN ──
        a = self.flow_a(f_norm, p, knn_idx)          # (B, N, C_out)
        a = torch.sigmoid(a)                          # readme: σ

        # ── Flow_B: Linear ──
        b = self.flow_b(f_norm)                       # (B, N, C_out)
        b = torch.sigmoid(b)                          # readme: σ

        # ── Attention: 双门 element-wise 相乘 ──
        merged = a * b                                # (B, N, C_out)

        # ── 输出特征映射 ──
        out = self.linear_out(merged)                 # (B, N, C_out)

        # ── 坐标残差 ⊕ 原始 Point ──
        out = out + self.linear_res(p)                # (B, N, C_out)
        f_out = self.act(out)

        # ── 可选层间下采样 ──
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
        """对外接口 (训练阶段启用梯度检查点省显存)。"""
        if self.use_checkpoint and self.training and _HAS_CKPT:
            return _ckpt(self._forward_impl, p, f, knn_idx, use_reentrant=False)
        return self._forward_impl(p, f, knn_idx)


# ══════════════════════════════════════════════════
# 外层多任务网络
# ══════════════════════════════════════════════════

class GraphResidualMultiTaskNet(nn.Module):
    """图残差多任务网络 v4 (双分支门控 + 坐标残差)。

    通道阶梯: 4 → 32 → 64 → 128 → 256 → 512
    点数阶梯: 1024 → 512 → 256 → 128 → 64 (默认 downsample=True)
    全局池化: max + avg → 1024 维
    双头: cls (num_classes) + box (3, 中心点; 与 SPAD bbox refactor 一致)

    Args:
        num_classes: 分类类别数。
        k: KNN 近邻数 (不含自身)。
        use_checkpoint: 是否启用梯度检查点。
        dropout: 预测头 Dropout 概率。
        box_dim: bbox 输出维度, SPAD 重构后默认 3 (中心点); 早期 ckpt 用 6 (角点)。
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

        # 4 层 Graph Residual Block, 每层 C 翻倍 + N 减半
        block_cfg = dict(k=k, downsample=True, use_checkpoint=use_checkpoint)
        self.block1 = GraphResidualBlock(32, 64, **block_cfg)
        self.block2 = GraphResidualBlock(64, 128, **block_cfg)
        self.block3 = GraphResidualBlock(128, 256, **block_cfg)
        self.block4 = GraphResidualBlock(256, 512, **block_cfg)

        pooled_dim = 1024  # 512 * 2 (max + avg)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Box 头 (SPAD center-only 重构: 默认 3 维中心)
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
            points: (B, N, 4) — x, y, z, intensity (raw 或归一化均可)。

        Returns:
            dict with:
                - 'logits':   (B, num_classes)
                - 'box_pred': (B, box_dim)
        """
        B, N, _ = points.shape

        # ── 初始化坐标 + 特征 + 4D KNN ──
        p = points.clone()                                     # (B, N, 4)
        f = self.stem(points)                                   # (B, N, 32)
        knn_idx = knn_4d(p, min(self.k, N - 1))                 # (B, N, k)

        # ── 4 层图残差 (每层 C 翻倍 + N 减半) ──
        f, p, knn_idx = self.block1(p, f, knn_idx)              # 1024 → 512,  32 → 64
        f, p, knn_idx = self.block2(p, f, knn_idx)              # 512 → 256,   64 → 128
        f, p, knn_idx = self.block3(p, f, knn_idx)              # 256 → 128,  128 → 256
        f, p, knn_idx = self.block4(p, f, knn_idx)              # 128 → 64,   256 → 512

        # ── 全局池化 ──
        f_max = f.max(dim=1)[0]                                 # (B, 512)
        f_avg = f.mean(dim=1)                                    # (B, 512)
        f_pooled = torch.cat([f_max, f_avg], dim=1)             # (B, 1024)

        # ── 双任务输出 ──
        logits = self.cls_head(f_pooled)                         # (B, num_classes)
        box_preds = self.box_head(f_pooled)                      # (B, box_dim)

        return {"logits": logits, "box_pred": box_preds}


# ══════════════════════════════════════════════════
# 快速验证 + GPU 显存测试
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    print("=== GraphResidualMultiTaskNet v4 验证 (readme 任务 1 双分支门控) ===\n")

    B, N = 2, 1024
    dummy = torch.randn(B, N, 4)

    # 默认 3 维 box (SPAD center-only refactor)
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

    # 通道 / 下采样配置打印
    for i, block in enumerate([model.block1, model.block2, model.block3, model.block4], 1):
        c_in = block.flow_b.in_features
        c_out = block.linear_out.out_features
        ds = "Y" if block.downsample else "N"
        print(f"  block{i}: C_in={c_in:3d} C_out={c_out:3d} downsample={ds}")

    # 与 utils.loss 接口兼容性 (split_cls_and_box_predictions)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.loss import split_cls_and_box_predictions
    logits_o, box_o = split_cls_and_box_predictions(out)
    assert logits_o is not None and box_o is not None
    print(f"\n  split_cls_and_box_predictions: logits {logits_o.shape}, box {box_o.shape}")

    print("\nAll checks pass.")

    # ══════════════════════════════════════════════
    # GPU 显存测试
    # ══════════════════════════════════════════════
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
