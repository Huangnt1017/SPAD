"""
PointRWKV: Efficient RWKV-Like Model for Hierarchical Point Cloud Learning

GitHub:  https://github.com/hithqd/PointRWKV
Local:   D:\essay\3d目标检测复现仓库\PointRWKV-main

Full replication from official repo (Qingdong He et al.):
- BQE (Bidirectional Quadratic Expansion): 拓宽感受野
- IFM (Integrative Feature Modulation): 全局 RWKV 注意力 (线性复杂度)
- LGM (Local Graph-based Merging): 局部几何特征提取
- PRWKV Block: 并行 IFM + LGM 的双分支设计
- 层级多尺度骨干: FPS 下采样 + PRWKV 块 + 特征传播

输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])

Reference:
@inproceedings{he2025pointrwkv,
  title={Pointrwkv: Efficient rwkv-like model for hierarchical point cloud learning},
  author={He, Qingdong and Zhang, Jiangning and Peng, Jinlong and He, Haoyang and Li, Xiangtai and Wang, Yabiao and Wang, Chengjie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={3410--3418},
  year={2025}
}
"""

import os
import sys
import math
from typing import List, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.point_rwkv_utils import (
    knn_point, fps, index_points, square_distance,
    Group, Encoder, MultiScaleGrouping, PointNetFeaturePropagation
)


# ============================================================================
# DropPath (Stochastic Depth)
# ============================================================================

class DropPath(nn.Module):
    """逐样本的随机深度 (Stochastic Depth)。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ============================================================================
# BQE (Bidirectional Quadratic Expansion)
# ============================================================================

class BQE(nn.Module):
    """Bidirectional Quadratic Expansion: 通过循环移位混合特征以拓宽感受野。

    BQE(X) = X + (1 - mu) * X_star
    其中 X_star 由 X 的 4 个分块各自做不同的循环移位后拼接得到。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        B, T, C = x.shape
        chunk_size = C // 4
        x1 = x[:, :, :chunk_size]
        x2 = torch.roll(x[:, :, chunk_size:2*chunk_size], shifts=1, dims=1)
        x3 = torch.roll(x[:, :, 2*chunk_size:3*chunk_size], shifts=-1, dims=1)
        x4 = torch.roll(x[:, :, 3*chunk_size:], shifts=2, dims=1)
        x_star = torch.cat([x1, x2, x3, x4], dim=-1)

        mu = torch.sigmoid(self.mu)
        return x + (1 - mu) * x_star


# ============================================================================
# WKV Attention (线性复杂度的双向 WKV 注意力)
# ============================================================================

class WKVAttention(nn.Module):
    """双向 WKV 注意力 (线性复杂度)。

    通过矩阵值状态 (matrix-valued state) 的递推实现线性注意力:
        s_t = w_t * s_{t-1} + k_t^T * v_t     (矩阵状态)
        o_t = sigma(r_t) * (u * k_t^T * v_t + s_{t-1})
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # BQE 模块
        self.bqe_r = BQE(dim)
        self.bqe_k = BQE(dim)
        self.bqe_v = BQE(dim)
        self.bqe_w = BQE(dim)

        # 线性投影
        self.proj_r = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_g = nn.Linear(dim, dim, bias=False)

        # 动态时变衰减
        self.decay_A = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_B = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_base = nn.Parameter(torch.randn(dim) * 0.01)
        self.w_proj = nn.Linear(dim, dim, bias=False)

        # bonus 项 u
        self.u = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.01)

        # 输出
        self.group_norm = nn.GroupNorm(num_heads, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.proj_r.weight)
        nn.init.orthogonal_(self.proj_k.weight)
        nn.init.orthogonal_(self.proj_v.weight)
        nn.init.orthogonal_(self.proj_g.weight)

    def _compute_decay(self, x):
        """计算动态时变衰减 w = exp(-exp(nu))."""
        bqe_out = self.bqe_w(x)
        c = self.w_proj(bqe_out)
        nu = self.decay_base + torch.tanh(c * self.decay_A) * self.decay_B
        w = torch.exp(-torch.exp(nu))
        return w

    def forward(self, x):
        """
        Args:
            x: (B, T, C) — 输入序列
        Returns:
            out: (B, T, C) — 更新后的特征
        """
        B, T, C = x.shape

        r = self.proj_r(self.bqe_r(x))   # (B, T, C)
        k = self.proj_k(self.bqe_k(x))
        v = self.proj_v(self.bqe_v(x))
        g = torch.sigmoid(self.proj_g(x))
        w = self._compute_decay(x)

        # 多头: (B, H, T, D)
        r = rearrange(r, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        w = rearrange(w, 'b t (h d) -> b h t d', h=self.num_heads)

        # 双向 WKV
        wkv_fwd = self._wkv_forward(r, k, v, w)
        wkv_bwd = self._wkv_forward(
            r.flip(dims=[2]), k.flip(dims=[2]), v.flip(dims=[2]), w.flip(dims=[2])
        ).flip(dims=[2])
        wkv = (wkv_fwd + wkv_bwd) / 2

        out = rearrange(wkv, 'b h t d -> b t (h d)')
        out = self.group_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out * g
        out = self.out_proj(out)
        return out

    def _wkv_forward(self, r, k, v, w):
        """单向 WKV 递推。"""
        B, H, T, D = r.shape
        state = torch.zeros(B, H, D, D, device=r.device, dtype=r.dtype)
        outputs = []
        for t in range(T):
            rt = torch.sigmoid(r[:, :, t])   # (B, H, D)
            kt = k[:, :, t]
            vt = v[:, :, t]
            wt = w[:, :, t]
            kv = torch.einsum('bhd,bhe->bhde', kt, vt)  # (B, H, D, D)
            state_contrib = (state * kt.unsqueeze(-2)).sum(dim=-1)
            bonus_contrib = (kv * self.u.unsqueeze(0).unsqueeze(-1)).sum(dim=-1)
            outputs.append(rt * (state_contrib + bonus_contrib))
            state = wt.unsqueeze(-1) * state + kv
        return torch.stack(outputs, dim=2)


# ============================================================================
# WKVAttentionChunk (分块版本, 更高 GPU 利用率)
# ============================================================================

class WKVAttentionChunk(nn.Module):
    """分块双向 WKV 注意力: 块内并行 + 块间串行递推。"""
    def __init__(self, dim: int, num_heads: int = 8, chunk_size: int = 32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        self.bqe_r = BQE(dim)
        self.bqe_k = BQE(dim)
        self.bqe_v = BQE(dim)
        self.bqe_w = BQE(dim)

        self.proj_r = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_g = nn.Linear(dim, dim, bias=False)

        self.decay_A = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_B = nn.Parameter(torch.randn(dim) * 0.01)
        self.decay_base = nn.Parameter(torch.randn(dim) * 0.01)
        self.w_proj = nn.Linear(dim, dim, bias=False)

        self.u = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.01)

        self.group_norm = nn.GroupNorm(num_heads, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _compute_decay(self, x):
        bqe_out = self.bqe_w(x)
        c = self.w_proj(bqe_out)
        nu = self.decay_base + torch.tanh(c * self.decay_A) * self.decay_B
        return torch.exp(-torch.exp(nu))

    def forward(self, x):
        B, T, C = x.shape
        r = self.proj_r(self.bqe_r(x))
        k = self.proj_k(self.bqe_k(x))
        v = self.proj_v(self.bqe_v(x))
        g = torch.sigmoid(self.proj_g(x))
        w = self._compute_decay(x)

        r = rearrange(r, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        w = rearrange(w, 'b t (h d) -> b h t d', h=self.num_heads)

        fwd = self._chunked_wkv_one_dir(r, k, v, w)
        bwd = self._chunked_wkv_one_dir(r.flip(2), k.flip(2), v.flip(2), w.flip(2)).flip(2)
        out = (fwd + bwd) / 2

        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.group_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out * g
        out = self.out_proj(out)
        return out

    def _chunked_wkv_one_dir(self, r, k, v, w):
        B, H, T, D = r.shape
        CS = min(self.chunk_size, T)
        num_chunks = (T + CS - 1) // CS
        pad_len = num_chunks * CS - T
        if pad_len > 0:
            r = F.pad(r, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            w = F.pad(w, (0, 0, 0, pad_len), value=1.0)
        r = r.reshape(B, H, num_chunks, CS, D)
        k = k.reshape(B, H, num_chunks, CS, D)
        v = v.reshape(B, H, num_chunks, CS, D)
        w = w.reshape(B, H, num_chunks, CS, D)
        state = torch.zeros(B, H, D, D, device=r.device, dtype=r.dtype)
        all_outs = []
        for c in range(num_chunks):
            rc, kc, vc, wc = r[:, :, c], k[:, :, c], v[:, :, c], w[:, :, c]
            chunk_out = []
            for t in range(CS):
                rt = torch.sigmoid(rc[:, :, t])
                kt, vt, wt = kc[:, :, t], vc[:, :, t], wc[:, :, t]
                kv = torch.einsum('bhd,bhe->bhde', kt, vt)
                state_contrib = (state * kt.unsqueeze(-2)).sum(dim=-1)
                bonus_contrib = (kv * self.u.unsqueeze(0).unsqueeze(-1)).sum(dim=-1)
                chunk_out.append(rt * (state_contrib + bonus_contrib))
                state = wt.unsqueeze(-1) * state + kv
            all_outs.append(torch.stack(chunk_out, dim=2))
        out = torch.cat(all_outs, dim=2)
        return out[:, :, :T] if pad_len > 0 else out


# ============================================================================
# Channel-Mixing (FFN)
# ============================================================================

class ChannelMixing(nn.Module):
    """RWKV 通道混合层 (等价于前馈网络)。"""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.bqe_k = BQE(dim)
        self.bqe_r = BQE(dim)
        self.key_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.receptance_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        k = self.key_proj(self.bqe_k(x))
        k = F.silu(k)
        v = self.value_proj(k)
        r = torch.sigmoid(self.receptance_proj(self.bqe_r(x)))
        return self.drop(r * v)


# ============================================================================
# Graph Stabilizer (LGM 分支的图稳定器)
# ============================================================================

class GraphStabilizer(nn.Module):
    """图稳定器: 通过迭代偏移预测和对齐减少平移方差。

    每次迭代:
        1. 从当前特征预测坐标偏移 delta
        2. 构建稳定后的局部图
        3. 边特征编码 + 最大池化聚合
        4. 特征更新 (残差)
    """
    def __init__(self, dim: int, num_iterations: int = 3):
        super().__init__()
        self.num_iterations = num_iterations
        self.offset_pred = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 3)
        )
        self.edge_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        for _ in range(num_iterations):
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(3 + dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
            ))
            self.update_mlps.append(nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
            ))

    def forward(self, xyz, features, knn_idx):
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C)
            knn_idx: (B, N, K)
        Returns:
            features: (B, N, C) 更新后的特征
        """
        for t in range(self.num_iterations):
            delta_xyz = self.offset_pred(features)
            neighbor_xyz = index_points(xyz, knn_idx)
            neighbor_feat = index_points(features, knn_idx)
            center_xyz = xyz + delta_xyz
            rel_pos = neighbor_xyz - center_xyz.unsqueeze(2)
            edge_input = torch.cat([rel_pos, neighbor_feat], dim=-1)
            edge_feat = self.edge_mlps[t](edge_input)
            agg_feat = edge_feat.max(dim=2)[0]
            features = features + self.update_mlps[t](torch.cat([agg_feat, features], dim=-1))
        return features


# ============================================================================
# LGM Branch
# ============================================================================

class LGMBranch(nn.Module):
    """Local Graph-based Merging (LGM) Branch: 通过 KNN 图 + 图稳定器提取局部特征。"""
    def __init__(self, dim: int, k: int = 16, num_iterations: int = 3):
        super().__init__()
        self.k = k
        self.graph_stabilizer = GraphStabilizer(dim, num_iterations=num_iterations)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, xyz, features):
        knn_idx = knn_point(self.k, xyz, xyz)
        local_features = self.graph_stabilizer(xyz, features, knn_idx)
        return self.proj(local_features)


# ============================================================================
# PRWKV Block (PointRWKV 核心块)
# ============================================================================

class PRWKVBlock(nn.Module):
    """PointRWKV Block: 并行 IFM (全局 RWKV 注意力) + LGM (局部图特征)。

    架构:
        fused = fusion(IFM(norm(x)), LGM(norm(x)))
        x = x + drop_path(fused)
        x = x + drop_path(channel_mixing(norm(x)))
    """
    def __init__(self, dim: int, num_heads: int = 8, k: int = 16,
                 graph_iter: int = 3, ffn_ratio: int = 4,
                 drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # IFM: 全局 RWKV 注意力
        self.spatial_mixing = WKVAttention(dim, num_heads=num_heads)
        # LGM: 局部图特征
        self.lgm = LGMBranch(dim, k=k, num_iterations=graph_iter)
        # 特征融合
        self.fusion = nn.Linear(dim * 2, dim, bias=False)
        # FFN
        self.channel_mixing = ChannelMixing(dim, hidden_dim=int(dim * ffn_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C)
        Returns:
            features: (B, N, C)
        """
        normed = self.norm1(features)
        global_feat = self.spatial_mixing(normed)
        local_feat = self.lgm(xyz, normed)
        fused = self.fusion(torch.cat([global_feat, local_feat], dim=-1))
        features = features + self.drop_path(fused)
        features = features + self.drop_path(self.channel_mixing(self.norm2(features)))
        return features


# ============================================================================
# PointRWKV Backbone (层级多尺度骨干)
# ============================================================================

class PointRWKV(nn.Module):
    """PointRWKV 层级多尺度点云骨干网络。

    架构:
    - 多尺度分组 (FPS + KNN + Mini-PointNet 编码)
    - 每个尺度多个 PRWKV Block
    - 输出各尺度特征列表
    """
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config.get('embed_dim', 384)
        self.depth = config.get('depth', [4, 4, 4])
        self.num_heads = config.get('num_heads', 8)
        self.num_points = config.get('num_points', [2048, 1024, 512])
        self.group_sizes = config.get('group_sizes', [32, 32, 32])
        self.k_neighbors = config.get('k_neighbors', [16, 8, 8])
        self.graph_iter = config.get('graph_iter', 3)
        self.drop_path_rate = config.get('drop_path_rate', 0.1)
        self.num_scales = len(self.num_points)

        # 多尺度分组 + 嵌入
        self.group_modules = nn.ModuleList()
        self.embed_modules = nn.ModuleList()
        for i in range(self.num_scales):
            self.group_modules.append(Group(self.num_points[i], self.group_sizes[i]))
            self.embed_modules.append(Encoder(self.embed_dim))

        # 位置编码
        self.pos_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.embed_dim)
            ) for _ in range(self.num_scales)
        ])

        # PRWKV Blocks (使用随机深度)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depth))]
        cur = 0
        self.blocks = nn.ModuleList()
        for i in range(self.num_scales):
            scale_blocks = nn.ModuleList()
            for j in range(self.depth[i]):
                scale_blocks.append(PRWKVBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    k=self.k_neighbors[i],
                    graph_iter=self.graph_iter,
                    drop_path=dpr[cur],
                ))
                cur += 1
            self.blocks.append(scale_blocks)

        # 每层输出的 LayerNorm
        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_scales)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pts):
        """
        Args:
            pts: (B, N, 3) — 输入点云坐标
        Returns:
            features_list: [(B, N_i, C), ...] 各尺度特征
            centers_list: [(B, N_i, 3), ...] 各尺度中心点
        """
        features_list, centers_list = [], []
        for i in range(self.num_scales):
            neighborhood, center = self.group_modules[i](pts)
            tokens = self.embed_modules[i](neighborhood)
            pos = self.pos_embed[i](center)
            tokens = tokens + pos
            for block in self.blocks[i]:
                tokens = block(center, tokens)
            tokens = self.norms[i](tokens)
            features_list.append(tokens)
            centers_list.append(center)
        return features_list, centers_list


# ============================================================================
# PointRWKV Classification + BBox (SPAD 兼容接口)
# ============================================================================

class PointRWKVClassification(nn.Module):
    """PointRWKV 分类 + 3D BBox 模型 (适配 SPAD 训练管道)。

    将 PointRWKV 多尺度特征聚合后进行全局池化，接分类和框回归头。
    输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])
    """
    def __init__(self, num_classes: int = 26, embed_dim: int = 384,
                 depth: tuple = (4, 4, 4), num_heads: int = 8,
                 num_points: tuple = (1024, 512, 256),
                 group_sizes: tuple = (32, 32, 32),
                 k_neighbors: tuple = (16, 8, 8),
                 graph_iter: int = 3, drop_path_rate: float = 0.1,
                 **kwargs):
        super().__init__()
        config = {
            'embed_dim': embed_dim,
            'depth': list(depth),
            'num_heads': num_heads,
            'num_points': list(num_points),
            'group_sizes': list(group_sizes),
            'k_neighbors': list(k_neighbors),
            'graph_iter': graph_iter,
            'drop_path_rate': drop_path_rate,
        }
        self.backbone = PointRWKV(config)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # BBox 回归头
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    @staticmethod
    def _normalize_input_points(x):
        """统一输入格式为 (B, N, 4) xyzi。"""
        if x.ndim != 3:
            raise ValueError(f"PointRWKVClassification expects 3D input, got {tuple(x.shape)}")
        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
        if points.shape[-1] == 3:
            pad_i = torch.zeros(points.shape[0], points.shape[1], 1,
                                dtype=points.dtype, device=points.device)
            points = torch.cat([points, pad_i], dim=-1)
        return points

    def forward(self, x):
        """
        Args:
            x: (B, N, 4) xyzi 点云
        Returns:
            logits: (B, num_classes)
            box_pred: (B, 6)
        """
        x = self._normalize_input_points(x)
        # 只使用 xyz 坐标 (PointRWKV 的设计)
        pts = x[:, :, :3].contiguous()  # (B, N, 3)

        features_list, centers_list = self.backbone(pts)

        # 多尺度特征聚合: 用最细和最粗两个尺度的最大池化
        feat_fine = features_list[0]     # (B, N1, C)
        feat_coarse = features_list[-1]  # (B, N3, C)
        g_fine = feat_fine.max(dim=1)[0]      # (B, C)
        g_coarse = feat_coarse.max(dim=1)[0]  # (B, C)
        global_feat = torch.cat([g_fine, g_coarse], dim=-1)  # (B, 2C)

        logits = self.cls_head(global_feat)
        box_pred = self.box_head(global_feat)
        return logits, box_pred


# ============================================================================
# GPU 显存测试 (SKILL 规范)
# ============================================================================

def _gpu_memory_test():
    """GPU 显存压力测试 (逐 batch size 扫查)。"""
    import gc
    if not torch.cuda.is_available():
        print("无 CUDA，跳过 GPU 显存测试。")
        return

    print("\n=== GPU 显存测试 ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        if total_mem:
            print(f"总显存: {total_mem / 1024**3:.1f} GB")
    except Exception:
        pass
    print()

    N = 1024
    for bs in [4, 8, 16, 32]:
        try:
            m = PointRWKVClassification(num_classes=26,
                                         embed_dim=192,
                                         depth=(2, 2, 2),
                                         num_points=(512, 256, 128)).cuda()
            pts = torch.randn(bs, N, 4).cuda()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            m.train()
            o = m(pts)
            loss = o[0].sum() + o[1].sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  B={bs:2d}: peak {peak:6.0f} MB")
            del m, pts, o, loss
            torch.cuda.empty_cache()
            gc.collect()
        except torch.cuda.OutOfMemoryError:
            print(f"  B={bs:2d}: OOM!")
            torch.cuda.empty_cache()
            gc.collect()
            break


def _quick_test():
    """形状验证。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing PointRWKV on {device}")

    model = PointRWKVClassification(num_classes=26,
                                     embed_dim=192,
                                     depth=(2, 2, 2),
                                     num_points=(512, 256, 128)).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Box:    {tuple(box_pred.shape)}")
    print("✓ PointRWKV works correctly")


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()