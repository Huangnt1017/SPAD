"""
Point-MAE: Masked Autoencoders for Point Cloud Self-Supervised Learning

GitHub:  https://github.com/Pang-Yatian/Point-MAE
Local:   D:\essay\3d目标检测复现仓库\Point-MAE-main

完整复现 Point-MAE 分类微调模型 (PointTransformer):
- Group: FPS + KNN 分组构建局部 patches
- Encoder: Mini-PointNet 编码 patches → patch tokens
- TransformerEncoder: 标准 Transformer (含 cls_token) + 位置编码
- 分类头: cls_token + 全局最大池化拼接 → BN → MLP
- 输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])

Reference:
@article{pang2022masked,
  title={Masked autoencoders for point cloud self-supervised learning},
  author={Pang, Yatian and Wang, Wenxiao and Tay, Francis EH and Liu, Wei and Tian, Yonghong and Yuan, Li},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
"""

import os
import sys
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.pointnet_utils import (
    square_distance, index_points, knn_point, farthest_point_sample
)


# ============================================================================
# Group: FPS + KNN 分组 (Point-MAE 官方实现)
# ============================================================================

class Group(nn.Module):
    """FPS 采样关键点 + KNN 获取邻居分组。

    与 Point-MAE 官方实现一致:
    - FPS 采样 num_group 个中心点 (直接返回坐标)
    - KNN 获取每个中心的 group_size 个邻居
    - 以中心为原点归一化邻域坐标
    """
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) — 输入点云坐标
        Returns:
            neighborhood: (B, G, K, 3) — 中心归一化后的局部 patches
            center: (B, G, 3) — 采样中心点坐标
        """
        batch_size, num_points, _ = xyz.shape

        # FPS 采样中心点 (获得索引后收集坐标)
        center_idx = farthest_point_sample(xyz, self.num_group)  # (B, G)
        center = index_points(xyz, center_idx)                    # (B, G, 3)

        # KNN 获取邻居
        idx = knn_point(self.group_size, xyz, center)  # (B, G, K)

        # 扁平索引收集邻居点
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # 中心归一化
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


# ============================================================================
# Encoder: Mini-PointNet 编码器
# ============================================================================

class Encoder(nn.Module):
    """Mini-PointNet 编码器: 将局部点 patches 编码为 patch tokens。

    架构: Conv1d(3→128→256) → MaxPool → 拼接 → Conv1d(512→512→C)
    """
    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, K, 3) — 归一化后的点 patches
        Returns:
            feature_global: (B, G, C) — 编码后的 patch tokens
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))  # (BG, 256, K)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (BG, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # (BG, 512, K)
        feature = self.second_conv(feature)  # (BG, C, K)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (BG, C)
        return feature_global.reshape(bs, g, self.encoder_channel)


# ============================================================================
# Transformer 组件 (与 Point-BERT 共享架构)
# ============================================================================

class Mlp(nn.Module):
    """MLP (FC → Act → Drop → FC → Drop)。"""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """标准多头自注意力。"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block: Self-Attention + MLP + 残差 + LayerNorm。"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer 编码器: 堆叠 Block, tokens + pos embedding 相加后送入。"""
    def __init__(self, embed_dim: int = 768, depth: int = 4, num_heads: int = 12,
                 mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i])
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


# ============================================================================
# DropPath (Stochastic Depth)
# ============================================================================

class DropPath(nn.Module):
    """逐样本随机深度。"""
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
# Point-MAE 分类微调模型
# ============================================================================

class PointMAEClassification(nn.Module):
    """Point-MAE 分类 + 3D BBox 模型 (适配 SPAD 训练管道)。

    对应官方 models/Point_MAE.py 中的 PointTransformer (finetune model)。
    与 Point-BERT 类似, 但分类头更宽 (含 BatchNorm)。

    架构:
    1. Group: FPS + KNN 分组构建局部 patches
    2. Encoder: Mini-PointNet 编码为 patch tokens
    3. cls_token + 位置编码
    4. TransformerEncoder: 标准 Transformer
    5. 分类头: cls_token + 全局最大池化拼接 → BN → MLP

    输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])
    """
    def __init__(self, num_classes: int = 26, trans_dim: int = 384,
                 depth: int = 6, num_heads: int = 6, drop_path_rate: float = 0.1,
                 group_size: int = 32, num_group: int = 64,
                 encoder_dims: int = 256, **kwargs):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims

        # Group + Encoder
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims)

        # 维度缩减: encoder_dims → trans_dim
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        # 位置编码 (MLP: 3 → 128 → trans_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        # Transformer 编码器
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim, depth=depth,
            drop_path_rate=drop_path_rate, num_heads=num_heads,
        )

        self.norm = nn.LayerNorm(trans_dim)

        # 分类头 (Point-MAE 官方设计: 带 BN 的双层 MLP)
        self.cls_head = nn.Sequential(
            nn.Linear(trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # BBox 回归头
        self.box_head = nn.Sequential(
            nn.Linear(trans_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _normalize_input_points(x):
        """统一输入为 (B, N, 4) xyzi。"""
        if x.ndim != 3:
            raise ValueError(f"PointMAEClassification expects 3D input, got {tuple(x.shape)}")
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
        pts = x[:, :, :3].contiguous()  # (B, N, 3)

        # 1. Group: 分组构建 patches
        neighborhood, center = self.group_divider(pts)

        # 2. Encoder: 编码 patch tokens
        group_input_tokens = self.encoder(neighborhood)  # (B, G, encoder_dims)

        # 3. 维度缩减
        group_input_tokens = self.reduce_dim(group_input_tokens)  # (B, G, trans_dim)

        # 4. 准备 cls_token + 位置编码
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # 5. Transformer 编码
        x = self.blocks(x, pos)
        x = self.norm(x)

        # 6. 全局特征聚合: cls_token + max pool
        cls_feat = x[:, 0]
        max_feat = x[:, 1:].max(dim=1)[0]
        concat_f = torch.cat([cls_feat, max_feat], dim=-1)

        logits = self.cls_head(concat_f)
        box_pred = self.box_head(concat_f)
        return logits, box_pred


# ============================================================================
# trunc_normal_ 初始化
# ============================================================================

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """截断正态分布初始化。"""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


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
            m = PointMAEClassification(num_classes=26).cuda()
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
    print(f"Testing Point-MAE on {device}")

    model = PointMAEClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Box:    {tuple(box_pred.shape)}")
    print("✓ Point-MAE works correctly")


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()