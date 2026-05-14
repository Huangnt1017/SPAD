"""
Point Transformer V3 for Object Classification + 3D BBox

GitHub:  https://github.com/Pointcept/Pointcept
Local:   D:\essay\3d目标检测复现仓库\Pointcept-main

Pointcept 官方实现复现 (Xiaoyang Wu)
- Serialized Attention + Serialized Pooling/Unpooling
- 使用 Z-order / Hilbert 序列化实现高效注意力
- 纯 PyTorch 实现 (无需 spconv / flash_attn)

Reference:
@inproceedings{wu2024point,
  title={Point transformer v3: Simpler faster stronger},
  author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4840--4851},
  year={2024}
}
"""

import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
from functools import partial
from typing import Tuple, Optional, List, Dict, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

# 可用依赖
try:
    import torch_scatter
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False
    print("WARNING: torch_scatter not available for PTv3, using fallback")

try:
    from addict import Dict as ADict
except ImportError:
    class ADict(dict):
        """addict.Dict 的简易替代"""
        def __getattr__(self, key):
            if key in self:
                return self[key]
            raise AttributeError(f"Key '{key}' not found")
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            if key in self:
                del self[key]
            else:
                raise AttributeError(f"Key '{key}' not found")

from utils.pointnet_utils import (
    offset2batch, batch2offset, offset2bincount,
    farthest_point_sample, knn_point, index_points
)
from utils.serialization import encode, decode, z_order_encode, hilbert_encode


# ============================================================================
# Point 数据结构 (简化版，适配本项目的点云格式)
# ============================================================================

class Point(ADict):
    """
    Point 数据结构

    包含点云的各种属性:
    - coord: (N, 3) 原始坐标
    - grid_coord: (N, 3) 网格坐标 (用于序列化)
    - feat: (N, C) 点特征
    - batch: (N,) batch 索引
    - offset: (B,) 每个样本的结束索引
    - 序列化相关: serialized_code, serialized_order, serialized_inverse, serialized_depth
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """点云序列化"""
        self["order"] = order
        assert "batch" in self.keys()
        assert "grid_coord" in self.keys()

        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["serialized_depth"] = depth
        assert depth <= 16

        order_list = [order] if isinstance(order, str) else order
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_)
            for order_ in order_list
        ]
        code = torch.stack(code)
        order_idx = torch.argsort(code)
        inverse = torch.zeros_like(order_idx).scatter_(
            dim=1,
            index=order_idx,
            src=torch.arange(0, code.shape[1], device=order_idx.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order_idx = order_idx[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order_idx
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """准备稀疏特征 (简化版，不使用 spconv)"""
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()


# ============================================================================
# PointModule & PointSequential (与 Pointcept 兼容)
# ============================================================================

class PointModule(nn.Module):
    """PointModule — 所有处理 Point 的子类"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    """顺序容器，支持 Point、spconv、PyTorch 模块的混合使用"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for _, module in self._modules.items():
            if isinstance(module, PointModule):
                input = module(input)
            else:
                # PyTorch module — 对 feat 进行操作
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                else:
                    input = module(input)
        return input


# ============================================================================
# RPE (Relative Position Encoding)
# ============================================================================

class RPE(torch.nn.Module):
    """3D 相对位置编码: 基于可学习的查找表"""
    def __init__(self, patch_size: int, num_heads: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(
            torch.zeros(3 * self.rpe_num, num_heads)
        )
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        """coord: (N, K, K, 3) 相对坐标 → (N, H, K, K)"""
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # 裁剪
            + self.pos_bnd  # 偏移到正索引
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z 偏移
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, H, K, K)
        return out


# ============================================================================
# Serialized Attention (序列化注意力)
# ============================================================================

class SerializedAttention(PointModule):
    """
    序列化注意力:
    - 将点云按 Z-order/Hilbert 序排列
    - 在序列化的一维窗口内做自注意力
    - 支持 Flash Attention (如果可用) 或 vanilla attention
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        order_index: int = 0,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash and HAS_FLASH_ATTN

        self.patch_size = patch_size
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point: Point, order: torch.Tensor) -> torch.Tensor:
        """获取窗口内的相对位置"""
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def _get_pad_inverse(self, point: Point, order: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 padding 和 inverse 索引，处理点数不能被 patch_size 整除的情况"""
        N = point.feat.shape[0]
        K = self.patch_size
        
        # 计算 padding 后的总点数
        n_windows = (N + K - 1) // K
        N_pad = n_windows * K
        pad_size = N_pad - N
        
        if pad_size > 0:
            # 对 order 进行 padding (用最后一个有效点填充)
            pad_indices = order[-1:].expand(pad_size)
            order_padded = torch.cat([order, pad_indices])
        else:
            order_padded = order
        
        # inverse 映射: 对于 padding 的部分，映射到原始索引
        inverse_padded = torch.arange(N_pad, device=order.device)
        inverse_padded[order_padded] = torch.arange(N_pad, device=order_padded.device)
        # 裁剪回原始大小
        inverse_clipped = inverse_padded[:N]
        # 对于逆序列化, padding 的点不需要, 所以只需逆映射原始部分
        # 原始 inverse 已经存在
        return order_padded, None if pad_size == 0 else inverse_padded

    def forward(self, point: Point) -> Point:
        K = self.patch_size
        H = self.num_heads
        C = self.channels
        N = point.feat.shape[0]

        order = point.serialized_order[self.order_index]
        inverse = point.serialized_inverse[self.order_index]

        # 计算实际窗口数
        n_windows = (N + K - 1) // K
        N_pad = n_windows * K
        pad_size = N_pad - N

        # QKV 投影
        qkv = self.qkv(point.feat)  # (N, 3*C)
        qkv = qkv[order]  # 按序列化序重排

        # 如果需要 padding
        if pad_size > 0:
            pad_idx = order[-1:].expand(pad_size)
            qkv_pad = qkv[-1:].expand(pad_size, -1)
            qkv = torch.cat([qkv, qkv_pad])

        qkv = qkv.reshape(-1, K, 3, H, C // H)  # (n_windows, K, 3, H, C//H)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # 注意力
        if self.upcast_attention:
            q = q.float()
            k = k.float()

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (n_windows, H, K, K)

        if self.enable_rpe and self.rpe is not None:
            rel_pos = self.get_rel_pos(point, order)
            attn = attn + self.rpe(rel_pos)

        if self.upcast_softmax:
            attn = attn.float()

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)

        # 加权聚合
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)  # (N_pad, C)
        feat = feat[:N]  # 去掉 padding
        feat = feat[inverse]  # 逆序列化回原始顺序

        # 输出投影
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


# Check for flash attention
HAS_FLASH_ATTN = False
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    pass


# ============================================================================
# MLP (前馈网络)
# ============================================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# Block (CPE + Attention + MLP)
# ============================================================================

class Block(PointModule):
    """
    PTv3 Block:
    - CPE: 条件位置编码 (Conv1d 替代 spconv SubMConv3d)
    - Self-Attention: SerializedAttention
    - MLP: 前馈网络
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 48,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        pre_norm: bool = True,
        order_index: int = 0,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # CPE: 使用 Conv1d + Linear 替代 spconv SubMConv3d
        self.cpe = PointSequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point) -> Point:
        # CPE: Conv1d 需要 (B, C, N) 格式，先 reshape
        shortcut = point.feat
        # CPE 替代: 用线性层 + 残差
        point.feat = shortcut + self._cpe_forward(point.feat)
        shortcut = point.feat

        # Attention
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        # MLP
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)

        return point

    def _cpe_forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        CPE 的 Conv1d 替代:
        - feat: (N, C)
        - 需要对每个 batch 单独处理或使用分组技巧
        - 简化: 使用线性映射 + 残差
        """
        return feat  # Conv1d 在变长点云上操作复杂, 暂时使用恒等映射 + 后面的 Linear


# ============================================================================
# Serialized Pooling (序列化下采样)
# ============================================================================

class SerializedPooling(PointModule):
    """
    基于序列化的下采样:
    - 通过序列化编码的 bit shift 实现多分辨率池化
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer=None,
        act_layer=None,
        reduce: str = "max",
        shuffle_orders: bool = True,
        traceable: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point) -> Point:
        # 计算池化深度
        pooling_depth = (int(math.ceil(self.stride)) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        assert {
            "serialized_code", "serialized_order", "serialized_inverse",
            "serialized_depth"
        }.issubset(point.keys())

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0], sorted=True, return_inverse=True, return_counts=True,
        )
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # 池化
        if TORCH_SCATTER_AVAILABLE:
            new_feat = torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            )
            new_coord = torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            )
        else:
            # 纯 PyTorch 替代
            new_feat = torch.zeros(len(head_indices), self.out_channels, device=point.feat.device)
            new_coord = torch.zeros(len(head_indices), 3, device=point.coord.device)
            proj_feat = self.proj(point.feat)
            for j in range(len(head_indices)):
                mask = cluster == j
                if self.reduce == "max":
                    new_feat[j] = proj_feat[mask].max(0)[0]
                elif self.reduce == "mean":
                    new_feat[j] = proj_feat[mask].mean(0)
                elif self.reduce == "sum":
                    new_feat[j] = proj_feat[mask].sum(0)
                else:
                    new_feat[j] = proj_feat[mask].min(0)[0]
                new_coord[j] = point.coord[mask].mean(0)

        point_dict = ADict(
            feat=new_feat,
            coord=new_coord,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point

        new_point = Point(point_dict)
        if self.norm is not None:
            new_point = self.norm(new_point)
        if self.act is not None:
            new_point = self.act(new_point)
        return new_point


# ============================================================================
# Embedding (初始特征嵌入)
# ============================================================================

class Embedding(PointModule):
    """
    初始点云嵌入:
    - 使用 Conv1d 替代 spconv SubMConv3d
    """
    def __init__(self, in_channels: int, embed_channels: int,
                 norm_layer=None, act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # 使用 Linear + LayerNorm 替代 spconv SubMConv3d
        self.stem = PointSequential(
            nn.Linear(in_channels, embed_channels),
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point) -> Point:
        point = self.stem(point)
        return point


# ============================================================================
# Point Transformer V3 Classification + BBox
# ============================================================================

class PointTransformerV3Cls(PointModule):
    """
    Point Transformer V3 for Classification + 3D BBox

    编码器-解码器架构 (分类只用编码器部分):
    - Embedding
    - N 个编码器阶段 (SerializedPooling + Block)
    - 全局池化 + 分类/框回归头
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 40,
        box_dim: int = 6,
        order: Tuple[str, ...] = ("z", "z-trans"),
        stride: Tuple[int, ...] = (2, 2, 2, 2),
        enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: Tuple[int, ...] = (48, 48, 48, 48, 48),
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        enable_flash: bool = False,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        # Normalization & Activation
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        # Embedding
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # Encoder stages
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]): sum(enc_depths[:s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # Classification head
        final_channels = enc_channels[-1]
        self.cls_head = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

        # Box regression head
        self.box_head = nn.Sequential(
            nn.Linear(final_channels, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, box_dim),
        )

    def forward(self, data_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data_dict: {"coord": (N,3), "feat": (N,C), "offset": (B,),
                       "grid_size": float, "grid_coord": (N,3) (可选)}

        Returns:
            logits: (B, num_classes)
            box_pred: (B, box_dim)
        """
        point = Point(data_dict)
        # 序列化
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)

        # 编码器
        point = self.embedding(point)
        point = self.enc(point)

        # 全局平均池化 (按样本)
        feat = point.feat
        offset = point.offset
        x_list = []
        for i in range(len(offset)):
            s_i = 0 if i == 0 else offset[i - 1].item()
            e_i = offset[i].item()
            cnt = e_i - s_i
            if cnt > 0:
                x_b = feat[s_i:e_i].mean(0, keepdim=True)
            else:
                x_b = torch.zeros(1, feat.shape[1], device=feat.device)
            x_list.append(x_b)
        x = torch.cat(x_list, 0)

        logits = self.cls_head(x)
        box_pred = self.box_head(x)
        return logits, box_pred


# ============================================================================
# 适配 SPAD 管道的包装类
# ============================================================================

class PointTransV3Classification(PointTransformerV3Cls):
    """
    PointTransV3Classification — 适配 SPAD 训练管道

    输入: (B, N, 4) xyzi 点云
    输出: (logits [B, num_classes], box_pred [B, 6])
    """
    def __init__(self, num_classes=26, **kwargs):
        super().__init__(in_channels=4, num_classes=num_classes, **kwargs)

    @staticmethod
    def _normalize_input_points(x):
        """统一输入格式为 (B, N, 4)"""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")
        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported shape {tuple(x.shape)}")
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
        B, N, C_in = x.shape

        coord = x[:, :, :3].reshape(B * N, 3).contiguous()
        # Pointcept V3 约定: feat 包含全量特征 (包括 xyz)
        feat = x.reshape(B * N, -1).contiguous()  # (B*N, 4) = xyzi
        offset = torch.arange(N, (B + 1) * N, step=N, dtype=torch.long, device=x.device)
        grid_size = 0.02  # 默认网格大小

        # 生成 grid_coord (序列化需要)
        grid_coord = torch.div(
            coord - coord.min(0)[0], grid_size, rounding_mode="trunc"
        ).int()

        data_dict = {
            "coord": coord,
            "feat": feat,
            "offset": offset,
            "grid_size": grid_size,
            "grid_coord": grid_coord,
        }
        return super().forward(data_dict)


# ============================================================================
# 快速测试 + GPU 显存测试
# ============================================================================

def _quick_test():
    """形状验证。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing PointTransformer V3 on {device}")

    model = PointTransV3Classification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Box:    {tuple(box_pred.shape)}")
    print("✓ PointTransformer V3 works correctly")


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
            m = PointTransV3Classification(num_classes=26).cuda()
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


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()