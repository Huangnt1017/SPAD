"""
3DETR 通用工具模块 (基于 facebookresearch/3detr 官方实现)。

对应原始仓库:
    - models/helpers.py       → GenericMLP, NORM_DICT, ACTIVATION_DICT, WEIGHT_INIT_DICT, BatchNormDim1Swap, get_clones
    - models/position_embedding.py → PositionEmbeddingCoordsSine
    - utils/pc_util.py        → shift_scale_points, scale_points
    - utils/box_ops3d.py / utils/box_util.py → SPAD box 编解码补充

组件:
    - GenericMLP / get_activation — 通用多层感知机 (官方 helpers.py)
    - PositionEmbeddingCoordsSine — 位置编码 (官方 position_embedding.py), 支持 sine/fourier
    - BatchNormDim1Swap — Transformer 专用 BN (官方 helpers.py)
    - NORM_DICT / ACTIVATION_DICT / WEIGHT_INIT_DICT — 配置字典
    - shift_scale_points / scale_points — 点云坐标缩放 (官方 pc_util.py)
    - SPAD Box 编解码 (自定义, 兼容 SPAD 轴对齐框格式)
"""

from __future__ import annotations

import copy
import math
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════
# 激活函数 & 归一化 & 初始化 配置字典
# ════════════════════════════════════════════════
# 对应 official/models/helpers.py

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    nn.Transformer 使用的 BatchNorm 包装器。
    输入形状 HW x N x C → permute to N x C x HW → BN → permute back.
    对应 official/models/helpers.py BatchNormDim1Swap.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super().forward(x)
        x = x.permute(2, 0, 1)
        return x


NORM_DICT: Dict[str, nn.Module] = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

WEIGHT_INIT_DICT: Dict[str, callable] = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


def get_activation(name: str) -> nn.Module:
    """获取激活函数实例。"""
    name = name.lower()
    if name in ACTIVATION_DICT:
        return ACTIVATION_DICT[name]()
    raise ValueError(f"Unknown activation: {name}")


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """克隆 N 个相同的模块。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ════════════════════════════════════════════════
# GenericMLP — 官方 3DETR helpers.py
# ════════════════════════════════════════════════

class GenericMLP(nn.Module):
    """通用 MLP，与 3DETR 官方 helpers.GenericMLP 接口一致。

    Args:
        input_dim: 输入维度。
        hidden_dims: 隐藏层维度列表。
        output_dim: 输出维度。
        norm_fn_name: 'bn'/'bn1d'/'ln'/'id' 归一化类型。
        activation: 激活函数名。
        use_conv: 是否使用 Conv1d 代替 Linear。
        dropout: Dropout 概率。
        hidden_use_bias: 隐藏层是否使用 bias。
        output_use_bias: 输出层是否使用 bias。
        output_use_activation: 输出后是否加激活。
        output_use_norm: 输出后是否加归一化。
        weight_init_name: 权重初始化方式，如 'xavier_uniform'。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        norm_fn_name: Optional[str] = None,
        activation: str = "relu",
        use_conv: bool = False,
        dropout: Optional[float] = None,
        hidden_use_bias: bool = False,
        output_use_bias: bool = True,
        output_use_activation: bool = False,
        output_use_norm: bool = False,
        weight_init_name: Optional[str] = None,
    ):
        super().__init__()
        act_fn = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # LayerNorm via GroupNorm(1, x)

        if dropout is not None and not isinstance(dropout, list):
            dropout = [dropout] * len(hidden_dims)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for idx, hdim in enumerate(hidden_dims):
            if use_conv:
                layers.append(nn.Conv1d(prev_dim, hdim, 1, bias=hidden_use_bias))
            else:
                layers.append(nn.Linear(prev_dim, hdim, bias=hidden_use_bias))
            if norm:
                layers.append(norm(hdim))
            layers.append(act_fn())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = hdim

        if use_conv:
            layers.append(nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias))
        else:
            layers.append(nn.Linear(prev_dim, output_dim, bias=output_use_bias))

        if output_use_norm and norm:
            layers.append(norm(output_dim))
        if output_use_activation:
            layers.append(act_fn())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name: str):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for _, param in self.named_parameters():
            if param.dim() > 1:
                func(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ════════════════════════════════════════════════
# 点云坐标缩放 — 对应 official/utils/pc_util.py
# ════════════════════════════════════════════════

def shift_scale_points(
    pred_xyz: torch.Tensor,
    src_range: List[torch.Tensor],
    dst_range: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    将点云从 src_range 线性映射到 dst_range。
    对应 official/utils/pc_util.py shift_scale_points.

    Args:
        pred_xyz: (B, N, 3) 或 (B, N_q, N, 3) 坐标。
        src_range: [min, max], 每个为 (B, 3) 或 broadcastable.
        dst_range: [min, max], 默认 [0, 1].

    Returns:
        映射后的坐标, 形状同 pred_xyz.
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


def scale_points(pred_xyz: torch.Tensor, mult_factor: torch.Tensor) -> torch.Tensor:
    """
    缩放点云坐标。
    对应 official/utils/pc_util.py scale_points.
    """
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    return pred_xyz * mult_factor[:, None, :]


# ════════════════════════════════════════════════
# 位置编码 — 官方 3DETR position_embedding.py
# ════════════════════════════════════════════════

class PositionEmbeddingCoordsSine(nn.Module):
    """3D 位置编码, 支持 sine / fourier 两种模式。

    对应 official/models/position_embedding.py PositionEmbeddingCoordsSine.

    Args:
        d_pos: 输出维度。
        pos_type: 'sine' 或 'fourier'.
        temperature: 正弦频率衰减温度。
        normalize: 是否归一化输入坐标。
        scale: 缩放系数 (仅 sine 模式)。
        gauss_scale: 高斯矩阵缩放 (仅 fourier 模式)。
    """

    def __init__(
        self,
        d_pos: int = 256,
        pos_type: str = "fourier",
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        gauss_scale: float = 1.0,
    ):
        super().__init__()
        assert pos_type in ["sine", "fourier"], f"Unknown pos_type: {pos_type}"
        self.pos_type = pos_type
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        if pos_type == "fourier":
            d_in = 3  # xyz
            assert d_pos is not None and d_pos % 2 == 0
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)

        self.d_pos = d_pos

    def get_sine_embeddings(
        self,
        xyz: torch.Tensor,
        num_channels: Optional[int] = None,
        input_range: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """正弦位置编码。

        Args:
            xyz: (B, N, 3) 坐标。
            num_channels: 输出通道数, 默认 self.d_pos.
            input_range: [min, max] 归一化范围。

        Returns:
            (B, num_channels, N) 位置编码。
        """
        if num_channels is None:
            num_channels = self.d_pos

        orig_xyz = xyz
        xyz = orig_xyz.clone()

        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ncoords = xyz.shape[1]
        ndim = num_channels // xyz.shape[2]  # 每轴分配维度
        if ndim % 2 != 0:
            ndim -= 1
        rems = num_channels - (ndim * xyz.shape[2])

        assert ndim % 2 == 0, f"Cannot handle odd ndim={ndim}"

        final_embeds = []
        prev_dim = 0
        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                cdim += 2
                rems -= 2
            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(
        self,
        xyz: torch.Tensor,
        num_channels: Optional[int] = None,
        input_range: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """傅里叶位置编码 (NeRF 风格)。

        Args:
            xyz: (B, N, 3) 坐标。
            num_channels: 输出通道数, 默认 self.d_pos.
            input_range: [min, max] 归一化范围。

        Returns:
            (B, num_channels, N) 位置编码。
        """
        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        orig_xyz = xyz
        xyz = orig_xyz.clone()

        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(
        self,
        xyz: torch.Tensor,
        num_channels: Optional[int] = None,
        input_range: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) 坐标。
            num_channels: 输出通道数。
            input_range: [min, max] 归一化范围。

        Returns:
            (B, C, N) 位置编码, C = num_channels 或 self.d_pos.
        """
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3

        with torch.no_grad():
            if self.pos_type == "sine":
                return self.get_sine_embeddings(xyz, num_channels, input_range)
            elif self.pos_type == "fourier":
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
            else:
                raise ValueError(f"Unknown pos_type: {self.pos_type}")

    def extra_repr(self) -> str:
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
        return st


# ════════════════════════════════════════════════
# 3D Box 编解码 (SPAD 自定义)
# ════════════════════════════════════════════════
# 以下为 SPAD 项目特有的轴对齐框编解码工具,
# 与官方 BoxProcessor (center/size/angle) 不同。

SPAD_BOUNDS = (1.0, 64.0, 1.0, 64.0, 1.0, 110.0)


def encode_box_abs_to_norm(
    boxes: torch.Tensor,
    bounds: Tuple[float, ...] = SPAD_BOUNDS,
) -> torch.Tensor:
    """绝对坐标框 → [0,1] 归一化。boxes: (...,6) [xmin,xmax,ymin,ymax,zmin,zmax]."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    scale = torch.tensor(
        [xmax - xmin] * 2 + [ymax - ymin] * 2 + [zmax - zmin] * 2,
        device=boxes.device, dtype=boxes.dtype,
    )
    offset = torch.tensor(
        [xmin, xmin, ymin, ymin, zmin, zmin],
        device=boxes.device, dtype=boxes.dtype,
    )
    return (boxes - offset) / scale.clamp(min=1e-6)


def decode_box_norm_to_abs(
    norm: torch.Tensor,
    bounds: Tuple[float, ...] = SPAD_BOUNDS,
) -> torch.Tensor:
    """归一化框 → 绝对坐标。"""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    scale = torch.tensor(
        [xmax - xmin] * 2 + [ymax - ymin] * 2 + [zmax - zmin] * 2,
        device=norm.device, dtype=norm.dtype,
    )
    offset = torch.tensor(
        [xmin, xmin, ymin, ymin, zmin, zmin],
        device=norm.device, dtype=norm.dtype,
    )
    return norm * scale + offset


def canonicalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """确保 min ≤ max。"""
    a = boxes[..., 0::2]
    b = boxes[..., 1::2]
    mins = torch.minimum(a, b)
    maxs = torch.maximum(a, b)
    return torch.cat([mins[..., :1], maxs[..., :1],
                      mins[..., 1:2], maxs[..., 1:2],
                      mins[..., 2:3], maxs[..., 2:3]], dim=-1)
