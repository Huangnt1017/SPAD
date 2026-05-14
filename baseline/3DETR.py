"""
3DETR — 3D Detection Transformer

GitHub:  https://github.com/facebookresearch/3detr
Local:   D:\essay\3d目标检测复现仓库\3detr (参考)

基于 facebookresearch/3detr 官方实现, 适配 SPAD 单光子点云:
    - PointNet++ pre-encoder (纯 Python 实现, 无需 CUDA 编译)
    - Sine/Fourier 3D 位置编码 (PositionEmbeddingCoordsSine)
    - Transformer Encoder: vanilla self-attention
    - FPS-based query selection: 从编码点云采样 queries
    - Transformer Decoder: self-attn + cross-attn → per-query box features
    - MLP heads: 官方 5 头 (sem_cls, center, size, angle_cls, angle_residual)
    - BoxProcessor: 将 MLP 输出解码为 3D 边界框

输出: dict {'logits': (B, num_classes), 'box_pred': (B, 6)} 兼容 utils/loss.py.

类名/函数名与官方一致:
    - models/model_3detr.py     → Model3DETR, BoxProcessor, build_3detr
    - models/transformer.py     → TransformerEncoder/Layer, TransformerDecoder/Layer
    - models/position_embedding.py → PositionEmbeddingCoordsSine
    - models/helpers.py         → GenericMLP, NORM_DICT, get_clones
    - third_party/pointnet2/    → PointnetSAModuleVotes (纯 Python 版)

@inproceedings{misra2021end,
  title={An end-to-end transformer model for 3d object detection},
  author={Misra, Ishan and Girdhar, Rohit and Joulin, Armand},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={2906--2917},
  year={2021}
}
"""

from __future__ import annotations

import math
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.detr3_util import (
    ACTIVATION_DICT,
    GenericMLP,
    NORM_DICT,
    PositionEmbeddingCoordsSine,
    WEIGHT_INIT_DICT,
    canonicalize_boxes,
    decode_box_norm_to_abs,
    encode_box_abs_to_norm,
    get_activation,
    get_clones,
    shift_scale_points,
    scale_points,
)


# ═══════════════════════════════════════════════════════
# PointNet++ 工具函数 (纯 Python, 无需 CUDA 编译)
# ═══════════════════════════════════════════════════════

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """计算两组点之间的平方欧氏距离。"""
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """根据索引从点集中收集点。"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=points.device)
        .view(view_shape).repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """最远点采样 (FPS), 返回索引。"""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """球查询, 返回分组索引。"""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = (torch.arange(N, dtype=torch.long, device=xyz.device)
                 .view(1, 1, N).repeat(B, S, 1))
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx[group_idx == N] = group_first[group_idx == N]
    return group_idx


def sample_and_group(
    npoint: int, radius: float, nsample: int,
    xyz: torch.Tensor, points: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FPS + 球查询 + 分组 + 坐标归一化。"""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


# ═══════════════════════════════════════════════════════
# PointnetSAModuleVotes — 对应 official/third_party/pointnet2/pointnet2_modules.py
# ═══════════════════════════════════════════════════════
# 纯 Python 实现, 无需 CUDA 编译.

class PointnetSAModuleVotes(nn.Module):
    """
    PointNet++ Set Abstraction 模块, 纯 Python 版.
    对应官方 PointnetSAModuleVotes (pointnet2_modules.py).

    支持 max / avg / rbf 三种池化方式.
    输出 (new_xyz, new_features, inds) 三元组.
    """

    def __init__(
        self,
        *,
        mlp: List[int],
        npoint: Optional[int] = None,
        radius: float = 0.2,
        nsample: int = 64,
        bn: bool = True,
        use_xyz: bool = True,
        pooling: str = "max",
        sigma: Optional[float] = None,
        normalize_xyz: bool = False,
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.sigma = sigma if sigma is not None else radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        # MLP spec: if use_xyz, first channel += 3
        mlp_spec = mlp.copy()
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3

        # 构建 SharedMLP (Conv2d + BN + ReLU)
        layers = []
        in_ch = mlp_spec[0]
        for out_ch in mlp_spec[1:]:
            layers.append(nn.Conv2d(in_ch, out_ch, 1))
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.mlp_module = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        inds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) 坐标.
            features: (B, C, N) 特征或 None.
            inds: (B, npoint) 预定义 FPS 索引或 None.

        Returns:
            new_xyz: (B, npoint, 3) 降采样坐标.
            new_features: (B, mlp[-1], npoint) 特征.
            inds: (B, npoint) FPS 索引.
        """
        B, N, _ = xyz.shape

        # FPS
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # (B, 3, N)
        if inds is None:
            inds = farthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = index_points(xyz, inds)  # (B, npoint, 3)

        # 球查询分组
        # grouped_xyz: (B, npoint, nsample, 3), grouped_features: (B, npoint, nsample, C)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]  # 相对坐标

        if self.normalize_xyz:
            grouped_xyz_norm = grouped_xyz_norm / self.radius

        if features is not None:
            features_t = features.transpose(1, 2).contiguous()  # (B, N, C)
            grouped_features = index_points(features_t, idx)  # (B, npoint, nsample, C)
            if self.use_xyz:
                new_points = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)  # (B, npoint, nsample, 3+C)
            else:
                new_points = grouped_features
        else:
            new_points = grouped_xyz_norm  # (B, npoint, nsample, 3)

        # MLP: (B, npoint, nsample, in_ch) → (B, in_ch, npoint, nsample) → Conv2d → (B, out_ch, npoint, nsample)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        new_features = self.mlp_module(new_points)  # (B, out_ch, npoint, nsample)

        # 池化
        if self.pooling == "max":
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == "avg":
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == "rbf":
            rbf = torch.exp(-1 * grouped_xyz_norm.pow(2).sum(-1) / (self.sigma ** 2) / 2)
            rbf = rbf.unsqueeze(1)  # (B, 1, npoint, nsample)
            new_features = torch.sum(new_features * rbf, -1, keepdim=True) / float(self.nsample)
        new_features = new_features.squeeze(-1)  # (B, out_ch, npoint)

        return new_xyz, new_features, inds


# ═══════════════════════════════════════════════════════
# Transformer 模块 — 对应 official/models/transformer.py
# ═══════════════════════════════════════════════════════

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 单层 (pre-norm/post-norm).
    对应官方 TransformerEncoderLayer.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        dropout_attn: Optional[float] = None,
        activation: str = "relu",
        normalize_before: bool = True,
        norm_name: str = "ln",
        use_ffn: bool = True,
        ffn_use_bias: bool = True,
    ):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout)

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.use_ffn:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(
            q, k, value=src2, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    """
    堆叠多个 TransformerEncoderLayer.
    对应官方 TransformerEncoder (transformer.py).
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        weight_init_name: str = "xavier_uniform",
    ):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name: str):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        xyz: Optional[torch.Tensor] = None,
        transpose_swap: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            (xyz, output, xyz_inds).
            注: 纯 Encoder (非 Masked) 返回 xyz=None, xyz_inds=None.
        """
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = src
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            cur_mask = None
            if orig_mask is not None:
                cur_mask = orig_mask[idx]
                bsz, n, _ = cur_mask.shape
                nhead = layer.nhead
                cur_mask = cur_mask.unsqueeze(1).repeat(1, nhead, 1, 1)
                cur_mask = cur_mask.view(bsz * nhead, n, n)
            output = layer(output, src_mask=cur_mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return None, output, None  # xyz=None, output, xyz_inds=None


class MaskedTransformerEncoder(TransformerEncoder):
    """
    Masked Transformer Encoder — 带半径掩码 + 中间降采样.
    对应官方 MaskedTransformerEncoder (transformer.py).
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        masking_radius: List[float],
        interim_downsampling: nn.Module,
        norm: Optional[nn.Module] = None,
        weight_init_name: str = "xavier_uniform",
    ):
        super().__init__(encoder_layer, num_layers, norm=norm, weight_init_name=weight_init_name)
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling

    def compute_mask(
        self, xyz: torch.Tensor, radius: float, dist: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            mask = dist >= radius  # True → 不参与注意力
        return mask, dist

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        xyz: Optional[torch.Tensor] = None,
        transpose_swap: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None

        for idx, layer in enumerate(self.layers):
            cur_mask = None
            if self.masking_radius[idx] > 0:
                cur_mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)
                bsz, n, _ = cur_mask.shape
                nhead = layer.nhead
                cur_mask = cur_mask.unsqueeze(1).repeat(1, nhead, 1, 1)
                cur_mask = cur_mask.view(bsz * nhead, n, n)

            output = layer(output, src_mask=cur_mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

            if idx == 0 and self.interim_downsampling:
                output = output.permute(1, 2, 0)  # (B, C, N)
                xyz, output, xyz_inds = self.interim_downsampling(xyz, output)
                output = output.permute(2, 0, 1)  # (N, B, C)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return xyz, output, xyz_inds


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder 单层 (pre-norm, 含 self-attn + cross-attn + FFN).
    对应官方 TransformerDecoderLayer (transformer.py).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        dropout_attn: Optional[float] = None,
        activation: str = "relu",
        normalize_before: bool = True,
        norm_fn_name: str = "ln",
    ):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)
        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos, return_attn_weights)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos, return_attn_weights)


class TransformerDecoder(nn.Module):
    """
    堆叠多个 TransformerDecoderLayer, 可选返回中间层.
    对应官方 TransformerDecoder (transformer.py).
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm_fn_name: Optional[str] = "ln",
        return_intermediate: bool = False,
        weight_init_name: str = "xavier_uniform",
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](decoder_layer.linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name: str):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        transpose_swap: bool = False,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = tgt
        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(
                output, memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos,
                return_attn_weights=return_attn_weights,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns) if attns else None

        if self.return_intermediate:
            return torch.stack(intermediate), attns
        return output, attns


# ═══════════════════════════════════════════════════════
# BoxProcessor — 对应 official/models/model_3detr.py
# ═══════════════════════════════════════════════════════

class BoxProcessor(nn.Module):
    """
    将 MLP heads 输出解码为 3D 边界框.
    对应官方 BoxProcessor (model_3detr.py).

    处理:
        - center_offset → center (unnormalized + normalized)
        - size_normalized → size_unnormalized (基于场景尺度)
        - angle_logits + angle_residual → angle_continuous
        - cls_logits → sem_cls_prob + objectness_prob
    """

    def __init__(self, dataset_config=None):
        super().__init__()
        self.dataset_config = dataset_config

    def compute_predicted_center(
        self, center_offset: torch.Tensor, query_xyz: torch.Tensor,
        point_cloud_dims: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算预测中心: offset + query_xyz → 归一化+非归一化."""
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(
        self, size_normalized: torch.Tensor,
        point_cloud_dims: List[torch.Tensor],
    ) -> torch.Tensor:
        """计算预测尺寸: 归一化 size → 绝对尺寸."""
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(
        self, angle_logits: torch.Tensor, angle_residual: torch.Tensor,
    ) -> torch.Tensor:
        """计算预测角度 (连续值)."""
        if angle_logits.shape[-1] == 1:
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(
        self, cls_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 objectness 概率和语义类别概率."""
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = F.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def forward(self, *args, **kwargs):
        raise NotImplementedError("BoxProcessor is used as a helper, call its methods directly.")


# ═══════════════════════════════════════════════════════
# Model3DETR — 对应 official/models/model_3detr.py
# ═══════════════════════════════════════════════════════

class Model3DETR(nn.Module):
    """
    3DETR 主模型.
    对应官方 Model3DETR.

    子模块:
        - pre_encoder: 点云降采样 + 特征投影 (PointNet++ SA)
        - encoder: Transformer 自注意力编码器
        - decoder: Transformer 交叉注意力解码器
        - mlp_heads: 五头 (sem_cls, center, size, angle_cls, angle_residual)
        - box_processor: 将 MLP 输出解码为 3D 框
        - pos_embedding: Sine/Fourier 位置编码

    输入: dict {
        'point_clouds': (B, N, 3+) 点云,
        'point_cloud_dims_min': (B, 3) 点云最小坐标,
        'point_cloud_dims_max': (B, 3) 点云最大坐标,
    }

    兼容 SPAD:
        - forward 也接受 (B, N, 4) 张量 (SPAD 原始点云)
        - 输出兼容 split_cls_and_box_predictions (keys: 'logits', 'box_pred')
    """

    def __init__(
        self,
        pre_encoder: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        dataset_config=None,
        encoder_dim: int = 256,
        decoder_dim: int = 256,
        position_embedding: str = "fourier",
        mlp_dropout: float = 0.3,
        num_queries: int = 256,
        num_classes: int = 26,  # SPAD: for compatibility with loss.py
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder

        # Encoder → Decoder 投影
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        # 位置编码
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )

        # Query 投影
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        self.decoder = decoder
        self.num_classes = num_classes
        self.build_mlp_heads(decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_mlp_heads(self, decoder_dim: int, mlp_dropout: float):
        """构建 MLP 预测头 (官方 5 头)."""
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # 语义类别 (含 background 类)
        semcls_head = mlp_func(output_dim=self.num_classes)

        # 几何: center, size, angle
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=1)  # SPAD: 无朝向, 简化为1 bin
        angle_reg_head = mlp_func(output_dim=1)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """将点云拆分为 xyz 和 features."""
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        运行预编码器 + Transformer Encoder.

        Returns:
            enc_xyz: (S, B, 3) 编码坐标.
            enc_features: (S, B, C) 编码特征.
            enc_inds: (B, S) 采样索引.
        """
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: (B, N, 3), features: (B, C, N), pre_enc_xyz: (B, npoint, 3), pre_enc_features: (B, C, npoint)

        # nn.MultiHeadAttention 期望 (N, B, C)
        pre_enc_features = pre_enc_features.permute(2, 0, 1)  # (npoint, B, C)

        enc_xyz_out, enc_features, enc_inds_out = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        # Vanilla encoder 返回 xyz=None, xyz_inds=None → 使用 pre_encoder 的输出
        enc_xyz = enc_xyz_out if enc_xyz_out is not None else pre_enc_xyz
        enc_inds = enc_inds_out if enc_inds_out is not None else pre_enc_inds

        return enc_xyz, enc_features, enc_inds

    def get_query_embeddings(
        self, encoder_xyz: torch.Tensor, point_cloud_dims: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从编码点云中采样 queries (FPS).
        对应官方 Model3DETR.get_query_embeddings.

        Notes:
            encoder_xyz 来自 run_encoder, 形状为 (B, S, 3) 或 (S, B, 3).
            对于 vanilla encoder, 直接返回 pre_enc_xyz (B, S, 3).

        Returns:
            query_xyz: (B, n_q, 3) query 坐标.
            query_embed: (B, C, n_q) query 特征嵌入.
        """
        # Official: encoder_xyz is (B, N, 3)
        xyz_for_fps = encoder_xyz
        # Detect (S, B, 3) format: sequence dim > batch dim
        if xyz_for_fps.dim() == 3 and xyz_for_fps.shape[0] > xyz_for_fps.shape[1]:
            xyz_for_fps = xyz_for_fps.permute(1, 0, 2).contiguous()

        query_inds = farthest_point_sample(xyz_for_fps, self.num_queries)
        query_inds = query_inds.long()

        # gather query xyz (official style)
        query_xyz = torch.stack([
            torch.gather(xyz_for_fps[..., x], 1, query_inds) for x in range(3)
        ], dim=-1)  # (B, n_q, 3)

        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)  # (B, C, n_q)
        query_embed = self.query_projection(pos_embed)  # (B, C, n_q)
        return query_xyz, query_embed

    def get_box_predictions(
        self, query_xyz: torch.Tensor, point_cloud_dims: List[torch.Tensor],
        box_features: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        从 decoder 输出计算 box 预测.

        Args:
            query_xyz: (B, n_q, 3).
            point_cloud_dims: [min, max].
            box_features: (num_layers, n_q, B, C).

        Returns:
            dict with 'outputs' (last layer) and 'aux_outputs' (intermediate).
        """
        # (num_layers, n_q, B, C) → (num_layers, B, C, n_q)
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = box_features.shape
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # MLP heads: (num_layers * B, noutput, n_q)
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        size_normalized = self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](box_features).transpose(1, 2)

        # Reshape: (num_layers, B, n_q, noutput)
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_residual = angle_residual_normalized * (np.pi / max(angle_residual_normalized.shape[-1], 1))

        outputs = []
        for l in range(num_layers):
            center_normalized, center_unnormalized = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )

            # 对于 SPAD: 没有朝向, 直接使用 center + size 构造轴对齐框
            # box_corners 在 SPAD 中暂时不使用 (分类任务)
            with torch.no_grad():
                sem_cls_prob = F.softmax(cls_logits[l], dim=-1)[..., :-1] if cls_logits[l].shape[-1] > 1 else F.softmax(cls_logits[l], dim=-1)
                objectness_prob = 1 - F.softmax(cls_logits[l], dim=-1)[..., -1] if cls_logits[l].shape[-1] > 1 else torch.ones_like(cls_logits[l][..., 0])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": sem_cls_prob,
            }
            outputs.append(box_prediction)

        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,
            "aux_outputs": aux_outputs,
        }

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], encoder_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播.

        Args:
            inputs: (B, N, 4) SPAD 点云 或 dict 包含 'point_clouds'.
            encoder_only: 是否仅运行 encoder (用于可视化).

        Returns:
            SPAD 兼容 dict: {'logits': (B, num_classes), 'box_pred': (B, 6)}.
        """
        # 兼容 SPAD: 直接传入 (B, N, 4) 张量
        if torch.is_tensor(inputs):
            point_clouds = inputs
            # 自动推断点云范围
            xyz = point_clouds[..., :3]
            batch_size = point_clouds.shape[0]
            point_cloud_dims_min = xyz.reshape(batch_size, -1, 3).min(dim=1)[0]
            point_cloud_dims_max = xyz.reshape(batch_size, -1, 3).max(dim=1)[0]
            inputs = {
                "point_clouds": point_clouds,
                "point_cloud_dims_min": point_cloud_dims_min,
                "point_cloud_dims_max": point_cloud_dims_max,
            }
        else:
            point_clouds = inputs["point_clouds"]

        # Encoder
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)

        # Encoder → Decoder 投影
        # enc_features: (S, B, C) → (B, C, S) → Conv1d → (B, C', S) → (S, B, C')
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)

        if encoder_only:
            return {
                "logits": enc_features.transpose(0, 1),  # (B, S, C')
                "box_pred": None,
            }

        # 点云范围
        point_cloud_dims = [inputs["point_cloud_dims_min"], inputs["point_cloud_dims_max"]]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)

        # 位置编码: enc_xyz 是 (B, S, 3)
        enc_pos = self.pos_embedding(
            enc_xyz, input_range=point_cloud_dims
        )  # (B, C, S)

        # Decoder 期望格式: (S, B, C)
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)  # (n_q, B, C)
        tgt = torch.zeros_like(query_embed)

        # Decoder: 返回 (stack(intermediate), attns)
        box_features, _ = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )
        # box_features: (num_layers, n_q, B, C) or (n_q, B, C)

        # 如果 return_intermediate=True, box_features 是 stack
        if isinstance(box_features, torch.Tensor) and box_features.dim() == 4:
            box_predictions = self.get_box_predictions(query_xyz, point_cloud_dims, box_features)
        else:
            # 单层输出
            box_features_stacked = box_features.unsqueeze(0)
            box_predictions = self.get_box_predictions(query_xyz, point_cloud_dims, box_features_stacked)

        # ─── SPAD 兼容输出 ───
        # 从 box_predictions 中提取 logits 和 box_pred
        final_out = box_predictions["outputs"]
        sem_cls_logits = final_out["sem_cls_logits"]  # (B, n_q, num_classes)

        # 对 queries 做 max-pooling → (B, num_classes)
        logits = sem_cls_logits.max(dim=1)[0]

        # 构建 SPAD 轴对齐框: center + size → [xmin, xmax, ymin, ymax, zmin, zmax]
        center = final_out["center_unnormalized"]  # (B, n_q, 3)
        size = final_out["size_unnormalized"]  # (B, n_q, 3)
        half = size / 2.0
        xmin = (center[..., 0] - half[..., 0]).unsqueeze(-1)
        xmax = (center[..., 0] + half[..., 0]).unsqueeze(-1)
        ymin = (center[..., 1] - half[..., 1]).unsqueeze(-1)
        ymax = (center[..., 1] + half[..., 1]).unsqueeze(-1)
        zmin = (center[..., 2] - half[..., 2]).unsqueeze(-1)
        zmax = (center[..., 2] + half[..., 2]).unsqueeze(-1)
        box_spad = torch.cat([xmin, xmax, ymin, ymax, zmin, zmax], dim=-1)  # (B, n_q, 6)
        # Aggregate queries → (B, 6)
        box_pred = box_spad.max(dim=1)[0]

        return {
            "logits": logits,
            "box_pred": box_pred,
            # 保留完整输出以供训练
            "_box_predictions": box_predictions,
            "_enc_xyz": enc_xyz,
            "_enc_features": enc_features,
        }


# ═══════════════════════════════════════════════════════
# 工厂函数 — 对应 official/models/model_3detr.py build_*
# ═══════════════════════════════════════════════════════

def build_preencoder(
    preenc_npoints: int = 512,
    enc_dim: int = 256,
    num_extra_channels: int = 1,
) -> PointnetSAModuleVotes:
    """构建 Pre-encoder (PointNet++ SA).

    Args:
        preenc_npoints: FPS 输出点数.
        enc_dim: 输出特征维度.
        num_extra_channels: 除 xyz 外的额外特征通道数 (SPAD: 1 个强度通道).
    """
    # mlp_dims[0] = num_extra_channels, 之后 use_xyz 会在 PointnetSAModuleVotes 中
    # 自动加 3 (xyz 坐标), 因此 MLP 总输入通道 = 3 + num_extra_channels
    mlp_dims = [num_extra_channels, 64, 128, enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(
    enc_dim: int = 256,
    enc_nhead: int = 4,
    enc_ffn_dim: int = 128,
    enc_dropout: float = 0.1,
    enc_activation: str = "relu",
    enc_nlayers: int = 3,
    enc_type: str = "vanilla",
    preenc_npoints: int = 512,
) -> nn.Module:
    """构建 Transformer Encoder."""
    encoder_layer = TransformerEncoderLayer(
        d_model=enc_dim,
        nhead=enc_nhead,
        dim_feedforward=enc_ffn_dim,
        dropout=enc_dropout,
        activation=enc_activation,
    )

    if enc_type == "vanilla":
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=enc_nlayers
        )
    elif enc_type == "masked":
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4, nsample=32, npoint=preenc_npoints // 2,
            mlp=[enc_dim, 256, 256, enc_dim], normalize_xyz=True,
        )
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {enc_type}")

    return encoder


def build_decoder(
    dec_dim: int = 256,
    dec_nhead: int = 4,
    dec_ffn_dim: int = 256,
    dec_dropout: float = 0.1,
    dec_nlayers: int = 3,
) -> TransformerDecoder:
    """构建 Transformer Decoder."""
    decoder_layer = TransformerDecoderLayer(
        d_model=dec_dim,
        nhead=dec_nhead,
        dim_feedforward=dec_ffn_dim,
        dropout=dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=dec_nlayers, return_intermediate=True
    )
    return decoder


def build_3detr(
    preenc_npoints: int = 512,
    enc_dim: int = 256,
    dec_dim: int = 256,
    enc_nhead: int = 4,
    dec_nhead: int = 4,
    enc_ffn_dim: int = 128,
    dec_ffn_dim: int = 256,
    enc_dropout: float = 0.1,
    dec_dropout: float = 0.1,
    enc_activation: str = "relu",
    enc_nlayers: int = 3,
    dec_nlayers: int = 3,
    enc_type: str = "vanilla",
    num_queries: int = 256,
    mlp_dropout: float = 0.3,
    position_embedding: str = "fourier",
    num_classes: int = 26,
    dataset_config=None,
) -> Tuple[Model3DETR, BoxProcessor]:
    """
    完整构建 3DETR 模型 + BoxProcessor.

    这是 SPAD 版本的 build_3detr, 参数直接传入而非通过 args 对象.
    """
    pre_encoder = build_preencoder(preenc_npoints, enc_dim)
    encoder = build_encoder(enc_dim, enc_nhead, enc_ffn_dim, enc_dropout,
                            enc_activation, enc_nlayers, enc_type, preenc_npoints)
    decoder = build_decoder(dec_dim, dec_nhead, dec_ffn_dim, dec_dropout, dec_nlayers)

    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config=dataset_config,
        encoder_dim=enc_dim,
        decoder_dim=dec_dim,
        position_embedding=position_embedding,
        mlp_dropout=mlp_dropout,
        num_queries=num_queries,
        num_classes=num_classes,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor


# ═══════════════════════════════════════════════════════
# 兼容导入: 保留旧类名作为别名
# ═══════════════════════════════════════════════════════
ThreeDETRClassification = Model3DETR
PointNetPreEncoder = PointnetSAModuleVotes


# ═══════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, gc
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 3DETR 验证 (device={device}) ===\n")

    # 测试 1: build_3detr 工厂函数
    print("--- test: build_3detr ---")
    model, processor = build_3detr(
        preenc_npoints=512, enc_dim=256, dec_dim=256,
        num_queries=64, num_classes=26,
        position_embedding="sine",
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f} M")

    # 测试 2: 前向传播 (SPAD 张量输入)
    print("\n--- test: forward with (B, N, 4) tensor ---")
    pts = torch.randn(2, 1024, 4, device=device)
    model.eval()
    with torch.no_grad():
        out = model(pts)

    print(f"输入:   {pts.shape}")
    print(f"logits: {out['logits'].shape}")
    print(f"box:    {out['box_pred'].shape}")
    assert out["logits"].shape == (2, 26), f"Expected (2, 26), got {out['logits'].shape}"
    assert out["box_pred"].shape == (2, 6), f"Expected (2, 6), got {out['box_pred'].shape}"

    # 测试 3: 兼容 split_cls_and_box_predictions
    print("\n--- test: utils.loss compatibility ---")
    from utils.loss import split_cls_and_box_predictions
    logits, box_preds = split_cls_and_box_predictions(out)
    assert logits is not None
    assert box_preds is not None
    print(f"split_cls_and_box_predictions: logits {logits.shape}, box_preds {box_preds.shape}")

    # 测试 4: dict 输入
    print("\n--- test: forward with dict input ---")
    with torch.no_grad():
        out2 = model({
            "point_clouds": pts,
            "point_cloud_dims_min": pts[..., :3].reshape(2, -1, 3).min(dim=1)[0],
            "point_cloud_dims_max": pts[..., :3].reshape(2, -1, 3).max(dim=1)[0],
        })
    assert out2["logits"].shape == (2, 26)
    print(f"dict input OK: logits {out2['logits'].shape}")

    # 测试 5: 导入检查
    print("\n--- test: imports ---")
    from utils.detr3_util import (
        PositionEmbeddingCoordsSine,
        NORM_DICT, WEIGHT_INIT_DICT,
        shift_scale_points, scale_points,
    )
    print("All imports OK")

    print("\n✅ 全部通过!")

    # ══════════════════════════════════════════════
    # GPU 显存测试
    # ══════════════════════════════════════════════
    print("\n=== GPU 显存测试 ===")
    if torch.cuda.is_available():
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
                m, _ = build_3detr(
                    preenc_npoints=512, enc_dim=256, dec_dim=256,
                    num_queries=64, num_classes=26, position_embedding="sine",
                )
                m = m.cuda()
                pts = torch.randn(bs, N, 4).cuda()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                m.train()
                o = m(pts)
                loss = o['logits'].sum() + o['box_pred'].sum()
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
    else:
        print("无 CUDA，跳过。")
