"""
Point Transformer V1 (Zhao et al., ICCV 2021) - 严格对齐 Pointcept 复现。

参考:
- D:\\essay\\3d目标检测复现仓库\\Pointcept-main\\Pointcept-main\\pointcept\\models\\point_transformer\\
    point_transformer_cls.py / point_transformer_seg.py / utils.py

复现说明:
- `PointTransformerLayer / TransitionDown / Bottleneck` 与 Pointcept 一一对应,
  仅把 `pointops.knn_query_and_group / farthest_point_sampling` 替换为本仓库 `utils/pointnet_utils.py`
  中的纯 PyTorch 等价实现 (variable-length 即 offset 约定)。
- `PointTransformerCls` 的网络骨架严格保持 [planes=32/64/128/256/512, stride=1/4/4/4/4,
  nsample=8/16/16/16/16, share_planes=8]; 只把官方的"单分类头"改为 SPAD 双头
  (logits + 3D 中心点)。
- `PointTransformerClassification` 是 SPAD 适配层: 接受 (B, N, 4) xyzi 输入,
  转成 Pointcept 的 `data_dict = {coord, feat, offset}` 调用主体网络。

数据契约 (与项目其他 baseline 一致):
- 输入: (B, N, 4) xyzi 点云 (xyz 已归一化到 [0, 1])。
- 输出: tuple (logits [B, num_classes], center_pred [B, 3])。

Reference:
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={16259--16268},
  year={2021}
}
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn

# 项目根目录入 sys.path 后再 import utils, 避免脚本式调用时找不到 utils。
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import einops

from utils.pointnet_utils import (
    LayerNorm1d,
    farthest_point_sample_varlen,
    knn_point_varlen,
)


# ============================================================================
# pointops.knn_query_and_group 的纯 PyTorch 等价实现
# ============================================================================

def _knn_query_and_group(
    feat: torch.Tensor,
    coord: torch.Tensor,
    offset: torch.Tensor,
    new_xyz: torch.Tensor,
    new_offset: torch.Tensor,
    nsample: int,
    idx: torch.Tensor | None = None,
    with_xyz: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对齐 ``pointops.knn_query_and_group`` 的输入/输出约定。

    Pointcept 中该函数原为 CUDA kernel; 此处用 ``utils.pointnet_utils`` 的变长 kNN
    辅助函数 + 普通 PyTorch gather 替代, 输出语义保持一致:
    - ``with_xyz=True`` 时, grouped 张量在最后一维前置 **相对 xyz** (neighbor - center) 三维, 再拼接特征。
    - 返回 idx 供同一调用点对 key 与 value 复用近邻关系 (PointTransformerLayer 里就这么用)。

    Args:
        feat: (N, C) 全点特征。
        coord: (N, 3) 全点坐标。
        offset: (B,) cumsum 形式的 batch 划分边界。
        new_xyz: (M, 3) 查询中心坐标。
        new_offset: (B,) cumsum 形式的查询点 batch 边界。
        nsample: 每个查询点取的近邻数 K。
        idx: 可选, 已计算好的近邻索引 (M, K); 提供时跳过 kNN 搜索。
        with_xyz: True 时输出 (M, K, 3+C); False 时输出 (M, K, C)。

    Returns:
        grouped: (M, K, 3+C) 或 (M, K, C) — 见 with_xyz 说明。
        idx: (M, K) 近邻索引 (全局索引)。
    """
    if idx is None:
        # (M, K) — 变长 kNN, 索引以"全局拼接坐标"为基准。
        idx = knn_point_varlen(nsample, coord, offset, new_xyz, new_offset)

    # (M, K, C) - 直接按全局索引 gather 特征。
    grouped_feat = feat[idx.long(), :]

    if not with_xyz:
        return grouped_feat, idx

    # 相对位置: 邻居 xyz - 中心 xyz。 (M, K, 3) - (M, 1, 3) → (M, K, 3)
    grouped_xyz = coord[idx.long(), :] - new_xyz.unsqueeze(1)
    # 数据流: 相对位置 + 特征 在最后一维拼接 → (M, K, 3+C)。
    grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
    return grouped, idx


# ============================================================================
# Point Transformer Layer — 向量注意力 (与 Pointcept seg.py:19-78 一致)
# ============================================================================

class PointTransformerLayer(nn.Module):
    """Point Transformer V1 向量注意力层。

    复现 Pointcept ``point_transformer_seg.PointTransformerLayer`` 的全部行为:
    - Q/K/V 都是 1×1 全连接;
    - 位置编码 ``linear_p`` 是 (3 → 3 → LN → ReLU → out_planes), 输入是 K 邻居的相对 xyz;
    - 注意力权重通过 ``linear_w`` (LN → ReLU → Linear(mid → out//share) → LN → ReLU
      → Linear(out//share, out//share)) 映射到分组维度后做 softmax;
    - 输出聚合用 einsum 与 einops.rearrange 实现"分组向量注意力"。
    """

    def __init__(self, in_planes: int, out_planes: int, share_planes: int = 8, nsample: int = 16):
        super().__init__()
        # mid_planes 来自 Pointcept 的 `out_planes // 1`, 保留原写法以减少分歧。
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        # 相对位置编码 MLP: (3 → 3 → out_planes)。LN 在通道维做 BN-as-LN, 配合后续 ReLU。
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )

        # 注意力权重 MLP: (mid → mid → out//share → out//share)。
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )

        # 注意力 softmax 在 K 邻居维 (=维度 1) 归一化。
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pxo: [coord, feat, offset] 三元组。
                coord: (N, 3)
                feat: (N, C_in)
                offset: (B,) 各 batch 累积点数。

        Returns:
            (N, out_planes) 注意力更新后的特征。
        """
        coord, feat, offset = pxo  # 解包变长批表示

        # Q/K/V 投影: 都是 (N, C_q/k/v)
        x_q = self.linear_q(feat)
        x_k = self.linear_k(feat)
        x_v = self.linear_v(feat)

        # Key: 取 K 邻居后拼接相对 xyz → (N, K, 3 + mid_planes)
        x_k_grouped, idx = _knn_query_and_group(
            x_k, coord, offset, new_xyz=coord, new_offset=offset,
            nsample=self.nsample, with_xyz=True,
        )
        # Value: 复用上一步算出的 idx, 只 gather 特征 → (N, K, out_planes)
        x_v_grouped, _ = _knn_query_and_group(
            x_v, coord, offset, new_xyz=coord, new_offset=offset,
            idx=idx, nsample=self.nsample, with_xyz=False,
        )

        # 拆出相对位置与 key 特征: (N, K, 3) 与 (N, K, mid_planes)
        p_r = x_k_grouped[:, :, 0:3]
        x_k_only = x_k_grouped[:, :, 3:]

        # 相对位置编码: (N, K, 3) → linear_p → (N, K, out_planes)
        p_r = self.linear_p(p_r)

        # r_qk = key - query + reduce_sum_split(p_r). 把 p_r 沿通道维拆成
        # ``out_planes // mid_planes`` 个等长段后求和, 把它"投影"到 mid_planes 维度,
        # 与 (x_k - x_q) 形状一致。对应 Pointcept 的:
        #     einops.reduce(p_r, "n ns (i j) -> n ns j", reduction="sum", j=mid_planes)
        p_r_reduced = einops.reduce(
            p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes,
        )
        r_qk = x_k_only - x_q.unsqueeze(1) + p_r_reduced  # (N, K, mid_planes)

        # 注意力权重: (N, K, mid_planes) → linear_w → (N, K, out_planes // share)
        w = self.linear_w(r_qk)
        w = self.softmax(w)

        # 分组向量注意力: 把 out_planes 重新切成 (share, i=out_planes//share)。
        # 数据流:
        #   x_v_grouped + p_r: (N, K, out_planes)
        #   → rearrange 成 (N, K, share, i)
        #   einsum 与 w (N, K, i) 做加权求和, 在 K 维归约
        #   → 输出 (N, share, i) → 再 rearrange 回 (N, out_planes)
        x_v_plus_p = einops.rearrange(
            x_v_grouped + p_r, "n ns (s i) -> n ns s i", s=self.share_planes,
        )
        feat_out = torch.einsum("n t s i, n t i -> n s i", x_v_plus_p, w)
        feat_out = einops.rearrange(feat_out, "n s i -> n (s i)")
        return feat_out


# ============================================================================
# Transition Down — FPS + kNN 分组 + MLP + MaxPool (与 Pointcept seg.py:81-119 一致)
# ============================================================================

class TransitionDown(nn.Module):
    """点云下采样模块。

    与 Pointcept 一致:
    - ``stride == 1``: 不做空间下采样, 只对每个点做 Linear → BN → ReLU 通道升维。
    - ``stride != 1``: 对每个样本独立做 FPS 取 1/stride 个中心点, 再对每个中心做 kNN,
      拼上相对位置后做 Linear → BN → ReLU → MaxPool(K) 得到 (M, out_planes)。
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, nsample: int = 16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample
        if stride != 1:
            # 输入维度 3 + in_planes (拼了相对 xyz)
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_subsampled_offset(offset: torch.Tensor, stride: int) -> torch.Tensor:
        """对应 Pointcept 的 ``n_o = [o[i]/stride 的累积]``。

        对每个 batch 把点数除 stride (整除) 得到下采样目标点数, 再生成累积 offset。

        Args:
            offset: (B,) 当前层的累积 offset。
            stride: 下采样因子。

        Returns:
            (B,) 下采样后的累积 offset, dtype 与 ``offset`` 一致。
        """
        # 数据流: 先把每个样本的点数 (e_i - s_i) 除 stride, 累加生成新 offset。
        sub_counts = []
        running = 0
        for i in range(offset.shape[0]):
            s_i = 0 if i == 0 else offset[i - 1].item()
            e_i = offset[i].item()
            n_i = max((e_i - s_i) // stride, 1)  # 至少保留 1 个点防止空 batch
            running += n_i
            sub_counts.append(running)
        return torch.tensor(sub_counts, dtype=offset.dtype, device=offset.device)

    def forward(self, pxo: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            pxo: [coord, feat, offset] — 含义同 ``PointTransformerLayer.forward``。

        Returns:
            [new_coord, new_feat, new_offset] — 新的变长批三元组。
            ``stride==1`` 时点数不变, 否则点数被压缩为原来的 1/stride。
        """
        coord, feat, offset = pxo

        if self.stride == 1:
            # 通道升维: (N, in) → (N, out)
            feat = self.relu(self.bn(self.linear(feat)))
            return [coord, feat, offset]

        # 1. 算新 offset (每样本点数 ÷ stride)
        new_offset = self._make_subsampled_offset(offset, self.stride)

        # 2. FPS 取中心点索引: (M,)
        fps_idx = farthest_point_sample_varlen(coord, offset, new_offset)
        # 数据流: 用 FPS 索引 gather 出新中心坐标 (M, 3)
        new_coord = coord[fps_idx.long(), :]

        # 3. 对每个新中心做 kNN + grouping (拼相对 xyz) → (M, K, 3 + in_planes)
        grouped, _ = _knn_query_and_group(
            feat, coord, offset, new_xyz=new_coord, new_offset=new_offset,
            nsample=self.nsample, with_xyz=True,
        )

        # 4. MLP + MaxPool:
        #    (M, K, 3+in) → Linear → (M, K, out)
        #    → transpose → (M, out, K) 适配 BN1d/MaxPool1d 输入
        #    → BN → ReLU → MaxPool(K) → (M, out, 1) → squeeze → (M, out)
        x = self.linear(grouped)                  # (M, K, out)
        x = x.transpose(1, 2).contiguous()         # (M, out, K)
        x = self.relu(self.bn(x))
        x = self.pool(x).squeeze(-1)               # (M, out)

        return [new_coord, x, new_offset]


# ============================================================================
# Bottleneck — Linear + Attention + Linear 残差块 (与 Pointcept seg.py:171-192 一致)
# ============================================================================

class Bottleneck(nn.Module):
    """Point Transformer 残差瓶颈块。

    结构: Linear → BN → ReLU → PointTransformerLayer → BN → ReLU → Linear → BN, + identity → ReLU。
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, share_planes: int = 8, nsample: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo: List[torch.Tensor]) -> List[torch.Tensor]:
        coord, feat, offset = pxo
        identity = feat
        feat = self.relu(self.bn1(self.linear1(feat)))
        feat = self.relu(self.bn2(self.transformer([coord, feat, offset])))
        feat = self.bn3(self.linear3(feat))
        feat = feat + identity
        feat = self.relu(feat)
        return [coord, feat, offset]


# ============================================================================
# 分类骨干 — 与 Pointcept point_transformer_cls.py 一致, 仅替换输出头
# ============================================================================

class PointTransformerCls(nn.Module):
    """Point Transformer V1 分类骨干 (Pointcept ``PointTransformerCls`` 复现)。

    与官方一致:
    - planes=[32, 64, 128, 256, 512], stride=[1, 4, 4, 4, 4], nsample=[8, 16, 16, 16, 16],
      share_planes=8。
    - 5 个 enc 阶段, 每阶段 = TransitionDown(首) + (blocks[i]-1) 个 Bottleneck。
    - forward 接受 ``data_dict = {coord, feat, offset}`` 与官方完全相同;
      ``in_channels==3`` 时输入只有 coord, 否则把 (coord, feat) 拼接成 (N, in_channels)。
    - 全局平均池化按 offset 分段求均值, 与官方逐样本累加除以 cnt 完全一致。

    **唯一不同**: 官方只有一个 ``cls`` 分类头, 这里改为 SPAD 双头 ``cls_head`` + ``center_head``,
    分别输出 (B, num_classes) 与 (B, 3) 中心点。
    """

    def __init__(self, block, blocks: List[int], in_channels: int = 6,
                 num_classes: int = 40, center_dim: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = in_channels
        planes = [32, 64, 128, 256, 512]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes,
                                   stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes,
                                   stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes,
                                   stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes,
                                   stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes,
                                   stride=stride[4], nsample=nsample[4])

        # 分类头与官方完全一致 (Linear → BN → ReLU → Dropout(0.5) ×2 + 最后 Linear)。
        self.cls_head = nn.Sequential(
            nn.Linear(planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

        # 中心点回归头 (SPAD 特有, center-only 约定): 与其它 baseline 形状统一。
        self.center_head = nn.Sequential(
            nn.Linear(planes[4], 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, center_dim),
        )

    def _make_enc(self, block, planes: int, blocks: int, share_planes: int = 8,
                  stride: int = 1, nsample: int = 16) -> nn.Sequential:
        # 与 Pointcept _make_enc 完全相同: TransitionDown 首层 + 后续 Bottleneck。
        layers: List[nn.Module] = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data_dict: 包含三个键:
                ``coord``: (N, 3) 全部点坐标 (variable-length 批拼接形式)。
                ``feat``: (N, C_in - 3) 附加特征 (``in_channels==3`` 时被忽略)。
                ``offset``: (B,) 累积 offset。

        Returns:
            logits: (B, num_classes)
            center_pred: (B, 3) 归一化空间中心点。
        """
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # 与官方一致: in_channels==3 时只用坐标; 否则把 coord 与 feat 拼成 (N, in_channels)。
        x0 = coord if self.in_channels == 3 else torch.cat((coord, feat), 1)

        # 5 阶段编码: 每阶段返回 [p_i, x_i, o_i] 变长批。
        coord1, x1, offset1 = self.enc1([coord, x0, offset])
        coord2, x2, offset2 = self.enc2([coord1, x1, offset1])
        coord3, x3, offset3 = self.enc3([coord2, x2, offset2])
        coord4, x4, offset4 = self.enc4([coord3, x3, offset3])
        coord5, x5, offset5 = self.enc5([coord4, x4, offset4])

        # 全局平均池化: 按 offset 把每个样本的特征求均值 (与 Pointcept 一致)。
        # 数据流: (N5, C5) → 逐 batch 取 [s_i:e_i] 切片 → mean(0) → (1, C5) → cat → (B, C5)。
        pooled = []
        for i in range(offset5.shape[0]):
            s_i = 0 if i == 0 else int(offset5[i - 1].item())
            e_i = int(offset5[i].item())
            cnt = e_i - s_i
            pooled.append(x5[s_i:e_i, :].sum(0, keepdim=True) / cnt)
        feat_global = torch.cat(pooled, dim=0)  # (B, planes[-1])

        logits = self.cls_head(feat_global)
        center_pred = self.center_head(feat_global)
        return logits, center_pred


# ============================================================================
# SPAD 适配封装 — 把 (B, N, 4) xyzi 输入转成 data_dict 调用主体网络
# ============================================================================

class PointTransformerClassification(PointTransformerCls):
    """SPAD 训练管道适配封装。

    与项目其它 baseline 一致, 接收 (B, N, 4) xyzi (或 (B, N, 3) xyz, 自动补零强度)
    点云, 内部转换为 Pointcept 期望的 ``data_dict = {coord, feat, offset}``。

    默认 ``block_config=(1, 1, 1, 1, 1)`` 对应官方 ``PointTransformer-Cls26``,
    单个 Bottleneck/阶段; 若需更深可换成 ``(1, 2, 2, 2, 2)`` 或 ``(1, 2, 3, 5, 2)``。
    """

    def __init__(self, num_classes: int = 26, block_config: Tuple[int, ...] = (1, 2, 2, 2, 2),
                 in_channels: int = 4, center_dim: int = 3):
        super().__init__(
            Bottleneck, list(block_config),
            in_channels=in_channels, num_classes=num_classes, center_dim=center_dim,
        )

    @staticmethod
    def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
        """统一外部传入的点云为 (B, N, 4) xyzi 布局。

        支持 (B, N, 4)、(B, 4, N)、(B, N, 3)、(B, 3, N) 四种形式;
        xyz-only 输入会补一列零强度。
        """
        if x.ndim != 3:
            raise ValueError(f"PointTransformerClassification 仅接受 3D 输入, 收到形状 {tuple(x.shape)}")
        # 数据流: 先按最后一维或第二维识别通道方向, 再统一成 (B, N, C)。
        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()  # (B, C, N) → (B, N, C)
        else:
            raise ValueError(f"PointTransformerClassification 不支持的输入形状 {tuple(x.shape)}")
        if points.shape[-1] == 3:
            # 数据流: (B, N, 3) → 末尾补 0 强度 → (B, N, 4)
            pad_intensity = torch.zeros(
                points.shape[0], points.shape[1], 1, dtype=points.dtype, device=points.device,
            )
            points = torch.cat([points, pad_intensity], dim=-1)
        return points

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, 4) xyzi 或可被 ``_normalize_input_points`` 识别的等价布局。

        Returns:
            (logits [B, num_classes], center_pred [B, 3])
        """
        points = self._normalize_input_points(x)         # (B, N, 4)
        batch_size, num_points, _ = points.shape

        # 数据流: (B, N, 4) → (B*N, 3) coord + (B*N, in_channels-3) feat,
        # 然后用 arange 生成 cumsum 形式的 offset = [N, 2N, ..., B*N]。
        coord = points[:, :, :3].reshape(batch_size * num_points, 3).contiguous()
        feat = points[:, :, 3:].reshape(batch_size * num_points, -1).contiguous()
        offset = torch.arange(
            num_points, (batch_size + 1) * num_points, step=num_points,
            dtype=torch.long, device=points.device,
        )

        data_dict = {"coord": coord, "feat": feat, "offset": offset}
        return super().forward(data_dict)


# ============================================================================
# 不同深度的便捷构造函数 (对应 Pointcept Cls26/Cls38/Cls50)
# ============================================================================

def point_transformer_cls26(num_classes: int = 26, **kwargs) -> PointTransformerClassification:
    """PointTransformer-Cls26: blocks=[1, 1, 1, 1, 1] (最浅)。"""
    return PointTransformerClassification(
        num_classes=num_classes, block_config=(1, 1, 1, 1, 1), **kwargs,
    )


def point_transformer_cls38(num_classes: int = 26, **kwargs) -> PointTransformerClassification:
    """PointTransformer-Cls38: blocks=[1, 2, 2, 2, 2]。"""
    return PointTransformerClassification(
        num_classes=num_classes, block_config=(1, 2, 2, 2, 2), **kwargs,
    )


def point_transformer_cls50(num_classes: int = 26, **kwargs) -> PointTransformerClassification:
    """PointTransformer-Cls50: blocks=[1, 2, 3, 5, 2] (最深)。"""
    return PointTransformerClassification(
        num_classes=num_classes, block_config=(1, 2, 3, 5, 2), **kwargs,
    )


# ============================================================================
# 形状自检 + GPU 显存压力测试 (符合 pointcloud-3d-workflows SKILL 规约)
# ============================================================================

def _quick_test() -> None:
    """简单 forward 形状检查; CPU/CUDA 通用。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing PointTransformer V1 on {device}")

    model = PointTransformerClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, center_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Center: {tuple(center_pred.shape)}")
    print("OK PointTransformer V1 works correctly")


def _gpu_memory_test() -> None:
    """逐 batch size 显存扫查 (4/8/16/32)。CPU 环境下直接跳过。"""
    import gc
    if not torch.cuda.is_available():
        print("无 CUDA, 跳过 GPU 显存测试。")
        return

    print("\n=== GPU 显存测试 ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        if total_mem:
            print(f"总显存: {total_mem / 1024 ** 3:.1f} GB")
    except Exception:
        pass
    print()

    num_points = 1024
    for batch_size in [4, 8, 16, 32]:
        try:
            m = PointTransformerClassification(num_classes=26).cuda()
            pts = torch.randn(batch_size, num_points, 4).cuda()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            m.train()
            out = m(pts)
            loss = out[0].sum() + out[1].sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"  B={batch_size:2d}: peak {peak:6.0f} MB")
            del m, pts, out, loss
            torch.cuda.empty_cache()
            gc.collect()
        except torch.cuda.OutOfMemoryError:
            print(f"  B={batch_size:2d}: OOM!")
            torch.cuda.empty_cache()
            gc.collect()
            break


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()
