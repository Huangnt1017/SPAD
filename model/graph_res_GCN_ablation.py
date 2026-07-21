"""GraphResidual-GCN 结构消融模型。

该文件与正式模型 ``model/graph_res_GCN.py`` 隔离，提供三个真正的硬开关：

- ``use_physical_branch``：关闭后不构造坐标图 GraphSAGE 分支；
- ``use_se_gate``：关闭后不构造 SE 通道门控；
- ``use_coord_residual``：关闭后不构造坐标门控、坐标残差、坐标编码器和缩放参数；
- ``operator``：在完全相同的 KNN、双分支、融合与残差结构下切换 GraphSAGE/EdgeCNN。

全开时的模块拓扑、参数命名与正式 GraphResidual-GCN 保持一致，可用于 A1--A3
头部消融；关闭单项时用于 P2 结构归因。输入为 ``(B, N, 4)``，输出字典至少含
``logits (B,C)`` 与 ``box_pred (B,3)``；质心头额外输出逐点 ``seg_logits``。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from model.graph_res_GCN import (
    BatchedEdgeCNNConv,
    BatchedSAGEConv,
    _HAS_CKPT,
    _ckpt,
    batched_knn_edge_index,
    knn_without_self,
)
from utils.graph_ops import knn_gpu
from utils.heads import (
    SegmentationCentroidHead,
    build_standard_box_head,
    build_standard_cls_head,
)


class GraphResidualAblationBlockGCN(nn.Module):
    """带物理分支、SE 和坐标残差硬开关的 GraphSAGE 残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        k: int = 20,
        se_ratio: int = 4,
        fuse_bottleneck_ratio: int = 1,
        coord_mid_ratio: int = 1,
        aggregation: str = "max",
        operator: str = "sage",
        exclude_self: bool = True,
        feature_residual: bool = True,
        coord_scale_init: float = 0.1,
        use_physical_branch: bool = True,
        use_se_gate: bool = True,
        use_coord_residual: bool = True,
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if se_ratio <= 0 or fuse_bottleneck_ratio <= 0 or coord_mid_ratio <= 0:
            raise ValueError("se_ratio, fuse_bottleneck_ratio and coord_mid_ratio must be positive")
        if coord_scale_init < 0:
            raise ValueError(f"coord_scale_init must be non-negative, got {coord_scale_init}")
        if operator not in {"sage", "edge_cnn"}:
            raise ValueError(f"operator must be 'sage' or 'edge_cnn', got {operator}")

        self.k = int(k)
        self.operator = operator
        self.exclude_self = bool(exclude_self)
        self.use_feature_residual = bool(feature_residual)
        self.use_physical_branch = bool(use_physical_branch)
        self.use_se_gate = bool(use_se_gate)
        self.use_coord_residual = bool(use_coord_residual)

        operator_cls = BatchedSAGEConv if operator == "sage" else BatchedEdgeCNNConv
        self.gcn_f = operator_cls(in_channels, out_channels, aggregation=aggregation)
        self.bn_f = nn.BatchNorm1d(out_channels)

        if self.use_physical_branch:
            self.gcn_p = operator_cls(4, out_channels, aggregation=aggregation)
            self.bn_p = nn.BatchNorm1d(out_channels)
        else:
            self.gcn_p = None
            self.bn_p = None

        if self.use_se_gate:
            se_hidden = max(out_channels // int(se_ratio), 1)
            self.se_gate = nn.Sequential(
                nn.Linear(out_channels, se_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(se_hidden, out_channels, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.se_gate = None

        fuse_in_channels = 2 * out_channels if self.use_physical_branch else out_channels
        if fuse_bottleneck_ratio == 1:
            self.fuse_conv = nn.Sequential(
                nn.Conv1d(fuse_in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            fuse_mid = max(out_channels // int(fuse_bottleneck_ratio), 16)
            self.fuse_conv = nn.Sequential(
                nn.Conv1d(fuse_in_channels, fuse_mid, 1, bias=False),
                nn.BatchNorm1d(fuse_mid),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(fuse_mid, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        if self.use_coord_residual:
            self.coord_gate = nn.Sequential(
                nn.Conv1d(4, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            self.coord_res = nn.Sequential(
                nn.Conv1d(4, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            coord_mid = (
                out_channels
                if coord_mid_ratio == 1
                else max(out_channels // int(coord_mid_ratio), 16)
            )
            self.coord_encoder = nn.Sequential(
                nn.Conv1d(4, coord_mid, 1, bias=False),
                nn.BatchNorm1d(coord_mid),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(coord_mid, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            self.coord_scale = nn.Parameter(torch.tensor(float(coord_scale_init)))
        else:
            self.coord_gate = None
            self.coord_res = None
            self.coord_encoder = None
            self.register_parameter("coord_scale", None)

        if self.use_feature_residual:
            self.feature_residual = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            )
        else:
            self.feature_residual = None

        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        p_edge_index: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行一个不改变点数的消融块前向。"""
        batch_size, _, num_points = f.shape
        k = min(self.k, num_points - 1)
        knn_f = knn_without_self(f, k) if self.exclude_self else knn_gpu(f, k)
        f_edge_index = batched_knn_edge_index(knn_f, batch_size, num_points)

        f_gcn = self.act(self.bn_f(self.gcn_f(f, f_edge_index)))
        if self.se_gate is not None:
            f_gcn = f_gcn * self.se_gate(f_gcn.mean(dim=-1)).unsqueeze(-1)

        if self.use_physical_branch:
            if p_edge_index is None or self.gcn_p is None or self.bn_p is None:
                raise RuntimeError("physical branch requires a coordinate edge index")
            p_gcn = self.act(self.bn_p(self.gcn_p(p, p_edge_index)))
            fused_input = torch.cat([f_gcn, p_gcn], dim=1)
        else:
            fused_input = f_gcn
        fused = self.fuse_conv(fused_input)

        feature_skip = self.feature_residual(f) if self.feature_residual is not None else 0.0
        out = fused + feature_skip
        if self.use_coord_residual:
            if self.coord_gate is None or self.coord_res is None or self.coord_encoder is None:
                raise RuntimeError("coordinate residual modules are not initialized")
            gate = torch.sigmoid(self.coord_gate(p))
            coord_delta = gate * (self.coord_res(p) + self.coord_encoder(p))
            out = out + self.coord_scale * coord_delta
        return p, self.act(out)


class GraphResidualGCNAblationNet(nn.Module):
    """用于 A1--A3 与 P2 结构消融的多任务 GraphResidual-GCN。"""

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 20,
        use_checkpoint: bool = True,
        dropout: float = 0.3,
        box_dim: int = 3,
        seg_centroid_box: bool = True,
        aggregation: str = "max",
        operator: str = "sage",
        exclude_self: bool = True,
        feature_residual: bool = True,
        coord_scale_init: float = 0.1,
        use_physical_branch: bool = True,
        use_se_gate: bool = True,
        use_coord_residual: bool = True,
        block_channels: Tuple[int, int, int, int] = (64, 64, 128, 256),
        agg_channels: int = 512,
        block_se_ratio: int = 4,
        fuse_bottleneck_ratio: int = 1,
        coord_mid_ratio: int = 1,
    ) -> None:
        super().__init__()
        if len(block_channels) != 4 or any(int(c) <= 0 for c in block_channels):
            raise ValueError(f"block_channels must contain four positive values, got {block_channels}")
        if agg_channels <= 0:
            raise ValueError(f"agg_channels must be positive, got {agg_channels}")
        if operator not in {"sage", "edge_cnn"}:
            raise ValueError(f"operator must be 'sage' or 'edge_cnn', got {operator}")

        self.k = int(k)
        self.use_checkpoint = bool(use_checkpoint)
        self.operator = operator
        self.exclude_self = bool(exclude_self)
        self.seg_centroid_box = bool(seg_centroid_box)
        self.use_physical_branch = bool(use_physical_branch)
        self.use_se_gate = bool(use_se_gate)
        self.use_coord_residual = bool(use_coord_residual)
        self.block_channels = tuple(int(c) for c in block_channels)
        self.agg_channels = int(agg_channels)

        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )

        block_cfg = dict(
            k=k,
            se_ratio=block_se_ratio,
            fuse_bottleneck_ratio=fuse_bottleneck_ratio,
            coord_mid_ratio=coord_mid_ratio,
            aggregation=aggregation,
            operator=operator,
            exclude_self=exclude_self,
            feature_residual=feature_residual,
            coord_scale_init=coord_scale_init,
            use_physical_branch=use_physical_branch,
            use_se_gate=use_se_gate,
            use_coord_residual=use_coord_residual,
        )
        c1, c2, c3, c4 = self.block_channels
        self.block1 = GraphResidualAblationBlockGCN(32, c1, **block_cfg)
        self.block2 = GraphResidualAblationBlockGCN(c1, c2, **block_cfg)
        self.block3 = GraphResidualAblationBlockGCN(c2, c3, **block_cfg)
        self.block4 = GraphResidualAblationBlockGCN(c3, c4, **block_cfg)

        self.agg_conv = nn.Sequential(
            nn.Conv1d(sum(self.block_channels), self.agg_channels, 1, bias=False),
            nn.BatchNorm1d(self.agg_channels),
            nn.LeakyReLU(0.2),
        )
        pooled_dim = self.agg_channels * 2
        self.cls_head = build_standard_cls_head(pooled_dim, num_classes, dropout=dropout)
        self.box_head = (
            SegmentationCentroidHead(in_channels=self.agg_channels, coord_dim=box_dim)
            if self.seg_centroid_box
            else build_standard_box_head(pooled_dim, box_dim=box_dim, dropout=dropout)
        )

    def effective_config(self) -> Dict[str, object]:
        """返回可直接写入日志/checkpoint 的有效结构配置。"""
        return {
            "box_head": "centroid" if self.seg_centroid_box else "mlp",
            "gcn_operator": self.operator,
            "gcn_use_physical_branch": self.use_physical_branch,
            "gcn_use_se_gate": self.use_se_gate,
            "gcn_use_coord_residual": self.use_coord_residual,
        }

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向；输入 ``(B,N,4)``，输出分类与三维中心预测。"""
        if points.ndim != 3 or points.shape[-1] != 4:
            raise ValueError(f"points must have shape (B, N, 4), got {tuple(points.shape)}")
        p = points.transpose(1, 2).contiguous()
        batch_size, _, num_points = p.shape
        if num_points < 2:
            raise ValueError(f"GraphResidual GCN requires at least 2 points, got N={num_points}")
        f = self.stem(p)

        p_edge_index = None
        if self.use_physical_branch:
            k = min(self.k, num_points - 1)
            knn_p = knn_without_self(p, k) if self.exclude_self else knn_gpu(p, k)
            p_edge_index = batched_knn_edge_index(knn_p, batch_size, num_points)

        def _run_block(block, block_p, block_f, block_edges):
            return block(block_p, block_f, block_edges)

        use_ckpt = self.use_checkpoint and self.training and _HAS_CKPT
        if use_ckpt:
            p, f1 = _ckpt(_run_block, self.block1, p, f, p_edge_index, use_reentrant=False)
            p, f2 = _ckpt(_run_block, self.block2, p, f1, p_edge_index, use_reentrant=False)
            p, f3 = _ckpt(_run_block, self.block3, p, f2, p_edge_index, use_reentrant=False)
            p, f4 = _ckpt(_run_block, self.block4, p, f3, p_edge_index, use_reentrant=False)
        else:
            p, f1 = self.block1(p, f, p_edge_index)
            p, f2 = self.block2(p, f1, p_edge_index)
            p, f3 = self.block3(p, f2, p_edge_index)
            p, f4 = self.block4(p, f3, p_edge_index)

        point_features = self.agg_conv(torch.cat([f1, f2, f3, f4], dim=1))
        pooled = torch.cat(
            [point_features.max(dim=-1)[0], point_features.mean(dim=-1)],
            dim=1,
        )
        logits = self.cls_head(pooled)
        if self.seg_centroid_box:
            centroid_out = self.box_head(
                point_features,
                points[..., :3].transpose(1, 2).contiguous(),
            )
            return {
                "logits": logits,
                "box_pred": centroid_out["centroid"],
                "seg_logits": centroid_out["seg_logits"],
                "seg_weights": centroid_out["seg_weights"],
            }
        return {"logits": logits, "box_pred": self.box_head(pooled)}
