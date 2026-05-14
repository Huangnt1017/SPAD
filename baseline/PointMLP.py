"""
PointMLP: Rethinking Network Design and Local Geometry in Point Cloud

GitHub:  https://github.com/ma-xu/pointMLP-pytorch
Local:   D:\essay\3d目标检测复现仓库\pointMLP-pytorch-main

完整复现 (Ma Xu et al.):
- LocalGrouper: FPS + KNN 分组 + 可选的 center/anchor 归一化 + 仿射变换
- PreExtraction: 预提取模块 (Conv1D + 残差块 + 自适应最大池化)
- PosExtraction: 后提取模块 (残差 Conv1D)
- 4 阶段层级架构: embedding → stage×4 → 全局池化 → 分类器
- 输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])

Reference:
@inproceedings{
    ma2022rethinking,
    title={Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual {MLP} Framework},
    author={Xu Ma and Can Qin and Haoxuan You and Haoxi Ran and Yun Fu},
    booktitle={International Conference on Learning Representations},
    year={2022},
    pages={1--19},
    url={https://openreview.net/forum?id=3Pbra-_u76D}
}
or
@article{ma2022rethinking,
    title={Rethinking network design and local geometry in point cloud: A simple residual MLP framework},
    author={Ma, Xu and Qin, Can and You, Haoxuan and Ran, Haoxi and Fu, Yun},
    journal={arXiv preprint arXiv:2202.07123},
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
# 工具函数
# ============================================================================

def get_activation(activation: str) -> nn.Module:
    """返回指定的激活函数模块。"""
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


# ============================================================================
# LocalGrouper: FPS + KNN 分组 + 归一化 + 仿射变换
# ============================================================================

class LocalGrouper(nn.Module):
    """局部分组模块。

    用 FPS 采样关键点 → KNN 获取邻居 → 可选归一化 (center/anchor) → 仿射变换。

    与官方实现一致，使用 xyz 坐标做 FPS 和 KNN，
    group 后拼接特征和坐标 (use_xyz=True时)。
    """
    def __init__(self, channel: int, groups: int, kneighbors: int,
                 use_xyz: bool = True, normalize: str = "center"):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize else None
        if self.normalize not in ["center", "anchor"]:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) — 点云坐标
            points: (B, N, C) — 点特征
        Returns:
            new_xyz: (B, G, 3) — 分组后中心坐标
            new_points: (B, G, K, C') — 分组后特征 (含归一化坐标)
        """
        B, N, C = xyz.shape
        S = self.groups

        # FPS 采样
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        # KNN
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        # 是否拼接坐标
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        # 归一化 + 仿射变换
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            elif self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1,
                            keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        # 拼接中心特征 (广播到每个邻居)
        new_points = torch.cat(
            [grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)],
            dim=-1
        )
        return new_xyz, new_points


# ============================================================================
# 基础卷积块
# ============================================================================

class ConvBNReLU1D(nn.Module):
    """Conv1d + BN + 激活。"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, bias: bool = True, activation: str = 'relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    """带残差连接的 Conv1d 块。

    架构: Conv1d → BN → Act → Conv1d → BN → (+ 输入) → Act
    支持分组卷积和通道扩展。
    """
    def __init__(self, channel: int, kernel_size: int = 1, groups: int = 1,
                 res_expansion: float = 1.0, bias: bool = True, activation: str = 'relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, int(channel * res_expansion), kernel_size,
                      groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size,
                          groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


# ============================================================================
# PreExtraction & PosExtraction
# ============================================================================

class PreExtraction(nn.Module):
    """预提取模块: 对分组后的特征做 MLP + 残差 + 自适应最大池化。

    将 (B, G, K, D) 转换为 (B, D_out, G)。
    """
    def __init__(self, channels: int, out_channels: int, blocks: int = 1,
                 groups: int = 1, res_expansion: float = 1.0, bias: bool = True,
                 activation: str = 'relu', use_xyz: bool = True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(
                out_channels, groups=groups, res_expansion=res_expansion,
                bias=bias, activation=activation,
            ))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        """
        Args:
            x: (B, G, K, D) — 分组特征
        Returns:
            out: (B, D_out, G) — 池化后特征
        """
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)       # (B, G, D, K)
        x = x.reshape(-1, d, s)          # (B*G, D, K)
        x = self.transfer(x)             # (B*G, D_out, K)
        x = self.operation(x)            # (B*G, D_out, K)
        x = F.adaptive_max_pool1d(x, 1).view(b, n, -1)  # (B, G, D_out)
        x = x.permute(0, 2, 1)           # (B, D_out, G)
        return x


class PosExtraction(nn.Module):
    """后提取模块: 对池化后的特征做残差 Conv1d。"""
    def __init__(self, channels: int, blocks: int = 1, groups: int = 1,
                 res_expansion: float = 1.0, bias: bool = True, activation: str = 'relu'):
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(
                channels, groups=groups, res_expansion=res_expansion,
                bias=bias, activation=activation,
            ))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        """
        Args:
            x: (B, D, G) — 池化后特征
        Returns:
            out: (B, D, G) — 处理后的特征
        """
        return self.operation(x)


# ============================================================================
# PointMLP 主模型
# ============================================================================

class PointMLPModel(nn.Module):
    """PointMLP 层级点云分类模型 (官方架构)。

    4 阶段设计:
    1. Embedding: Conv1d(3→embed_dim)
    2-5. 每个阶段: LocalGrouper → PreExtraction → PosExtraction
    6. 全局自适应最大池化 + 分类器

    每阶段分组点数递减 (reducer), 通道数递增 (dim_expansion).
    """
    def __init__(self, points: int = 1024, class_num: int = 40,
                 embed_dim: int = 64, groups: int = 1, res_expansion: float = 1.0,
                 activation: str = "relu", bias: bool = True,
                 use_xyz: bool = True, normalize: str = "center",
                 dim_expansion: tuple = (2, 2, 2, 2),
                 pre_blocks: tuple = (2, 2, 2, 2),
                 pos_blocks: tuple = (2, 2, 2, 2),
                 k_neighbors: tuple = (32, 32, 32, 32),
                 reducers: tuple = (2, 2, 2, 2)):
        super().__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = self.points

        for i in range(self.stages):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor,
                                          use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)

            pre_block_module = PreExtraction(
                last_channel, out_channel, pre_block_num, groups=groups,
                res_expansion=res_expansion, bias=bias, activation=activation,
                use_xyz=use_xyz,
            )
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(
                out_channel, pos_block_num, groups=groups,
                res_expansion=res_expansion, bias=bias, activation=activation,
            )
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, class_num),
        )

        # BBox 回归头
        self.box_head = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) — 输入点云坐标 (PointMLP 原始接口)
        Returns:
            logits: (B, class_num)
            box_pred: (B, 6)
        """
        batch_size, _, _ = x.size()

        # 保存原始坐标用于 LocalGrouper: (B, 3, N) → (B, N, 3)
        xyz = x.permute(0, 2, 1).contiguous()  # (B, N, 3)

        # Embedding: (B, 3, N) → (B, D, N)
        x = self.embedding(x)

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            cur_xyz = xyz

        # 全局自适应最大池化: (B, D, G) → (B, D)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)

        logits = self.cls_head(x)
        box_pred = self.box_head(x)
        return logits, box_pred


# ============================================================================
# SPAD 兼容包装类
# ============================================================================

class PointMLPClassification(nn.Module):
    """PointMLP 分类 + 3D BBox (适配 SPAD 训练管道)。

    封装 PointMLPModel, 处理 (B, N, 4) xyzi 输入到 (logits, box_pred) 输出。
    默认使用 pointMLP 配置 (use_xyz=False, normalize="anchor")。
    """
    def __init__(self, num_classes: int = 26, variant: str = "pointmlp", **kwargs):
        super().__init__()

        if variant == "pointmlp":
            config = dict(
                embed_dim=64, groups=1, res_expansion=1.0,
                activation="relu", bias=False, use_xyz=False, normalize="anchor",
                dim_expansion=(2, 2, 2, 2), pre_blocks=(2, 2, 2, 2),
                pos_blocks=(2, 2, 2, 2), k_neighbors=(24, 24, 24, 24),
                reducers=(2, 2, 2, 2),
            )
        elif variant == "pointmlpelite":
            config = dict(
                embed_dim=32, groups=1, res_expansion=0.25,
                activation="relu", bias=False, use_xyz=False, normalize="anchor",
                dim_expansion=(2, 2, 2, 1), pre_blocks=(1, 1, 2, 1),
                pos_blocks=(1, 1, 2, 1), k_neighbors=(24, 24, 24, 24),
                reducers=(2, 2, 2, 2),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        config.update(kwargs)
        config['class_num'] = num_classes
        self.model = PointMLPModel(**config)

    @staticmethod
    def _normalize_input_points(x):
        """统一输入为 (B, N, 4) xyzi。"""
        if x.ndim != 3:
            raise ValueError(f"PointMLPClassification expects 3D input, got {tuple(x.shape)}")
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
        # PointMLP 只使用 xyz: (B, N, 3) → (B, 3, N)
        pts = x[:, :, :3].transpose(1, 2).contiguous()
        return self.model(pts)


# ============================================================================
# 工厂函数 (对应官方 pointMLP / pointMLPElite)
# ============================================================================

def pointMLP(num_classes=40, **kwargs):
    """创建标准 PointMLP 模型。"""
    return PointMLPClassification(num_classes=num_classes, variant="pointmlp", **kwargs)


def pointMLPElite(num_classes=40, **kwargs):
    """创建轻量 PointMLP-Elite 模型。"""
    return PointMLPClassification(num_classes=num_classes, variant="pointmlpelite", **kwargs)


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
            m = PointMLPClassification(num_classes=26).cuda()
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
    print(f"Testing PointMLP on {device}")

    model = PointMLPClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Box:    {tuple(box_pred.shape)}")
    print("✓ PointMLP works correctly")


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()
