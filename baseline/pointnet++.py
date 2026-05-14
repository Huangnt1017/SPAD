"""
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

GitHub:  https://github.com/charlesq34/pointnet2 (TensorFlow)
Local:   D:\essay\3d目标检测复现仓库\pointnet2-master

官方参考实现复现 (Charles R. Qi):
- 完整包含 SSG (Single-Scale Grouping) 与 MSG (Multi-Scale Grouping) 变体
- Set Abstraction (SA) 模块 + Feature Propagation (FP) 模块
- 输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])

Reference:
@article{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pointnet_utils import (
    square_distance, index_points, farthest_point_sample, query_ball_point
)


# ============================================================================
# 分组相关函数: ball query / kNN / sample & group
# ============================================================================

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    FPS 采样 + Ball Query 分组 (SSG)。
    先将坐标归一化到以采样点为中心，再与特征拼接。

    Args:
        npoint: FPS 采样点数
        radius: ball query 半径
        nsample: 每组最大近邻数
        xyz: [B, N, 3] 全部点坐标
        points: [B, N, D] 点特征，或 None (只用坐标)
    Returns:
        new_xyz: [B, npoint, 3] 采样点坐标
        new_points: [B, npoint, nsample, 3+D] 归一化坐标 + 特征
    """
    fps_idx = farthest_point_sample(xyz, npoint)         # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)                 # [B, npoint, 3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)                 # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]  # 中心归一化

    if points is not None:
        grouped_points = index_points(points, idx)       # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    全局分组: 将全部点当作一个组，用于最后一层 SA。
    等价于 npoint=1, radius=inf 的特殊情况。
    """
    device = xyz.device
    B, N, _ = xyz.shape
    new_xyz = torch.zeros(B, 1, 3, device=device)
    grouped_xyz = xyz.view(B, 1, N, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# ============================================================================
# Set Abstraction (SA) 模块 — SSG 与 MSG 变体
# ============================================================================

class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction (SSG — Single-Scale Grouping)。

    流程: FPS → Ball Query → 中心归一化 → MLP(Conv2d) → 最大池化
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # MLP 输入通道 = 归一化坐标(3) + 点特征通道
        layers = []
        last_channel = in_channel + 3
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        Args:
            xyz: [B, 3, N] — 点云坐标
            points: [B, D, N] — 点特征，或 None
        Returns:
            new_xyz: [B, 3, S] — 采样点坐标
            new_points: [B, D', S] — 池化后特征
        """
        # 转为 (B, N, C) 格式便于操作
        xyz = xyz.transpose(1, 2).contiguous()     # [B, N, 3]
        if points is not None:
            points = points.transpose(1, 2).contiguous()  # [B, N, D]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )  # new_points: [B, S, K, 3+D]

        # Conv2d 需要 (B, C, K, S) 格式
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]    # [B, D', S]

        new_xyz = new_xyz.transpose(1, 2).contiguous()  # [B, 3, S]
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    PointNet++ Set Abstraction with Multi-Scale Grouping (MSG)。

    在同一层中，使用多个半径的 ball query 分别提取特征后拼接。
    参考官方实现 pointnet_sa_module_msg。
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        # 每个尺度对应一个独立的 MLP
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for i in range(len(radius_list)):
            layers = []
            last_channel = in_channel + 3  # 归一化 xyz(3) + 输入特征
            for out_channel in mlp_list[i]:
                layers.append(nn.Conv2d(last_channel, out_channel, 1))
                layers.append(nn.BatchNorm2d(out_channel))
                layers.append(nn.ReLU(inplace=True))
                last_channel = out_channel
            self.mlp_convs.append(nn.Sequential(*layers))

    def forward(self, xyz, points):
        """
        Args:
            xyz: [B, 3, N]
            points: [B, D, N] 或 None
        Returns:
            new_xyz: [B, 3, S]
            new_points: [B, sum(mlp_list[i][-1]), S] — 多尺度特征拼接
        """
        xyz = xyz.transpose(1, 2).contiguous()      # [B, N, 3]
        if points is not None:
            points = points.transpose(1, 2).contiguous()  # [B, N, D]

        B, N, _ = xyz.shape
        S = self.npoint

        # FPS 采样关键点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))  # [B, S, 3]

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            nsample = self.nsample_list[i]
            # Ball query
            idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)            # [B, S, K, 3]
            grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]  # 中心归一化

            if points is not None:
                grouped_points = index_points(points, idx)   # [B, S, K, D]
                grouped_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_features = grouped_xyz_norm

            # MLP + 池化
            grouped_features = grouped_features.permute(0, 3, 2, 1).contiguous()  # [B, C, K, S]
            grouped_features = self.mlp_convs[i](grouped_features)
            grouped_features = torch.max(grouped_features, 2)[0]  # [B, C', S]
            new_points_list.append(grouped_features)

        # 多尺度拼接
        new_points_concat = torch.cat(new_points_list, dim=1)   # [B, sum(C'), S]
        new_xyz = new_xyz.transpose(1, 2).contiguous()          # [B, 3, S]
        return new_xyz, new_points_concat


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction (SSG)
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        layers = []
        last_channel = in_channel + 3
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        xyz: [B, 3, N]
        points: [B, D, N] or None
        Return:
            new_xyz: [B, 3, S]
            new_points: [B, D', S]
        """
        xyz = xyz.transpose(1, 2).contiguous()  # [B, N, 3]
        if points is not None:
            points = points.transpose(1, 2).contiguous()  # [B, N, D]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )  # new_points: [B, S, K, C]

        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # [B, C, K, S]
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]  # [B, D', S]

        new_xyz = new_xyz.transpose(1, 2).contiguous()  # [B, 3, S]
        return new_xyz, new_points


# ============================================================================
# Feature Propagation (FP) 模块 — 用于上采样/分割网络
# ============================================================================

class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation (FP) 模块。

    通过三近邻插值将稀疏层特征传播回稠密层，再与跳连接特征拼接后做 MLP。
    参考官方实现 pointnet_fp_module。
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp_convs = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: [B, 3, N1] — 稠密层坐标 (待上采样到的目标分辨率)
            xyz2: [B, 3, N2] — 稀疏层坐标 (上一层 SA 输出, N2 < N1)
            points1: [B, D1, N1] — 稠密层特征 (来自跳连接)
            points2: [B, D2, N2] — 稀疏层特征 (需要插值上采样)
        Returns:
            new_points: [B, mlp[-1], N1] — 融合后的特征
        """
        xyz1 = xyz1.transpose(1, 2).contiguous()   # [B, N1, 3]
        xyz2 = xyz2.transpose(1, 2).contiguous()   # [B, N2, 3]
        points2 = points2.transpose(1, 2).contiguous()  # [B, N2, D2]

        B, N1, _ = xyz1.shape
        N2 = xyz2.shape[1]

        if N2 == 1:
            # 只有一个点: 直接广播到所有点
            interpolated_points = points2.repeat(1, N1, 1)
        else:
            # 三近邻反距离加权插值
            dists = square_distance(xyz1, xyz2)     # [B, N1, N2]
            dists, idx = dists.topk(k=3, dim=-1, largest=False)  # [B, N1, 3]
            dists_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dists_recip, dim=-1, keepdim=True)
            weight = dists_recip / norm             # [B, N1, 3]
            # 收集三个近邻的特征
            grouped_points = index_points(points2, idx)  # [B, N1, 3, D2]
            interpolated_points = torch.sum(
                grouped_points * weight.unsqueeze(-1), dim=2
            )  # [B, N1, D2]

        if points1 is not None:
            points1 = points1.transpose(1, 2).contiguous()  # [B, N1, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # MLP: (B, N1, C) → Conv2d with kernel 1 → (B, C', 1, N1)
        new_points = new_points.permute(0, 2, 1).unsqueeze(2).contiguous()
        new_points = self.mlp_convs(new_points)
        new_points = new_points.squeeze(2)  # [B, C', N1]
        return new_points


# ============================================================================
# 分类模型: SSG (Single-Scale Grouping)
# ============================================================================

class PointNet2ClassificationSSG(nn.Module):
    """
    PointNet++ 分类网络 (SSG)，对齐原始论文设计。

    输入为 (B, N, 4) 的 xyzi 点云：
    - xyz（前3维）用于 FPS 采样和 ball query 分组（几何操作）
    - intensity（第4维）作为额外特征，在各 SA 模块中与归一化 xyz 拼接后送入 MLP
    强度信息不会丢失，它通过 points 特征流贯穿整个网络。
    """
    def __init__(self, num_class=26):
        super().__init__()
        # 额外特征通道数: 1 = 强度(intensity)
        in_channel = 1

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256, mlp=[256, 512, 1024], group_all=True
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    @staticmethod
    def _normalize_input_points(x):
        """
        统一输入为 (B, N, 4) 的 xyzi 格式:
        - 支持 (B, N, 4)、(B, N, 3) 及其转置形式 (B, 4, N)、(B, 3, N)
        - 仅 xyz 输入时自动补零强度通道
        """
        if x.ndim != 3:
            raise ValueError(f"PointNet2ClassificationSSG expects 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "PointNet2ClassificationSSG expects input layout (B,N,4)/(B,4,N) or xyz-only "
                f"(B,N,3)/(B,3,N), got {tuple(x.shape)}"
            )

        if points.shape[-1] == 3:
            pad_i = torch.zeros(points.shape[0], points.shape[1], 1, dtype=points.dtype, device=points.device)
            points = torch.cat([points, pad_i], dim=-1)

        return points

    def forward(self, x):
        """
        Args:
            x: [B, N, 4] 完整 xyzi 点云（也支持 (B, 4, N)、(B, N, 3)、(B, 3, N)）
        Returns:
            logits:   [B, num_class] — 类别 logits
            box_pred: [B, 6] — 包围盒预测 [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        x = self._normalize_input_points(x)              # 统一为 (B, N, 4)

        # PointNet++ 的标准数据流: xyz(3D) 用于几何采样/分组, points(特征) 在 MLP 中与归一化 xyz 拼接
        # 强度(intensity)作为 points 的第 1 维特征，在 SA 模块中与相对坐标(dxyz)拼接 → 共 4 维输入 MLP
        xyz = x[:, :, :3].transpose(1, 2).contiguous()   # [B, 3, N] — 仅用于 FPS / ball query
        points = x[:, :, 3:].transpose(1, 2).contiguous() # [B, 1, N] — 强度特征流

        l1_xyz, l1_points = self.sa1(xyz, points)        # SA1: 强度参与 MLP(4→64→64→128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # SA2: 特征流继续
        _, l3_points = self.sa3(l2_xyz, l2_points)       # SA3: 全局池化 → 1024 维

        x = l3_points.view(x.shape[0], 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        feat = self.drop2(F.relu(self.bn2(self.fc2(x))))

        logits = self.fc3(feat)
        box_pred = self.box_head(feat)
        return logits, box_pred


# ============================================================================
# 分类模型: MSG (Multi-Scale Grouping)
# ============================================================================

class PointNet2ClassificationMSG(nn.Module):
    """
    PointNet++ 分类网络 (MSG — Multi-Scale Grouping)。

    与 SSG 的区别在于前两层 SA 使用多半径 ball query 提取多尺度特征。
    参考官方实现 pointnet2_cls_msg.py。
    """
    def __init__(self, num_class=26):
        super().__init__()
        in_channel = 1  # 强度通道

        # SA1: 三层尺度 [0.1, 0.2, 0.4], 每组分别 [16, 32, 128] 个近邻
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channel=in_channel,
            mlp_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128],
            ],
        )
        # SA2: 尺度 [0.2, 0.4, 0.8]
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=128 + 64 + 128,  # = 320 (三个尺度的输出通道之和)
            mlp_list=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256],
            ],
        )
        # SA3: 全局池化 (SSG)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=128 + 256 + 256,  # = 640
            mlp=[256, 512, 1024], group_all=True
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    @staticmethod
    def _normalize_input_points(x):
        """统一输入为 (B, N, 4)，同 SSG。"""
        if x.ndim != 3:
            raise ValueError(f"PointNet2ClassificationMSG expects 3D input, got shape {tuple(x.shape)}")
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
            x: [B, N, 4] xyzi 点云
        Returns:
            logits: [B, num_class]
            box_pred: [B, 6]
        """
        x = self._normalize_input_points(x)
        xyz = x[:, :, :3].transpose(1, 2).contiguous()   # [B, 3, N]
        points = x[:, :, 3:].transpose(1, 2).contiguous() # [B, 1, N]

        l1_xyz, l1_points = self.sa1(xyz, points)         # MSG SA1
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # MSG SA2
        _, l3_points = self.sa3(l2_xyz, l2_points)        # 全局 SA3

        x = l3_points.view(x.shape[0], 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        feat = self.drop2(F.relu(self.bn2(self.fc2(x))))

        logits = self.fc3(feat)
        box_pred = self.box_head(feat)
        return logits, box_pred


# ============================================================================
# 统一 GPU 显存测试 (SKILL 规范)
# ============================================================================

def _run_gpu_memory_test(model_class, model_name, num_class=26):
    """
    对指定的模型类执行 GPU 显存压力测试。
    逐 batch size [4, 8, 16, 32] 扫查，打印峰值显存。
    """
    import gc
    if not torch.cuda.is_available():
        print(f"  无 CUDA，跳过 {model_name} 显存测试。")
        return

    N = 1024
    for bs in [4, 8, 16, 32]:
        try:
            m = model_class(num_class=num_class).cuda()
            pts = torch.randn(bs, N, 4).cuda()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            m.train()
            o = m(pts)
            loss = o[0].sum() + o[1].sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  {model_name} B={bs:2d}: peak {peak:6.0f} MB")
            del m, pts, o, loss
            torch.cuda.empty_cache()
            gc.collect()
        except torch.cuda.OutOfMemoryError:
            print(f"  {model_name} B={bs:2d}: OOM!")
            torch.cuda.empty_cache()
            gc.collect()
            break


def _quick_shape_test():
    """快速形状验证 + GPU 显存压力测试 (覆盖 SSG 与 MSG)。"""
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.randn(4, 1024, 4, device=device)

    # SSG 测试
    model_ssg = PointNet2ClassificationSSG(num_class=26).to(device)
    logits, box_pred = model_ssg(pts)
    print("PointNet2ClassificationSSG logits:", logits.shape)    # [4, 26]
    print("PointNet2ClassificationSSG box_pred:", box_pred.shape)  # [4, 6]

    # MSG 测试
    model_msg = PointNet2ClassificationMSG(num_class=26).to(device)
    logits, box_pred = model_msg(pts)
    print("PointNet2ClassificationMSG logits:", logits.shape)    # [4, 26]
    print("PointNet2ClassificationMSG box_pred:", box_pred.shape)  # [4, 6]

    # ══════════════════════════════════════════════
    # GPU 显存测试 (SKILL 规范)
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
        _run_gpu_memory_test(PointNet2ClassificationSSG, "SSG")
        _run_gpu_memory_test(PointNet2ClassificationMSG, "MSG")
    else:
        print("无 CUDA，跳过显存测试。")


if __name__ == "__main__":
    _quick_shape_test()
