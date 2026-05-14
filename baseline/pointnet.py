"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

GitHub:  https://github.com/fxia22/pointnet.pytorch
Local:   D:\essay\3d目标检测复现仓库\pointnet.pytorch-master

完整复现 (Charles R. Qi):
- 输入变换 (STN3d): 预测 3x3 旋转矩阵对齐点云
- 特征变换 (STNkd): 对 64 维特征做对齐 (可选)
- PointNetfeat: 逐点 MLP(64→128→1024) + 全局最大池化
- PointNetCls: 全局特征 → FC(512→256→K) 分类
- 输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])

Reference:
@InProceedings{Qi_2017_CVPR,
  author = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
  title = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}
"""

import os
import sys
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ============================================================================
# 输入变换网络 (STN3d): 预测 3x3 仿射变换对齐输入点云
# ============================================================================

class STN3d(nn.Module):
    """Spatial Transformer Network (3D): 预测 3x3 旋转矩阵。

    架构: Conv1d(3→64→128→1024) → MaxPool → FC(1024→512→256→9)
    输出加单位矩阵初始化保证稳定训练。
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) — 输入点云坐标
        Returns:
            trans: (B, 3, 3) — 变换矩阵
        """
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 初始化为单位矩阵
        iden = (
            torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32))
            .view(1, 9).repeat(batchsize, 1).to(x.device)
        )
        x = x + iden
        return x.view(-1, 3, 3)


# ============================================================================
# 特征变换网络 (STNkd): 对 k 维特征做对齐
# ============================================================================

class STNkd(nn.Module):
    """Feature Transformer: 预测 k×k 变换矩阵对齐特征空间。"""
    def __init__(self, k: int = 64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Args:
            x: (B, k, N) — 输入特征
        Returns:
            trans: (B, k, k) — 变换矩阵
        """
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            torch.from_numpy(np.eye(self.k, dtype=np.float32).flatten())
            .view(1, self.k * self.k).repeat(batchsize, 1).to(x.device)
        )
        x = x + iden
        return x.view(-1, self.k, self.k)


# ============================================================================
# PointNet 特征提取器
# ============================================================================

class PointNetfeat(nn.Module):
    """PointNet 特征提取器 — 逐点 MLP + 全局最大池化。

    包含可选的输入变换 (STN3d) 和特征变换 (STNkd)。
    global_feat=True: 只返回全局特征 (B, 1024)
    global_feat=False: 返回全局+局部拼接特征 (B, 1088, N)
    """
    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) — 输入点云 (xyz)
        Returns:
            x: 全局特征 (B, 1024) 或 逐点拼接特征 (B, 1088, N)
            trans: (B, 3, 3) — 输入变换矩阵
            trans_feat: (B, 64, 64) 或 None — 特征变换矩阵
        """
        n_pts = x.size(2)

        # 输入变换
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        # MLP: 3→64
        x = F.relu(self.bn1(self.conv1(x)))

        # 特征变换
        trans_feat = None
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        pointfeat = x  # (B, 64, N) — 用于拼接的逐点特征

        # MLP: 64→128→1024
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # 全局最大池化
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # (B, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# ============================================================================
# PointNet 分类模型 (带 3D BBox 头)
# ============================================================================

class PointNetCls(nn.Module):
    """PointNet 分类 + 3D BBox 模型 (适配 SPAD 训练管道)。

    基于 PointNet 官方实现:
    - 输入变换 (STN3d): 对齐点云
    - 特征变换 (STNkd): 对齐 64 维特征 (默认启用)
    - 逐点 MLP: 3→64→128→1024
    - 全局最大池化
    - 分类头: FC(1024→512→256→C)
    - BBox 头: FC(1024→128→6)

    Input:  (B, N, 4) xyzi
    Output: (logits [B, C], box_pred [B, 6])
    """
    def __init__(self, num_classes: int = 26, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)

        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.box_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    @staticmethod
    def _normalize_input_points(x):
        """统一输入为 (B, N, 4) xyzi。"""
        if x.ndim != 3:
            raise ValueError(f"PointNetCls expects 3D input, got {tuple(x.shape)}")
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
        B, N, _ = x.shape

        # PointNet 只使用 xyz 坐标 (前3维)
        # 格式转换: (B, N, 3) → (B, 3, N)
        pts = x[:, :, :3].transpose(1, 2).contiguous()

        global_feat, _, _ = self.feat(pts)  # (B, 1024)

        logits = self.cls_head(global_feat)
        box_pred = self.box_head(global_feat)
        return logits, box_pred


# ============================================================================
# PointNet 密集分割模型 (保留但不注册)
# ============================================================================

class PointNetDenseCls(nn.Module):
    """PointNet 逐点分割模型。

    拼接全局(1024) + 局部(64)=1088 维逐点特征 → MLP(1088→512→256→128→K)。
    """
    def __init__(self, k: int = 50, feature_transform: bool = False):
        super().__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) — 点云坐标
        Returns:
            x: (B, N, K) — 逐点分类 logits
            trans: (B, 3, 3)
            trans_feat: optional
        """
        batchsize = x.size(0)
        n_pts = x.size(2)
        pointfeat, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(pointfeat)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        return x, trans, trans_feat


# ============================================================================
# 特征变换正则化损失
# ============================================================================

def feature_transform_regularizer(trans):
    """计算特征变换矩阵的正则化损失: ||T T^T - I||_F。

    鼓励变换矩阵尽量接近正交矩阵。
    """
    d = trans.size(1)
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


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
            m = PointNetCls(num_classes=26).cuda()
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
    print(f"Testing PointNet on {device}")

    model = PointNetCls(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Box:    {tuple(box_pred.shape)}")
    print("✓ PointNet works correctly")


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()