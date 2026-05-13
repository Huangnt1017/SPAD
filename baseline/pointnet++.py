"""
@article{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclid squared distance between each two points.

    src: [B, N, C]
    dst: [B, M, C]
    return: [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    points: [B, N, C]
    idx: [B, S] or [B, S, K]
    return:
        [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """
    xyz: [B, N, 3]
    return:
        centroids: [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: float
        nsample: int
        xyz: [B, N, 3]
        new_xyz: [B, S, 3]
    Return:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N  # invalid index sentinel

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    xyz: [B, N, 3]
    points: [B, N, D] or None
    Return:
        new_xyz: [B, npoint, 3]
        new_points: [B, npoint, nsample, 3 + D]
    """
    fps_idx = farthest_point_sample(xyz, npoint)      # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)              # [B, npoint, 3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)              # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]

    if points is not None:
        grouped_points = index_points(points, idx)    # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    xyz: [B, N, 3]
    points: [B, N, D] or None
    Return:
        new_xyz: [B, 1, 3]
        new_points: [B, 1, N, 3 + D]
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


def _quick_shape_test():
    """快速形状验证 + GPU 显存压力测试。"""
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNet2ClassificationSSG(num_class=26).to(device)
    pts = torch.randn(4, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print("PointNet2ClassificationSSG logits:", logits.shape)   # [4, 26]
    print("PointNet2ClassificationSSG box_pred:", box_pred.shape)  # [4, 6]

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
                m = PointNet2ClassificationSSG(num_class=26).cuda()
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
    else:
        print("无 CUDA，跳过。")


if __name__ == "__main__":
    _quick_shape_test()
