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


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Interpolate features from xyz2 (sparser) to xyz1 (denser)

        xyz1: [B, 3, N]
        xyz2: [B, 3, S]
        points1: [B, D1, N] or None
        points2: [B, D2, S]
        Return:
            new_points: [B, D', N]
        """
        xyz1 = xyz1.transpose(1, 2).contiguous()  # [B, N, 3]
        xyz2 = xyz2.transpose(1, 2).contiguous()  # [B, S, 3]
        points2 = points2.transpose(1, 2).contiguous()  # [B, S, D2]

        B, N, _ = xyz1.shape
        S = xyz2.shape[1]

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)              # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]     # 3-NN

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight[:, :, :, None], dim=2
            )  # [B, N, D2]

        if points1 is not None:
            points1 = points1.transpose(1, 2).contiguous()  # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(1, 2).contiguous()  # [B, D, N]
        new_points = self.mlp(new_points)
        return new_points


class PointNet2ClassificationSSG(nn.Module):
    """
    PointNet++ classification network (SSG), aligned to original paper style.
    """
    def __init__(self, num_class, normal_channel=False):
        super().__init__()
        in_channel = 3 if normal_channel else 0

        self.normal_channel = normal_channel
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

    def forward(self, x):
        """
        x:
          - if normal_channel=False: [B, 3, N]
          - if normal_channel=True:  [B, 6, N] (xyz + normal)
        return:
          logits: [B, num_class]
        """
        xyz = x[:, :3, :]
        points = x[:, 3:, :] if self.normal_channel else None

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(x.shape[0], 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class PointNet2PartSegmentationSSG(nn.Module):
    """
    PointNet++ part segmentation network (SSG), aligned to original paper style.
    Supports optional class one-hot vector as in ShapeNet part setup.
    """
    def __init__(self, num_part, num_class=16, normal_channel=False, use_cls_label=True):
        super().__init__()
        self.normal_channel = normal_channel
        self.use_cls_label = use_cls_label

        in_channel = 3 if normal_channel else 0

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

        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])

        # L0 concatenation: xyz(3) + optional normal(3) + optional cls_onehot(num_class) + fp feature(128)
        fp1_in = 128 + 3
        if normal_channel:
            fp1_in += 3
        if use_cls_label:
            fp1_in += num_class

        self.fp1 = PointNetFeaturePropagation(in_channel=fp1_in, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, x, cls_label=None):
        """
        x:
          - if normal_channel=False: [B, 3, N]
          - if normal_channel=True:  [B, 6, N]
        cls_label: [B, num_class] one-hot or None
        return:
          seg_logits: [B, num_part, N]
        """
        B, _, N = x.shape
        l0_xyz = x[:, :3, :]
        l0_points = x[:, 3:, :] if self.normal_channel else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_cat = [l0_xyz]
        if self.normal_channel:
            l0_cat.append(x[:, 3:, :])

        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("cls_label is required when use_cls_label=True")
            cls_label_feat = cls_label.view(B, -1, 1).repeat(1, 1, N)
            l0_cat.append(cls_label_feat)

        l0_cat.append(l1_points)
        l0_points = torch.cat(l0_cat, dim=1)

        feat = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        feat = self.drop1(F.relu(self.bn1(self.conv1(feat))))
        seg_logits = self.conv2(feat)
        return seg_logits
