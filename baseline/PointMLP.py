import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    src: [B, N, 3]
    dst: [B, M, 3]
    return: [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    points: [B, N, C]
    idx: [B, N, K] or [B, S]
    return: [B, N, K, C] or [B, S, C]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_idx = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def knn_point(k, xyz, new_xyz):
    """
    xyz: [B, N, 3]
    new_xyz: [B, S, 3]
    return idx: [B, S, k]
    """
    dist = square_distance(new_xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


class TransitionDown(nn.Module):
    """Downsampling block using FPS + kNN grouping + local max pooling."""
    def __init__(self, in_channels, out_channels, npoint, k=16):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def farthest_point_sample(xyz, npoint):
        device = xyz.device
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_idx = torch.arange(B, dtype=torch.long, device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]
        return centroids

    def forward(self, xyz_full, feat):
        """
        xyz_full: [B, N, 4]
        feat: [B, N, C]
        return:
          new_xyz_full: [B, S, 4]
          new_feat: [B, S, C_out]
        """
        xyz = xyz_full[:, :, :3].contiguous()
        fps_idx = self.farthest_point_sample(xyz, self.npoint)
        new_xyz_full = index_points(xyz_full, fps_idx)
        new_xyz = new_xyz_full[:, :, :3]

        idx = knn_point(self.k, xyz, new_xyz)
        grouped_xyz_full = index_points(xyz_full, idx)
        grouped_feat = index_points(feat, idx)

        rel_xyz_full = grouped_xyz_full - new_xyz_full.unsqueeze(2)
        x = torch.cat([grouped_feat, rel_xyz_full], dim=-1)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]
        new_feat = x.transpose(1, 2).contiguous()
        return new_xyz_full, new_feat


class LocalMLPBlock(nn.Module):
    """Local point-wise MLP block with kNN grouping and residual connection."""
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz_full, feat):
        """
        xyz_full: [B, N, 4]
        feat: [B, N, C]
        return: [B, N, C_out]
        """
        B, N, C = feat.shape
        xyz = xyz_full[:, :, :3].contiguous()
        idx = knn_point(self.k, xyz, xyz)
        grouped_xyz_full = index_points(xyz_full, idx)
        grouped_feat = index_points(feat, idx)

        rel_xyz_full = grouped_xyz_full - xyz_full.unsqueeze(2)
        x = torch.cat([grouped_feat, rel_xyz_full], dim=-1)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]
        x = x.transpose(1, 2).contiguous()

        if self.shortcut is not None:
            shortcut = self.shortcut(feat.reshape(B * N, C)).reshape(B, N, -1)
        else:
            shortcut = feat

        return self.relu(x + shortcut)


class PointMLPClassification(nn.Module):
    """PointMLP-style backbone for classification with an extra box head."""
    def __init__(self, num_classes=26, k=16):
        super().__init__()
        self.k = k

        self.stem = nn.Sequential(
            nn.Linear(4, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.block1 = LocalMLPBlock(64, 64, k=k)
        self.td1 = TransitionDown(64, 128, npoint=512, k=k)
        self.block2 = LocalMLPBlock(128, 128, k=k)
        self.td2 = TransitionDown(128, 256, npoint=128, k=k)
        self.block3 = LocalMLPBlock(256, 256, k=k)

        self.cls_head = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    @staticmethod
    def _normalize_input_points(x):
        """Normalize input to (B, N, 4)."""
        if x.ndim != 3:
            raise ValueError(f"PointMLPClassification expects 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "PointMLPClassification expects input layout (B,N,4)/(B,4,N) or xyz-only "
                f"(B,N,3)/(B,3,N), got {tuple(x.shape)}"
            )

        if points.shape[-1] == 3:
            pad_i = torch.zeros(points.shape[0], points.shape[1], 1, dtype=points.dtype, device=points.device)
            points = torch.cat([points, pad_i], dim=-1)

        return points

    def forward(self, x):
        """
        x: [B, N, 4] or [B, 4, N] (also supports xyz-only with 3 channels)
        return:
          logits: [B, num_classes]
          box_pred: [B, 6]
        """
        x = self._normalize_input_points(x)
        xyz_full = x
        B, N, _ = x.shape

        feat = self.stem(x.reshape(B * N, 4)).reshape(B, N, 64)
        feat = self.block1(xyz_full, feat)

        x1, f1 = self.td1(xyz_full, feat)
        f1 = self.block2(x1, f1)

        x2, f2 = self.td2(x1, f1)
        f2 = self.block3(x2, f2)

        g = torch.max(f2, dim=1)[0]
        logits = self.cls_head(g)
        box_pred = self.box_head(g)
        return logits, box_pred


def _quick_shape_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointMLPClassification(num_classes=26, k=16).to(device)
    pts = torch.randn(4, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print("PointMLP logits:", logits.shape)
    print("PointMLP box_pred:", box_pred.shape)


if __name__ == "__main__":
    _quick_shape_test()
