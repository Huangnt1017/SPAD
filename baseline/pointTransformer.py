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
    dist = square_distance(new_xyz, xyz)  # [B, S, N]
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


class TransitionDown(nn.Module):
    """
    Point Transformer downsampling block:
    FPS + kNN grouping + local max pooling (PointNet-style)
    """
    def __init__(self, in_channels, out_channels, npoint, k=16):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 3, out_channels, 1, bias=False),
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

    def forward(self, xyz, feat):
        """
        xyz: [B, N, 3]
        feat: [B, N, C]
        return:
          new_xyz: [B, S, 3]
          new_feat: [B, S, C_out]
        """
        fps_idx = self.farthest_point_sample(xyz, self.npoint)  # [B, S]
        new_xyz = index_points(xyz, fps_idx)                    # [B, S, 3]

        idx = knn_point(self.k, xyz, new_xyz)                   # [B, S, K]
        grouped_xyz = index_points(xyz, idx)                    # [B, S, K, 3]
        grouped_feat = index_points(feat, idx)                  # [B, S, K, C]

        rel_xyz = grouped_xyz - new_xyz.unsqueeze(2)            # [B, S, K, 3]
        x = torch.cat([grouped_feat, rel_xyz], dim=-1)          # [B, S, K, C+3]
        x = x.permute(0, 3, 2, 1).contiguous()                  # [B, C+3, K, S]
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]                              # [B, C_out, S]
        new_feat = x.transpose(1, 2).contiguous()               # [B, S, C_out]
        return new_xyz, new_feat


class PointTransformerLayer(nn.Module):
    """
    Point Transformer layer (ICCV 2021):
    x'_i = sum_j softmax(gamma(phi(x_i)-psi(x_j)+delta_ij)) * (alpha(x_j)+delta_ij)
    """
    def __init__(self, channels, k=16):
        super().__init__()
        self.k = k
        self.channels = channels
        self.mid = channels

        self.to_q = nn.Linear(channels, self.mid, bias=False)
        self.to_k = nn.Linear(channels, self.mid, bias=False)
        self.to_v = nn.Linear(channels, self.mid, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

        self.out_linear = nn.Linear(channels, channels, bias=False)

    def forward(self, xyz, feat):
        """
        xyz: [B, N, 3]
        feat: [B, N, C]
        return: [B, N, C]
        """
        B, N, C = feat.shape
        idx = knn_point(self.k, xyz, xyz)                  # [B, N, k]
        neighbor_xyz = index_points(xyz, idx)              # [B, N, k, 3]
        neighbor_feat = index_points(feat, idx)            # [B, N, k, C]

        q = self.to_q(feat).unsqueeze(2)                   # [B, N, 1, C]
        k = self.to_k(neighbor_feat)                       # [B, N, k, C]
        v = self.to_v(neighbor_feat)                       # [B, N, k, C]

        rel_pos = neighbor_xyz - xyz.unsqueeze(2)          # [B, N, k, 3]
        pos = self.pos_mlp(rel_pos.reshape(B * N * self.k, 3)).reshape(B, N, self.k, C)

        attn = q - k + pos                                 # [B, N, k, C]
        attn = self.attn_mlp(attn.reshape(B * N * self.k, C)).reshape(B, N, self.k, C)
        attn = F.softmax(attn, dim=2)                      # vector attention on neighbors

        out = torch.sum(attn * (v + pos), dim=2)           # [B, N, C]
        out = self.out_linear(out)
        return out


class PTBlock(nn.Module):
    def __init__(self, channels, k=16):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.pt = PointTransformerLayer(channels, k=k)
        self.fc2 = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz, feat):
        """
        xyz: [B, N, 3]
        feat: [B, N, C]
        """
        B, N, C = feat.shape
        x = self.fc1(feat.reshape(B * N, C)).reshape(B, N, C)
        x = self.pt(xyz, x)
        x = self.fc2(x.reshape(B * N, C)).reshape(B, N, C)
        return self.relu(x + feat)


class PointTransformerClassification(nn.Module):
    def __init__(self, num_classes=26, k=16):
        super().__init__()
        self.k = k

        self.stem = nn.Sequential(
            nn.Linear(4, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = PTBlock(32, k)
        self.td1 = TransitionDown(32, 64, npoint=512, k=k)
        self.block2 = PTBlock(64, k)
        self.td2 = TransitionDown(64, 128, npoint=128, k=k)
        self.block3 = PTBlock(128, k)
        self.td3 = TransitionDown(128, 256, npoint=32, k=k)
        self.block4 = PTBlock(256, k)

        self.cls_head = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        x: [B, N, 4]
        """
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError(f"PointTransformerClassification expects input shape (B, N, 4), got {tuple(x.shape)}")

        xyz = x[:, :, :3].contiguous()                     # [B, N, 3]
        B, N, _ = x.shape

        feat = self.stem(x.reshape(B * N, 4)).reshape(B, N, 32)
        feat = self.block1(xyz, feat)

        x1, f1 = self.td1(xyz, feat)
        f1 = self.block2(x1, f1)

        x2, f2 = self.td2(x1, f1)
        f2 = self.block3(x2, f2)

        x3, f3 = self.td3(x2, f2)
        f3 = self.block4(x3, f3)

        g = torch.max(f3, dim=1)[0]                        # global max pool
        out = self.cls_head(g)
        return out


def _quick_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_model = PointTransformerClassification(num_classes=26, k=16).to(device)

    pts = torch.randn(2, 1024, 4, device=device)

    cls_out = cls_model(pts)

    print("cls:", cls_out.shape)  # [2, 26]


if __name__ == "__main__":
    _quick_test()