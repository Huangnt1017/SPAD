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
    Point Transformer 下采样模块:
    FPS + kNN 分组 + 局部最大池化（PointNet 风格）
    - FPS 和 kNN 仅在 xyz 坐标（前3维）上进行，保持几何合理性
    - 相对位置编码使用完整 4 维（dxyz + dintensity），让强度参与空间-特征融合
    """
    def __init__(self, in_channels, out_channels, npoint, k=16):
        super().__init__()
        self.npoint = npoint
        self.k = k
        # in_channels + 4: 特征通道 + 相对位置(dx,dy,dz,dintensity)
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
        Args:
            xyz_full: [B, N, 4] — 完整 xyzi 点云（xyz + intensity）
            feat:     [B, N, C] — 点特征
        Returns:
            new_xyz_full: [B, S, 4] — 下采样后的 xyzi 点
            new_feat:     [B, S, C_out] — 下采样后的特征
        """
        # 仅用 xyz 坐标（前3维）做 FPS 和 kNN，保证几何合理性
        xyz = xyz_full[:, :, :3].contiguous()               # [B, N, 3]

        fps_idx = self.farthest_point_sample(xyz, self.npoint)  # [B, S]
        new_xyz_full = index_points(xyz_full, fps_idx)          # [B, S, 4] — 完整 xyzi
        new_xyz = new_xyz_full[:, :, :3]                        # [B, S, 3] — 仅用于 kNN

        idx = knn_point(self.k, xyz, new_xyz)                   # [B, S, K]
        grouped_xyz_full = index_points(xyz_full, idx)          # [B, S, K, 4] — 完整 xyzi
        grouped_feat = index_points(feat, idx)                  # [B, S, K, C]

        # 4 维相对位置: dxyz + dintensity，让强度参与局部几何编码
        rel_xyz_full = grouped_xyz_full - new_xyz_full.unsqueeze(2)  # [B, S, K, 4]
        x = torch.cat([grouped_feat, rel_xyz_full], dim=-1)          # [B, S, K, C+4]
        x = x.permute(0, 3, 2, 1).contiguous()                       # [B, C+4, K, S]
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]                                   # [B, C_out, S]
        new_feat = x.transpose(1, 2).contiguous()                    # [B, S, C_out]
        return new_xyz_full, new_feat


class PointTransformerLayer(nn.Module):
    """
    Point Transformer 层 (ICCV 2021):
    x'_i = sum_j softmax(gamma(phi(x_i)-psi(x_j)+delta_ij)) * (alpha(x_j)+delta_ij)
    - kNN 仅在 xyz 坐标上进行（保留原文几何定义）
    - 位置编码 delta 使用 4 维相对特征（dxyz + dintensity），让强度参与注意力计算
    """
    def __init__(self, channels, k=16):
        super().__init__()
        self.k = k
        self.channels = channels
        self.mid = channels

        self.to_q = nn.Linear(channels, self.mid, bias=False)
        self.to_k = nn.Linear(channels, self.mid, bias=False)
        self.to_v = nn.Linear(channels, self.mid, bias=False)

        # 位置编码: 4 维相对特征 (dx, dy, dz, dintensity) → 通道维
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, channels),
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

    def forward(self, xyz_full, feat):
        """
        Args:
            xyz_full: [B, N, 4] — 完整 xyzi 点云
            feat:     [B, N, C] — 点特征
        Returns:
            [B, N, C] — 更新后的点特征
        """
        B, N, C = feat.shape
        # 仅用 xyz 坐标（前3维）做 kNN 近邻搜索
        xyz = xyz_full[:, :, :3].contiguous()               # [B, N, 3]
        idx = knn_point(self.k, xyz, xyz)                   # [B, N, k]
        neighbor_xyz_full = index_points(xyz_full, idx)      # [B, N, k, 4] — 完整 xyzi
        neighbor_feat = index_points(feat, idx)              # [B, N, k, C]

        q = self.to_q(feat).unsqueeze(2)                     # [B, N, 1, C]
        k = self.to_k(neighbor_feat)                         # [B, N, k, C]
        v = self.to_v(neighbor_feat)                         # [B, N, k, C]

        # 4 维相对位置编码: dxyz + dintensity
        rel_pos_full = neighbor_xyz_full - xyz_full.unsqueeze(2)  # [B, N, k, 4]
        pos = self.pos_mlp(rel_pos_full.reshape(B * N * self.k, 4)).reshape(B, N, self.k, C)

        attn = q - k + pos                                   # [B, N, k, C]
        attn = self.attn_mlp(attn.reshape(B * N * self.k, C)).reshape(B, N, self.k, C)
        attn = F.softmax(attn, dim=2)                        # 向量注意力在近邻维度上

        out = torch.sum(attn * (v + pos), dim=2)             # [B, N, C]
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

    def forward(self, xyz_full, feat):
        """
        Args:
            xyz_full: [B, N, 4] — 完整 xyzi 点云
            feat:     [B, N, C] — 点特征
        Returns:
            [B, N, C] — 残差连接后的特征
        """
        B, N, C = feat.shape
        x = self.fc1(feat.reshape(B * N, C)).reshape(B, N, C)
        x = self.pt(xyz_full, x)
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

        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    @staticmethod
    def _normalize_input_points(x):
        """
        统一输入为 (B, N, 4):
        - 支持 (B, N, 4) 与 (B, 4, N)
        - 支持仅 xyz 的 (B, N, 3)/(B, 3, N)，自动补一维 0 强度
        """
        if x.ndim != 3:
            raise ValueError(f"PointTransformerClassification expects 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "PointTransformerClassification expects input layout (B,N,4)/(B,4,N) or xyz-only "
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
            logits:   [B, num_classes] — 类别 logits
            box_pred: [B, 6] — 包围盒预测 [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        x = self._normalize_input_points(x)              # 统一为 (B, N, 4)

        # 完整 xyzi 作为几何+强度联合表示贯穿整个网络
        # 各模块内部仅用前3维做 kNN/FPS，位置编码使用完整 4 维
        xyz_full = x                                      # [B, N, 4]
        B, N, _ = x.shape

        # stem: xyzi(4) → 32 维初始特征
        feat = self.stem(x.reshape(B * N, 4)).reshape(B, N, 32)
        feat = self.block1(xyz_full, feat)

        x1, f1 = self.td1(xyz_full, feat)                 # x1: [B, 512, 4], f1: [B, 512, 64]
        f1 = self.block2(x1, f1)

        x2, f2 = self.td2(x1, f1)                         # x2: [B, 128, 4], f2: [B, 128, 128]
        f2 = self.block3(x2, f2)

        x3, f3 = self.td3(x2, f2)                         # x3: [B, 32, 4], f3: [B, 32, 256]
        f3 = self.block4(x3, f3)

        g = torch.max(f3, dim=1)[0]                       # 全局最大池化 → [B, 256]
        logits = self.cls_head(g)
        box_pred = self.box_head(g)
        return logits, box_pred


def _quick_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_model = PointTransformerClassification(num_classes=26, k=16).to(device)

    pts = torch.randn(2, 1024, 4, device=device)

    cls_logits, box_pred = cls_model(pts)

    print("cls:", cls_logits.shape)  # [2, 26]
    print("box:", box_pred.shape)    # [2, 6]


if __name__ == "__main__":
    _quick_test()