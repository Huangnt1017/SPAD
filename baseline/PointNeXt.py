import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    计算两组点的平方距离。

    Args:
        src: [B, N, 3]
        dst: [B, M, 3]

    Returns:
        dist: [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    按索引采样点。

    Args:
        points: [B, N, C]
        idx: [B, S] 或 [B, S, K]

    Returns:
        sampled: [B, S, C] 或 [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    kNN 邻域索引。

    Args:
        k: 邻居数
        xyz: [B, N, 3]
        new_xyz: [B, S, 3]

    Returns:
        idx: [B, S, k]
    """
    dist = square_distance(new_xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    FPS 采样，返回采样点索引。

    Args:
        xyz: [B, N, 3]
        npoint: 采样点数

    Returns:
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


class LocalAggregation(nn.Module):
    """PointNeXt 风格的局部聚合：kNN 分组 + 相对位置编码 + 最大池化。"""
    def __init__(self, channels: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, xyz_full: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz_full: [B, N, 4] 完整 xyzi
            feat: [B, N, C]

        Returns:
            聚合后特征 [B, N, C]
        """
        xyz = xyz_full[:, :, :3].contiguous()
        idx = knn_point(self.k, xyz, xyz)
        grouped_xyz_full = index_points(xyz_full, idx)  # [B, N, k, 4]
        grouped_feat = index_points(feat, idx)          # [B, N, k, C]

        rel = grouped_xyz_full - xyz_full.unsqueeze(2)  # [B, N, k, 4]
        x = torch.cat([grouped_feat, rel], dim=-1)      # [B, N, k, C+4]
        x = x.permute(0, 3, 2, 1).contiguous()          # [B, C+4, k, N]
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]                      # [B, C, N]
        return x.permute(0, 2, 1).contiguous()          # [B, N, C]


class PointNeXtBlock(nn.Module):
    """PointNeXt 基本块：扩展 MLP + 局部聚合 + 残差。"""
    def __init__(self, channels: int, k: int = 16, expansion: int = 2):
        super().__init__()
        hidden = int(channels * expansion)
        self.fc1 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.local = LocalAggregation(hidden, k=k)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, channels, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz_full: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz_full: [B, N, 4]
            feat: [B, N, C]

        Returns:
            [B, N, C]
        """
        B, N, C = feat.shape
        x = self.fc1(feat.reshape(B * N, C)).reshape(B, N, -1)
        x = self.local(xyz_full, x)
        x = self.fc2(x.reshape(B * N, -1)).reshape(B, N, C)
        return self.act(x + feat)


class PointNeXtDown(nn.Module):
    """PointNeXt 下采样：FPS + kNN 分组 + 局部聚合。"""
    def __init__(self, in_channels: int, out_channels: int, npoint: int, k: int = 16):
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

    def forward(self, xyz_full: torch.Tensor, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz_full: [B, N, 4]
            feat: [B, N, C]

        Returns:
            new_xyz_full: [B, S, 4]
            new_feat: [B, S, C_out]
        """
        xyz = xyz_full[:, :, :3].contiguous()
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz_full = index_points(xyz_full, fps_idx)
        new_xyz = new_xyz_full[:, :, :3]

        idx = knn_point(self.k, xyz, new_xyz)
        grouped_xyz_full = index_points(xyz_full, idx)
        grouped_feat = index_points(feat, idx)

        rel = grouped_xyz_full - new_xyz_full.unsqueeze(2)  # [B, S, k, 4]
        x = torch.cat([grouped_feat, rel], dim=-1)          # [B, S, k, C+4]
        x = x.permute(0, 3, 2, 1).contiguous()              # [B, C+4, k, S]
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]                          # [B, C_out, S]
        new_feat = x.transpose(1, 2).contiguous()           # [B, S, C_out]
        return new_xyz_full, new_feat


class PointNeXtClassification(nn.Module):
    """PointNeXt 风格分类模型，输出类别与 3D box 预测。"""
    def __init__(self, num_classes: int = 26, k: int = 16):
        super().__init__()
        self.k = k

        self.stem = nn.Sequential(
            nn.Linear(4, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.ModuleList([
            PointNeXtBlock(64, k=k, expansion=2),
            PointNeXtBlock(64, k=k, expansion=2),
        ])
        self.down1 = PointNeXtDown(64, 128, npoint=512, k=k)

        self.stage2 = nn.ModuleList([
            PointNeXtBlock(128, k=k, expansion=2),
            PointNeXtBlock(128, k=k, expansion=2),
        ])
        self.down2 = PointNeXtDown(128, 256, npoint=128, k=k)

        self.stage3 = nn.ModuleList([
            PointNeXtBlock(256, k=k, expansion=2),
            PointNeXtBlock(256, k=k, expansion=2),
        ])

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
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

    @staticmethod
    def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
        """
        统一输入为 (B, N, 4)：支持 (B, N, 4)/(B, 4, N) 以及 xyz-only 的 (B, N, 3)/(B, 3, N)。
        """
        if x.ndim != 3:
            raise ValueError(f"PointNeXtClassification expects 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "PointNeXtClassification expects input layout (B,N,4)/(B,4,N) or xyz-only "
                f"(B,N,3)/(B,3,N), got {tuple(x.shape)}"
            )

        if points.shape[-1] == 3:
            pad_i = torch.zeros(points.shape[0], points.shape[1], 1, dtype=points.dtype, device=points.device)
            points = torch.cat([points, pad_i], dim=-1)

        return points

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, 4] 或 [B, 4, N]（支持 xyz-only）

        Returns:
            logits: [B, num_classes]
            box_pred: [B, 6]
        """
        x = self._normalize_input_points(x)
        B, N, _ = x.shape
        xyz_full = x

        feat = self.stem(x.reshape(B * N, 4)).reshape(B, N, 64)
        for block in self.stage1:
            feat = block(xyz_full, feat)

        xyz1, feat1 = self.down1(xyz_full, feat)
        for block in self.stage2:
            feat1 = block(xyz1, feat1)

        xyz2, feat2 = self.down2(xyz1, feat1)
        for block in self.stage3:
            feat2 = block(xyz2, feat2)

        global_feat = torch.max(feat2, dim=1)[0]
        logits = self.cls_head(global_feat)
        box_pred = self.box_head(global_feat)
        return logits, box_pred


def _quick_shape_test() -> None:
    """快速形状验证 + GPU 显存压力测试。"""
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNeXtClassification(num_classes=26, k=16).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print("PointNeXt logits:", logits.shape)
    print("PointNeXt box_pred:", box_pred.shape)

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
                m = PointNeXtClassification(num_classes=26, k=16).cuda()
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
