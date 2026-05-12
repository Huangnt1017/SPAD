import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """计算两组点的平方距离。"""
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按索引采样点。"""
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """kNN 邻域索引。"""
    dist = square_distance(new_xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS 采样，返回采样点索引。"""
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


class TNPCPatchEmbed(nn.Module):
    """TNPC 风格的点云 token 化模块：FPS + kNN + 局部聚合。"""
    def __init__(self, embed_dim: int = 256, num_tokens: int = 256, k: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.k = k

        self.mlp = nn.Sequential(
            nn.Conv2d(5, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz_full: [B, N, 4] 完整 xyzi

        Returns:
            token_xyz_full: [B, S, 4]
            token_feat: [B, S, C]
        """
        B, N, _ = xyz_full.shape
        npoint = min(self.num_tokens, N)
        k = min(self.k, N)

        xyz = xyz_full[:, :, :3].contiguous()
        fps_idx = farthest_point_sample(xyz, npoint)
        token_xyz_full = index_points(xyz_full, fps_idx)
        token_xyz = token_xyz_full[:, :, :3]

        idx = knn_point(k, xyz, token_xyz)
        grouped_xyz_full = index_points(xyz_full, idx)  # [B, S, k, 4]

        rel = grouped_xyz_full - token_xyz_full.unsqueeze(2)
        intensity = grouped_xyz_full[..., 3:4]
        local_feat = torch.cat([rel, intensity], dim=-1)  # [B, S, k, 5]

        x = local_feat.permute(0, 3, 2, 1).contiguous()  # [B, 5, k, S]
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]  # [B, C, S]
        token_feat = x.transpose(1, 2).contiguous()
        return token_xyz_full, token_feat


class TNPCClassification(nn.Module):
    """TNPC 模型：Transformer 编码 + 分类/框回归头。"""
    def __init__(
        self,
        num_classes: int = 26,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        num_tokens: int = 256,
        k: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = TNPCPatchEmbed(embed_dim=embed_dim, num_tokens=num_tokens, k=k)

        self.pos_mlp = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.box_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6),
        )

    @staticmethod
    def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
        """
        统一输入为 (B, N, 4)：支持 (B, N, 4)/(B, 4, N) 以及 xyz-only 的 (B, N, 3)/(B, 3, N)。
        """
        if x.ndim != 3:
            raise ValueError(f"TNPCClassification expects 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "TNPCClassification expects input layout (B,N,4)/(B,4,N) or xyz-only "
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
        token_xyz, token_feat = self.tokenizer(x)

        B, S, C = token_feat.shape
        pos = self.pos_mlp(token_xyz.reshape(B * S, 4)).reshape(B, S, C)
        tokens = token_feat + pos

        encoded = self.encoder(tokens)
        pooled_max = torch.max(encoded, dim=1)[0]
        pooled_mean = torch.mean(encoded, dim=1)
        global_feat = torch.cat([pooled_max, pooled_mean], dim=1)

        logits = self.cls_head(global_feat)
        box_pred = self.box_head(global_feat)
        return logits, box_pred


def _quick_shape_test() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TNPCClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print("TNPC logits:", logits.shape)
    print("TNPC box_pred:", box_pred.shape)


if __name__ == "__main__":
    _quick_shape_test()
