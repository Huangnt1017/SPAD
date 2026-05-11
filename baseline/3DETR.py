import torch
import torch.nn as nn


def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
    """Normalize input to (B, N, 4).

    Supports (B, N, 4), (B, 4, N), and xyz-only (B, N, 3)/(B, 3, N).
    """
    if x.ndim != 3:
        raise ValueError(f"ThreeDETRClassification expects 3D input, got shape {tuple(x.shape)}")

    if x.shape[-1] in (3, 4):
        points = x
    elif x.shape[1] in (3, 4):
        points = x.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            "ThreeDETRClassification expects input layout (B,N,4)/(B,4,N) or xyz-only "
            f"(B,N,3)/(B,3,N), got {tuple(x.shape)}"
        )

    if points.shape[-1] == 3:
        pad_i = torch.zeros(points.shape[0], points.shape[1], 1, dtype=points.dtype, device=points.device)
        points = torch.cat([points, pad_i], dim=-1)

    return points


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: [B, N, 3]
    dst: [B, M, 3]
    return: [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, C]
    idx: [B, S] or [B, S, K]
    return: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
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


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    radius: float
    nsample: int
    xyz: [B, N, 3]
    new_xyz: [B, S, 3]
    return:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor | None):
    """
    xyz: [B, N, 3]
    points: [B, N, D] or None
    return:
        new_xyz: [B, npoint, 3]
        new_points: [B, npoint, nsample, 3 + D]
    """
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor | None):
    """
    xyz: [B, N, 3]
    points: [B, N, D] or None
    return:
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
    """PointNet++ set abstraction used as 3DETR backbone."""

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

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """
        xyz: [B, 3, N]
        points: [B, D, N] or None
        return:
            new_xyz: [B, 3, S]
            new_points: [B, D', S]
        """
        xyz = xyz.transpose(1, 2).contiguous()
        if points is not None:
            points = points.transpose(1, 2).contiguous()

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]

        new_xyz = new_xyz.transpose(1, 2).contiguous()
        return new_xyz, new_points


class ThreeDETRClassification(nn.Module):
    """3DETR-style backbone with transformer encoder-decoder and single-box output."""

    def __init__(
        self,
        num_classes: int = 26,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_queries: int = 16,
        sa_npoint_1: int = 512,
        sa_npoint_2: int = 128,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)

        # PointNet++ style backbone
        self.sa1 = PointNetSetAbstraction(
            npoint=sa_npoint_1,
            radius=0.2,
            nsample=32,
            in_channel=1,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=sa_npoint_2,
            radius=0.4,
            nsample=64,
            in_channel=128,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.input_proj = nn.Linear(256, d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, d_model))

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_classes),
        )

        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 6),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, N, 4] or [B, 4, N] (also supports xyz-only with 3 channels)
        Returns:
            logits: [B, num_classes]
            box_pred: [B, 6]
        """
        points = _normalize_input_points(x)
        xyz = points[:, :, :3]
        intensity = points[:, :, 3:].transpose(1, 2).contiguous()

        l1_xyz, l1_points = self.sa1(xyz.transpose(1, 2).contiguous(), intensity)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        tokens = l2_points.transpose(1, 2).contiguous()
        pos = self.pos_mlp(l2_xyz.transpose(1, 2).contiguous())

        memory = self.encoder(self.input_proj(tokens) + pos)

        query = self.query_embed.unsqueeze(0).repeat(tokens.size(0), 1, 1)
        tgt = torch.zeros_like(query)
        hs = self.decoder(tgt, memory)

        # For SPAD single-object supervision, pool queries into one feature.
        pooled = torch.max(hs, dim=1)[0]
        logits = self.cls_head(pooled)
        box_pred = self.box_head(pooled)
        return logits, box_pred


def _quick_test() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThreeDETRClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print("ThreeDETR logits:", logits.shape)
    print("ThreeDETR box_pred:", box_pred.shape)


if __name__ == "__main__":
    _quick_test()
