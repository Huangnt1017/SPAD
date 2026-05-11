import torch
import torch.nn as nn


def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
    """Normalize input to (B, N, 4).

    Supports (B, N, 4), (B, 4, N), and xyz-only (B, N, 3)/(B, 3, N).
    """
    if x.ndim != 3:
        raise ValueError(f"DCTClassification expects 3D input, got shape {tuple(x.shape)}")

    if x.shape[-1] in (3, 4):
        points = x
    elif x.shape[1] in (3, 4):
        points = x.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            "DCTClassification expects input layout (B,N,4)/(B,4,N) or xyz-only "
            f"(B,N,3)/(B,3,N), got {tuple(x.shape)}"
        )

    if points.shape[-1] == 3:
        pad_i = torch.zeros(points.shape[0], points.shape[1], 1, dtype=points.dtype, device=points.device)
        points = torch.cat([points, pad_i], dim=-1)

    return points


class DCTClassification(nn.Module):
    """Minimal dual-channel transformer (DCT) with class + 3D box heads."""

    def __init__(
        self,
        num_classes: int = 26,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.xyz_proj = nn.Linear(3, d_model)
        self.i_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.xyz_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.i_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

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
        intensity = points[:, :, 3:4]

        xyz_feat = self.xyz_proj(xyz)
        i_feat = self.i_proj(intensity)

        xyz_enc = self.xyz_encoder(xyz_feat)
        i_enc = self.i_encoder(i_feat)

        cross_out, _ = self.cross_attn(xyz_enc, i_enc, i_enc)
        fused = self.fuse_mlp(xyz_enc + cross_out)

        g = torch.max(fused, dim=1)[0]
        logits = self.cls_head(g)
        box_pred = self.box_head(g)
        return logits, box_pred


def _quick_test() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DCTClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print("DCT logits:", logits.shape)
    print("DCT box_pred:", box_pred.shape)


if __name__ == "__main__":
    _quick_test()
