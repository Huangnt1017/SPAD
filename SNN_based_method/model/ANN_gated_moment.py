"""非脉冲 ANN gate-moment 成像 baseline。

该模型保留 ``SPADSpikeNet`` 的输入输出协议和 gated moment 聚合方式,
但把时序主干中的脉冲神经元替换为逐帧 ANN 卷积与 GELU 激活。它用于区分
性能收益来自 gate-moment 结构本身, 还是来自 SNN/PLIF 的脉冲时序机制。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from SNN_based_method.model.SNN_new import (
    LearnableTofEmbedding,
    MultiScaleDSConv,
    SpatialRefineHead,
    encode_tof,
)


def _seq_to_ann_forward(x_seq: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """把 [T, B, C, H, W] 展平成 [T*B, C, H, W] 后调用 ANN 模块。"""
    time_steps, batch_size = x_seq.shape[:2]
    flat = x_seq.reshape(time_steps * batch_size, *x_seq.shape[2:])
    out = module(flat)
    return out.reshape(time_steps, batch_size, *out.shape[1:])


class _ANNStem(nn.Module):
    """逐帧 ANN stem, 将 ToF 编码特征映射到工作通道。"""

    def __init__(self, c_enc: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_enc, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return _seq_to_ann_forward(x_seq, self.net)


class ANNGatedBlock(nn.Module):
    """逐帧 ANN 残差块, 与 SpikeBlock 使用相同的多尺度深度可分离卷积。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            MultiScaleDSConv(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        residual = x_seq
        x_seq = _seq_to_ann_forward(x_seq, self.net)
        return self.act(x_seq + residual)


class _ANNGateHead(nn.Module):
    """逐帧 ANN gate 头, 输出 [T, B, 1, H, W] 的连续 gate。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(1, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(_seq_to_ann_forward(x_seq, self.net))


class ANNGatedMomentNet(nn.Module):
    """SPAD gated-moment 的非脉冲 ANN baseline。

    构造函数保留 ``SPADSpikeNet`` 的主要参数签名, 其中 spike 相关参数仅用于
    与配置系统兼容, 不参与模型构建。
    """

    def __init__(
        self,
        C: int = 32,
        chunk_size: int = 128,
        spike_mode: str = "plif",
        spike_tau: float = 2.0,
        spike_v_threshold: float = 0.5,
        spike_v_reset: float | None = 0.0,
        t_max: int = 128,
        n_freq: int = 8,
        num_blocks: int = 3,
        encoding_mode: str = "sinusoidal",
        embed_dim: int = 16,
        lut_init: str = "sinusoidal",
        refine_mid: int = 8,
        return_sequence: bool = True,
        spike_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.C = int(C)
        self.chunk_size = int(chunk_size)
        self.t_max = int(t_max)
        self.n_freq = int(n_freq)
        self.encoding_mode = str(encoding_mode).lower()
        self.return_sequence = bool(return_sequence)

        if self.encoding_mode == "lut":
            self.tof_embedding = LearnableTofEmbedding(
                t_max=self.t_max,
                embed_dim=embed_dim,
                init_mode=lut_init,
                n_freq_init=self.n_freq,
            )
            c_enc = int(embed_dim)
        elif self.encoding_mode == "sinusoidal":
            self.tof_embedding = None
            c_enc = 2 * self.n_freq + 1
        else:
            raise ValueError("encoding_mode must be 'sinusoidal' or 'lut'")

        self.stem = _ANNStem(c_enc, self.C)
        self.blocks = nn.ModuleList([ANNGatedBlock(self.C) for _ in range(num_blocks)])
        self.gate_head = _ANNGateHead(self.C)
        self.refine = SpatialRefineHead(mid=refine_mid, depth_range=self.t_max)

    def _encode_chunk(
        self,
        chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对一个 chunk 生成编码特征、masked ToF 与有效 mask。"""
        valid = ((chunk >= 1) & (chunk <= self.t_max)).float()
        tof = chunk.float() * valid
        if self.encoding_mode == "lut":
            x_seq = self.tof_embedding(chunk, valid)
        else:
            x_seq = encode_tof(tof, valid, self.n_freq, self.t_max)
        return x_seq, tof, valid

    def _forward_chunk(
        self,
        data_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理单个 chunk, 返回连续 gate、ToF 和 valid mask。"""
        x_seq, tof, valid = self._encode_chunk(data_chunk)
        x_seq = self.stem(x_seq)
        for block in self.blocks:
            x_seq = block(x_seq)
        gate = self.gate_head(x_seq)
        return gate, tof, valid

    def forward(
        self,
        raw_data: torch.Tensor,
        *,
        return_sequence: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """执行 ANN gate-moment 成像。

        Args:
            raw_data: [B, N, P], N=H*W, P 为 page 数。
        """
        should_return_sequence = self.return_sequence if return_sequence is None else bool(return_sequence)
        batch_size, num_pixels, num_pages = raw_data.shape
        height = width = int(num_pixels ** 0.5)
        if height * width != num_pixels:
            raise ValueError(f"num_pixels must be a square number, got {num_pixels}")

        data = raw_data.view(batch_size, height, width, num_pages)
        data = data.permute(3, 0, 1, 2).contiguous()
        device = raw_data.device
        weighted_sum = torch.zeros(batch_size, 1, height, width, device=device)
        weight_sum = torch.zeros(batch_size, 1, height, width, device=device)
        all_gates: list[torch.Tensor] = []
        all_tofs: list[torch.Tensor] = []
        all_valids: list[torch.Tensor] = []

        chunk_size = max(1, self.chunk_size)
        for chunk_index in range(math.ceil(num_pages / chunk_size)):
            t0 = chunk_index * chunk_size
            t1 = min(t0 + chunk_size, num_pages)
            chunk = data[t0:t1]
            gate, tof, valid = self._forward_chunk(chunk)

            tof_exp = tof.unsqueeze(2)
            valid_exp = valid.unsqueeze(2)
            weighted_sum = weighted_sum + (gate * tof_exp * valid_exp).sum(0)
            weight_sum = weight_sum + (gate * valid_exp).sum(0)

            if should_return_sequence:
                all_gates.append(gate)
                all_tofs.append(tof)
                all_valids.append(valid)

        depth = weighted_sum / (weight_sum + 1.0e-6)
        intensity = weight_sum / num_pages
        confidence = (weight_sum / (weight_sum + 1.0)).clamp(0.0, 1.0)
        coarse = torch.cat([depth, intensity], dim=1)
        output = self.refine(coarse, confidence)

        result: dict[str, torch.Tensor] = {
            "output": output,
            "depth": output[:, 0:1],
            "intensity": output[:, 1:2],
            "depth_coarse": depth,
            "intensity_coarse": intensity,
            "confidence": confidence,
        }
        if should_return_sequence:
            result["gate"] = torch.cat(all_gates, dim=0)
            result["tof"] = torch.cat(all_tofs, dim=0)
            result["valid"] = torch.cat(all_valids, dim=0)
        if self.tof_embedding is not None:
            result["lut_smooth"] = self.tof_embedding.smoothness_loss()
            result["lut_norm"] = self.tof_embedding.norm_loss()
        return result


SPADANNGatedMoment = ANNGatedMomentNet
