"""全量时间轴的 frame-photon gated-moment 成像模型。

该模型参考 ``ANN_gated_moment`` 的 ToF 编码、ANN 空间主干和 gated moment
输出口径，但不再按 chunk 截断时间轴。主干固定为 3 个 frame-photon block，
每个 block 使用 stride=2 的时间卷积把时间长度压缩约一半，最后用压缩后的
gate 直接对完整原始 photon 序列做分组 gated moment 聚合，避免把 gate
上采样回完整 P 帧。

时间维没有固定位置表或固定长度全连接层，因此训练时 P=640，测试时 P=1920
这类变长输入可以走同一套权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from SNN_based_method.model.SNN_new import (
    LearnableTofEmbedding,
    MultiScaleDSConv,
    SpatialRefineHead,
    _finalize_gated_peak_maps,
    encode_tof,
)


def _seq_to_ann_forward(x_seq: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """把 [T, B, C, H, W] 展平成 [T*B, C, H, W] 后调用 ANN 模块。"""
    time_steps, batch_size = x_seq.shape[:2]
    flat = x_seq.reshape(time_steps * batch_size, *x_seq.shape[2:])
    out = module(flat)
    return out.reshape(time_steps, batch_size, *out.shape[1:])


def _halve_time_3d(x_3d: torch.Tensor) -> torch.Tensor:
    """对 [B, C, T, H, W] 做相邻帧平均，输出时间长度为 ceil(T/2)。"""
    if x_3d.dim() != 5:
        raise ValueError(f"x_3d must have shape [B, C, T, H, W], got {tuple(x_3d.shape)}")
    time_steps = x_3d.shape[2]
    if time_steps <= 1:
        return x_3d

    if time_steps % 2 == 1:
        x_3d = torch.cat([x_3d, x_3d[:, :, -1:]], dim=2)
        time_steps += 1

    batch_size, channels, _, height, width = x_3d.shape
    return x_3d.reshape(
        batch_size,
        channels,
        time_steps // 2,
        2,
        height,
        width,
    ).mean(dim=3)


def _iter_time_groups(total_steps: int, num_groups: int):
    """把完整时间轴按低分辨率 gate 数量切成连续分组。"""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")

    for group_index in range(num_groups):
        start = group_index * total_steps // num_groups
        end = (group_index + 1) * total_steps // num_groups
        if end <= start:
            end = min(start + 1, total_steps)
        yield start, end


def _group_photons_for_gate(
    gate: torch.Tensor,
    tof: torch.Tensor,
    valid: torch.Tensor,
    t_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """用低分辨率 gate 直接聚合原始 photon 分组。

    ``gate`` 的每个时间步对应原始 P 帧中的一个连续分组。返回的
    ``grouped_tof`` 是组内有效 photon 的平均 ToF，``grouped_valid`` 是组内
    有效 photon 计数，用于让现有 gate 方差和稀疏 loss 在低分辨率上继续工作。
    """
    if gate.dim() != 5 or gate.shape[2] != 1:
        raise ValueError(f"gate must have shape [G, B, 1, H, W], got {tuple(gate.shape)}")
    if tof.shape != valid.shape:
        raise ValueError("tof and valid must have the same shape")
    if tof.dim() != 4:
        raise ValueError(f"tof must have shape [P, B, H, W], got {tuple(tof.shape)}")

    num_groups, batch_size, _, height, width = gate.shape
    num_pages = tof.shape[0]
    num_pixels = batch_size * height * width

    weighted_sum = tof.new_zeros(batch_size, 1, height, width)
    weight_sum = tof.new_zeros(batch_size, 1, height, width)
    gate_hist = tof.new_zeros(int(t_max) + 1, num_pixels)
    grouped_tofs: list[torch.Tensor] = []
    grouped_valids: list[torch.Tensor] = []

    for group_index, (start, end) in enumerate(_iter_time_groups(num_pages, num_groups)):
        group_tof = tof[start:end]
        group_valid = valid[start:end]
        valid_count = group_valid.sum(0)
        tof_sum = (group_tof * group_valid).sum(0)
        tof_mean = tof_sum / valid_count.clamp_min(1e-6)

        gate_group = gate[group_index]
        weighted_sum = weighted_sum + gate_group * tof_sum.unsqueeze(1)
        weight_sum = weight_sum + gate_group * valid_count.unsqueeze(1)

        gate_weight = (
            gate_group.squeeze(1).unsqueeze(0).to(group_valid.dtype) * group_valid
        ).reshape(group_tof.shape[0], num_pixels)
        bin_index = group_tof.long().clamp(0, int(t_max)).reshape(group_tof.shape[0], num_pixels)
        gate_hist.scatter_add_(0, bin_index, gate_weight.to(gate_hist.dtype))

        grouped_tofs.append(tof_mean)
        grouped_valids.append(valid_count)

    return (
        weighted_sum,
        weight_sum,
        gate_hist,
        torch.stack(grouped_tofs, dim=0),
        torch.stack(grouped_valids, dim=0),
    )


class _FramePhotonStem(nn.Module):
    """逐帧 ANN stem，将 ToF 编码特征映射到工作通道。"""

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


class FramePhotonBlock(nn.Module):
    """空间残差提取 + 时间 stride=2 压缩的 frame-photon block。

    输入输出均为 ``[T, B, C, H, W]``。时间卷积使用 padding=1、stride=2，
    因此输出时间长度为 ``ceil(T / 2)``；当 T=1 时保持 1，不会产生空序列。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            MultiScaleDSConv(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.temporal_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.temporal_pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.temporal_bn = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        residual = x_seq
        x_seq = _seq_to_ann_forward(x_seq, self.spatial)
        x_seq = self.act(x_seq + residual)

        x_3d = x_seq.permute(1, 2, 0, 3, 4).contiguous()
        out = self.temporal_dw(x_3d)
        out = self.act(out)
        out = self.temporal_pw(out)
        out = self.temporal_bn(out)

        skip = _halve_time_3d(x_3d)
        if skip.shape[2] != out.shape[2]:
            raise RuntimeError(
                f"temporal skip length {skip.shape[2]} does not match conv length {out.shape[2]}"
            )
        out = self.act(out + skip)
        return out.permute(2, 0, 1, 3, 4).contiguous()


class _FramePhotonGateHead(nn.Module):
    """压缩时间轴上的 ANN gate logits 头。"""

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
        return _seq_to_ann_forward(x_seq, self.net)


class FramePhotonNet(nn.Module):
    """固定 3 层时间压缩 block 的全量 frame-photon 模型。

    构造函数保留 ``SPADSpikeNet`` 的主要参数签名，便于复用现有
    ``SNNConfig``、训练脚本和 checkpoint 元数据。``chunk_size``、spike 相关
    参数以及 ``num_blocks`` 仅用于兼容旧配置；本模型不分 chunk，并固定构建
    3 个时间压缩 block。
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
        depth_peak_half_width: int = 2,
        refine_max_depth_delta: float = 0.10,
        refine_max_intensity_blend: float = 1.0,
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
        self.depth_peak_half_width = int(depth_peak_half_width)
        self.num_blocks = 3
        self.requested_num_blocks = int(num_blocks)
        if self.depth_peak_half_width < 0:
            raise ValueError("depth_peak_half_width must be non-negative")

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

        self.stem = _FramePhotonStem(c_enc, self.C)
        self.blocks = nn.ModuleList([FramePhotonBlock(self.C) for _ in range(3)])
        self.gate_head = _FramePhotonGateHead(self.C)
        self.refine = SpatialRefineHead(
            mid=refine_mid,
            depth_range=self.t_max,
            max_depth_delta=refine_max_depth_delta,
            max_intensity_blend=refine_max_intensity_blend,
        )

    def _encode_full(
        self,
        data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对完整时间轴生成编码特征、masked ToF 与有效 mask。"""
        valid = ((data >= 1) & (data <= self.t_max)).float()
        tof = data.float() * valid
        if self.encoding_mode == "lut":
            x_seq = self.tof_embedding(data, valid)
        else:
            x_seq = encode_tof(tof, valid, self.n_freq, self.t_max)
        return x_seq, tof, valid

    def _forward_full(
        self,
        data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一次性处理完整 P 帧，并返回压缩时间轴上的低分辨率 gate。"""
        x_seq, tof, valid = self._encode_full(data)

        x_seq = self.stem(x_seq)
        for block in self.blocks:
            x_seq = block(x_seq)

        gate_logits = self.gate_head(x_seq)
        gate = torch.sigmoid(gate_logits)
        return gate, tof, valid

    def forward(
        self,
        raw_data: torch.Tensor,
        *,
        return_sequence: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """执行全量 frame-photon gated-moment 成像。

        Args:
            raw_data: [B, N, P]，N=H*W，P 可以随训练/测试样本变化。
        """
        if raw_data.dim() != 3:
            raise ValueError(f"raw_data must have shape [B, N, P], got {tuple(raw_data.shape)}")

        should_return_sequence = self.return_sequence if return_sequence is None else bool(return_sequence)
        batch_size, num_pixels, num_pages = raw_data.shape
        if num_pages <= 0:
            raise ValueError("raw_data must contain at least one page")

        height = width = int(num_pixels ** 0.5)
        if height * width != num_pixels:
            raise ValueError(f"num_pixels must be a square number, got {num_pixels}")

        data = raw_data.view(batch_size, height, width, num_pages)
        data = data.permute(3, 0, 1, 2).contiguous()

        gate, tof, valid = self._forward_full(data)
        weighted_sum, weight_sum, gate_hist, grouped_tof, grouped_valid = _group_photons_for_gate(
            gate,
            tof,
            valid,
            self.t_max,
        )

        maps = _finalize_gated_peak_maps(
            weighted_sum=weighted_sum,
            weight_sum=weight_sum,
            gate_hist=gate_hist,
            num_pages=num_pages,
            depth_peak_half_width=self.depth_peak_half_width,
        )
        depth = maps["depth"]
        intensity = maps["intensity"]
        confidence = maps["confidence"]
        coarse = torch.cat([depth, intensity], dim=1)
        output = self.refine(coarse, confidence)

        result: dict[str, torch.Tensor] = {
            "output": output,
            "depth": output[:, 0:1],
            "intensity": output[:, 1:2],
            "depth_coarse": depth,
            "intensity_coarse": intensity,
            "confidence": confidence,
            "support": maps["support"],
            "selectivity": maps["selectivity"],
            "gate_hist": (
                gate_hist.detach()
                .reshape(self.t_max + 1, batch_size, height, width)
                .permute(1, 0, 2, 3)
                .contiguous()
            ),
        }
        if should_return_sequence:
            result["gate"] = gate
            result["tof"] = grouped_tof
            result["valid"] = grouped_valid
        if self.tof_embedding is not None:
            result["lut_smooth"] = self.tof_embedding.smoothness_loss()
            result["lut_norm"] = self.tof_embedding.norm_loss()
        return result


SPADFramePhoton = FramePhotonNet

__all__ = ["FramePhotonBlock", "FramePhotonNet", "SPADFramePhoton"]
