"""SPAD 浓雾场景 SNN 的显式 ConvGRU 版本。

输入:
    ``raw_data`` 形状为 ``[B, 4096, P]``，其中 0 表示无效或未触发 ToF。

输出:
    字典形式结果，核心输出 ``output`` 形状为 ``[B, 2, 64, 64]``，
    通道 0 为深度，通道 1 为强度。

设计说明:
    本文件保持 ``SNN_new.SPADSpikeNet`` 的编码、卷积主干、gate 聚合和
    输出协议不变，只把脉冲层替换成显式的时空 ConvGRU 递推。这样可以在
    不改训练脚本和 loss 接口的前提下，对比 SNN / RNN / LSTM / GRU 的
    时序建模差异。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from SNN_based_method.model.SNN_c_RNN import _detach_state_tree, _seq_to_ann_forward
from SNN_based_method.model.SNN_new import (
    LearnableTofEmbedding,
    MultiScaleDSConv,
    SpatialRefineHead,
    encode_tof,
)


class ConvGRUCell(nn.Module):
    """标准 ConvGRU cell，输入和隐藏状态均保留空间布局。"""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.gate_conv = nn.Conv2d(
            self.input_channels + self.hidden_channels,
            2 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias,
        )
        self.candidate_conv = nn.Conv2d(
            self.input_channels + self.hidden_channels,
            self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def _init_state(self, x_t: torch.Tensor) -> torch.Tensor:
        """根据当前输入尺寸创建零初始化的隐藏状态。"""
        batch_size, _, height, width = x_t.shape
        return x_t.new_zeros(batch_size, self.hidden_channels, height, width)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """处理 ``[T, B, C, H, W]`` 输入序列。"""
        if x_seq.dim() != 5:
            raise ValueError(f"x_seq must have shape [T, B, C, H, W], got {tuple(x_seq.shape)}")
        if x_seq.shape[0] <= 0:
            raise ValueError("x_seq 的时间维 T 必须大于 0")

        h_t = self._init_state(x_seq[0]) if state is None else state
        outputs: list[torch.Tensor] = []

        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]
            gates = self.gate_conv(torch.cat([x_t, h_t], dim=1))
            z_t, r_t = torch.chunk(gates, 2, dim=1)
            z_t = torch.sigmoid(z_t)
            r_t = torch.sigmoid(r_t)

            candidate_input = torch.cat([x_t, r_t * h_t], dim=1)
            n_t = torch.tanh(self.candidate_conv(candidate_input))
            h_t = (1.0 - z_t) * n_t + z_t * h_t
            outputs.append(h_t)

        return torch.stack(outputs, dim=0), h_t

    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"kernel_size={self.kernel_size}"
        )


class _StemGRU(nn.Module):
    """Stem 网络的 ConvGRU 版本。"""

    def __init__(
        self,
        c_enc: int,
        c_hidden: int,
        spike_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.spike_backend = str(spike_backend).lower()
        self.conv1 = nn.Conv2d(c_enc, c_hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hidden)
        self.gru = ConvGRUCell(c_hidden, c_hidden, kernel_size=3)
        self.conv2 = nn.Conv2d(c_hidden, c_hidden, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_hidden)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = _seq_to_ann_forward(x_seq, self.conv1, self.bn1)
        x_seq, state = self.gru(x_seq, state)
        x_seq = _seq_to_ann_forward(x_seq, self.conv2, self.bn2)
        return x_seq, state


class GRUBlock(nn.Module):
    """保持残差结构的 ConvGRU 版时序块。"""

    def __init__(self, c_hidden: int, spike_backend: str = "auto") -> None:
        super().__init__()
        self.spike_backend = str(spike_backend).lower()
        self.gru_in = ConvGRUCell(c_hidden, c_hidden, kernel_size=3)
        self.ms_dsconv = MultiScaleDSConv(c_hidden)
        self.gru_mid = ConvGRUCell(c_hidden, c_hidden, kernel_size=3)
        self.pw = nn.Conv2d(c_hidden, c_hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_hidden)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        identity = x_seq
        state_in, state_mid = (None, None) if state is None else state

        x_seq, state_in = self.gru_in(x_seq, state_in)
        x_seq = _seq_to_ann_forward(x_seq, self.ms_dsconv)
        x_seq, state_mid = self.gru_mid(x_seq, state_mid)
        x_seq = _seq_to_ann_forward(x_seq, self.pw, self.bn)
        return x_seq + identity, (state_in, state_mid)


class _GateHeadGRU(nn.Module):
    """Gate 头的 ConvGRU 版本。"""

    def __init__(self, c_hidden: int, spike_backend: str = "auto") -> None:
        super().__init__()
        self.spike_backend = str(spike_backend).lower()
        self.mid_channels = max(1, c_hidden // 2)
        self.gru1 = ConvGRUCell(c_hidden, c_hidden, kernel_size=3)
        self.conv1 = nn.Conv2d(c_hidden, self.mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.gru2 = ConvGRUCell(self.mid_channels, self.mid_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(self.mid_channels, 1, 1, bias=True)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        state1, state2 = (None, None) if state is None else state

        x_seq, state1 = self.gru1(x_seq, state1)
        x_seq = _seq_to_ann_forward(x_seq, self.conv1, self.bn1)
        x_seq, state2 = self.gru2(x_seq, state2)
        x_seq = _seq_to_ann_forward(x_seq, self.conv2)
        return torch.sigmoid(x_seq), (state1, state2)


class SNN_c_GRU(nn.Module):
    """保持 SNN 输出协议不变的 ConvGRU 成像模型。"""

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
        self.spike_mode = str(spike_mode).lower()
        self.spike_tau = float(spike_tau)
        self.spike_v_threshold = float(spike_v_threshold)
        self.spike_v_reset = None if spike_v_reset is None else float(spike_v_reset)
        self.spike_backend = str(spike_backend).lower()

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

        self.stem = _StemGRU(c_enc, self.C, spike_backend=self.spike_backend)
        self.blocks = nn.ModuleList(
            [GRUBlock(self.C, spike_backend=self.spike_backend) for _ in range(num_blocks)]
        )
        self.gate_head = _GateHeadGRU(self.C, spike_backend=self.spike_backend)
        self.refine = SpatialRefineHead(mid=refine_mid, depth_range=self.t_max)

    def _encode_chunk(
        self,
        chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对一个 chunk 生成编码特征与有效 mask。"""
        valid = ((chunk >= 1) & (chunk <= self.t_max)).float()
        tof = chunk.float() * valid

        if self.encoding_mode == "lut":
            x_seq = self.tof_embedding(chunk, valid)
        else:
            x_seq = encode_tof(tof, valid, self.n_freq, self.t_max)
        return x_seq, tof, valid

    def _initial_state(self) -> dict[str, Any]:
        """构造显式 GRU 状态容器。"""
        return {
            "stem": None,
            "blocks": [None for _ in self.blocks],
            "gate_head": None,
        }

    def _forward_chunk(
        self,
        data_chunk: torch.Tensor,
        state: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """处理单个 chunk，并显式返回新的 GRU 状态。"""
        x_seq, tof, valid = self._encode_chunk(data_chunk)
        prev_state = self._initial_state() if state is None else state

        x_seq, stem_state = self.stem(x_seq, prev_state["stem"])

        block_states: list[Any] = []
        for block, block_state in zip(self.blocks, prev_state["blocks"]):
            x_seq, new_block_state = block(x_seq, block_state)
            block_states.append(new_block_state)

        gate, gate_state = self.gate_head(x_seq, prev_state["gate_head"])
        new_state = {
            "stem": stem_state,
            "blocks": block_states,
            "gate_head": gate_state,
        }
        return gate, tof, valid, new_state

    def forward(
        self,
        raw_data: torch.Tensor,
        *,
        return_sequence: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """前向接口与 ``SNN_new.SPADSpikeNet`` 保持一致。"""
        should_return_sequence = self.return_sequence if return_sequence is None else bool(return_sequence)
        batch_size, num_pixels, num_pages = raw_data.shape
        height = width = int(num_pixels ** 0.5)
        if height * width != num_pixels:
            raise ValueError(f"N 必须可还原为正方形 H*W，当前 N={num_pixels}")

        device = raw_data.device
        t_chunk = self.chunk_size
        data = raw_data.view(batch_size, height, width, num_pages).permute(3, 0, 1, 2).contiguous()

        weighted_sum = torch.zeros(batch_size, 1, height, width, device=device)
        weight_sum = torch.zeros(batch_size, 1, height, width, device=device)
        all_gates: list[torch.Tensor] = []
        all_tofs: list[torch.Tensor] = []
        all_valids: list[torch.Tensor] = []

        state: dict[str, Any] | None = None
        n_chunks = math.ceil(num_pages / t_chunk)
        for chunk_index in range(n_chunks):
            t0 = chunk_index * t_chunk
            t1 = min(t0 + t_chunk, num_pages)
            chunk = data[t0:t1]

            t_actual = t1 - t0
            if t_actual < t_chunk:
                pad = torch.zeros(t_chunk - t_actual, batch_size, height, width, device=device)
                chunk = torch.cat([chunk, pad], dim=0)

            gate, tof, valid, state = self._forward_chunk(chunk, state)

            gate = gate[:t_actual]
            tof = tof[:t_actual]
            valid = valid[:t_actual]

            tof_exp = tof.unsqueeze(2)
            valid_exp = valid.unsqueeze(2)
            weighted_sum = weighted_sum + (gate * tof_exp * valid_exp).sum(0)
            weight_sum = weight_sum + (gate * valid_exp).sum(0)

            if should_return_sequence:
                all_gates.append(gate)
                all_tofs.append(tof)
                all_valids.append(valid)

            if chunk_index < n_chunks - 1:
                state = _detach_state_tree(state)

        depth = weighted_sum / (weight_sum + 1e-6)
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


SPADSpikeGRU = SNN_c_GRU


__all__ = [
    "SNN_c_GRU",
    "SPADSpikeGRU",
    "ConvGRUCell",
    "GRUBlock",
]
