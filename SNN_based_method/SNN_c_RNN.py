"""SPAD 浓雾场景 SNN 的显式 RNN 等价实现。

输入:
    ``raw_data`` 形状为 ``[B, 4096, P]``，其中 0 表示无效或未触发 ToF。

输出:
    字典形式结果，核心输出 ``output`` 形状为 ``[B, 2, 64, 64]``，
    通道 0 为深度，通道 1 为强度。

设计说明:
    本文件保持 ``SNN_new.SPADSpikeNet`` 的编码、卷积主干、gate 聚合和
    输出协议不变，只把脉冲神经元的隐式膜电位演化显式展开成 RNN 递推。
    对于每个脉冲层，隐藏状态就是膜电位 ``v_t``，递推公式与
    spikingjelly.activation_based 中的 IF/LIF/PLIF 一致。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate

from SNN_based_method.SNN_new import (
    LearnableTofEmbedding,
    MultiScaleDSConv,
    SpatialRefineHead,
    encode_tof,
)


def _seq_to_ann_forward(x_seq: torch.Tensor, *modules: nn.Module) -> torch.Tensor:
    """把 ``[T, B, ...]`` 序列展平成 ``[T*B, ...]`` 后送入 ANN 子模块。

    这样可以保持与 ``functional.seq_to_ann_forward`` 一致的 BatchNorm 统计行为。
    """
    if x_seq.dim() < 2:
        raise ValueError(f"x_seq must have at least 2 dims [T, B, ...], got {tuple(x_seq.shape)}")

    t_steps, batch_size = x_seq.shape[:2]
    x_flat = x_seq.flatten(0, 1)
    for module in modules:
        x_flat = module(x_flat)
    return x_flat.reshape(t_steps, batch_size, *x_flat.shape[1:])


def _detach_state_tree(state: Any) -> Any:
    """递归 detach 嵌套状态，用于 chunk 间截断 BPTT。"""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(_detach_state_tree(item) for item in state)
    if isinstance(state, list):
        return [_detach_state_tree(item) for item in state]
    if isinstance(state, dict):
        return {key: _detach_state_tree(value) for key, value in state.items()}
    raise TypeError(f"Unsupported state type: {type(state)!r}")


class SpikingRecurrentCell(nn.Module):
    """把 IF/LIF/PLIF 神经元显式改写为 RNN cell。

    对于输入序列 ``x[t]``，隐藏状态 ``v[t]`` 表示膜电位。每个时间步执行:
    1. charge: 按 IF/LIF/PLIF 方程更新膜电位
    2. fire: 用 surrogate Heaviside 生成脉冲
    3. reset: 按 hard/soft reset 更新下一时刻状态
    """

    def __init__(
        self,
        spike_mode: str = "plif",
        tau: float = 2.0,
        v_threshold: float = 0.5,
        v_reset: float | None = 0.0,
        detach_reset: bool = True,
        decay_input: bool = True,
    ) -> None:
        super().__init__()
        self.spike_mode = str(spike_mode).lower()
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.v_reset = None if v_reset is None else float(v_reset)
        self.detach_reset = bool(detach_reset)
        self.decay_input = bool(decay_input)
        self.surrogate_function = surrogate.Sigmoid(alpha=4.0, spiking=True)

        if self.spike_mode not in {"if", "lif", "plif"}:
            raise ValueError(f"不支持的 spike_mode: {spike_mode}, 可选 plif/lif/if")
        if self.spike_mode == "plif":
            if self.tau <= 1.0:
                raise ValueError("PLIF 要求 tau > 1.0")
            init_w = -math.log(self.tau - 1.0)
            self.w = nn.Parameter(torch.as_tensor(init_w, dtype=torch.float32))
        elif self.spike_mode == "lif":
            if self.tau <= 0.0:
                raise ValueError("LIF 要求 tau > 0")

    def _charge(self, x_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        """按当前神经元类型执行一次 charge 更新。"""
        if self.spike_mode == "if":
            return v_t + x_t

        if self.spike_mode == "lif":
            if self.decay_input:
                if self.v_reset is None or self.v_reset == 0.0:
                    return v_t + (x_t - v_t) / self.tau
                return v_t + (x_t - (v_t - self.v_reset)) / self.tau

            if self.v_reset is None or self.v_reset == 0.0:
                return v_t * (1.0 - 1.0 / self.tau) + x_t
            return v_t - (v_t - self.v_reset) / self.tau + x_t

        reciprocal_tau = self.w.sigmoid()
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.0:
                return v_t + (x_t - v_t) * reciprocal_tau
            return v_t + (x_t - (v_t - self.v_reset)) * reciprocal_tau

        if self.v_reset is None or self.v_reset == 0.0:
            return v_t * (1.0 - reciprocal_tau) + x_t
        return v_t - (v_t - self.v_reset) * reciprocal_tau + x_t

    def _reset(self, v_t: torch.Tensor, spike_t: torch.Tensor) -> torch.Tensor:
        """按 hard/soft reset 公式更新膜电位。"""
        spike_for_reset = spike_t.detach() if self.detach_reset else spike_t
        if self.v_reset is None:
            return v_t - spike_for_reset * self.v_threshold
        return (1.0 - spike_for_reset) * v_t + spike_for_reset * self.v_reset

    def forward(
        self,
        x_seq: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """处理 ``[T, B, ...]`` 输入序列并返回脉冲序列与最终状态。"""
        if x_seq.dim() < 2:
            raise ValueError(f"x_seq must have shape [T, B, ...], got {tuple(x_seq.shape)}")
        if x_seq.shape[0] <= 0:
            raise ValueError("x_seq 的时间维 T 必须大于 0")

        v_t = torch.zeros_like(x_seq[0]) if state is None else state
        spikes: list[torch.Tensor] = []

        for t in range(x_seq.shape[0]):
            v_t = self._charge(x_seq[t], v_t)
            spike_t = self.surrogate_function(v_t - self.v_threshold)
            v_t = self._reset(v_t, spike_t)
            spikes.append(spike_t)

        return torch.stack(spikes, dim=0), v_t

    def extra_repr(self) -> str:
        tau_repr = self.tau if self.spike_mode != "plif" else "learnable"
        return (
            f"spike_mode={self.spike_mode}, tau={tau_repr}, "
            f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, "
            f"detach_reset={self.detach_reset}, decay_input={self.decay_input}"
        )


class _StemRNN(nn.Module):
    """Stem 网络的 RNN 等价版本。"""

    def __init__(
        self,
        c_enc: int,
        c_hidden: int,
        spike_mode: str,
        spike_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.spike_backend = str(spike_backend).lower()
        self.conv1 = nn.Conv2d(c_enc, c_hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hidden)
        self.spike = SpikingRecurrentCell(spike_mode=spike_mode)
        self.conv2 = nn.Conv2d(c_hidden, c_hidden, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_hidden)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = _seq_to_ann_forward(x_seq, self.conv1, self.bn1)
        x_seq, state = self.spike(x_seq, state)
        x_seq = _seq_to_ann_forward(x_seq, self.conv2, self.bn2)
        return x_seq, state


class SpikeBlockRNN(nn.Module):
    """SpikeBlock 的显式 RNN 版本，保持模块命名与原模型一致。"""

    def __init__(self, c_hidden: int, spike_mode: str, spike_backend: str = "auto") -> None:
        super().__init__()
        self.spike_backend = str(spike_backend).lower()
        self.spike_in = SpikingRecurrentCell(spike_mode=spike_mode)
        self.ms_dsconv = MultiScaleDSConv(c_hidden)
        self.spike_mid = SpikingRecurrentCell(spike_mode=spike_mode)
        self.pw = nn.Conv2d(c_hidden, c_hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_hidden)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        identity = x_seq
        state_in, state_mid = (None, None) if state is None else state

        x_seq, state_in = self.spike_in(x_seq, state_in)
        x_seq = _seq_to_ann_forward(x_seq, self.ms_dsconv)
        x_seq, state_mid = self.spike_mid(x_seq, state_mid)
        x_seq = _seq_to_ann_forward(x_seq, self.pw, self.bn)
        return x_seq + identity, (state_in, state_mid)


class _GateHeadRNN(nn.Module):
    """GateHead 的显式 RNN 版本。"""

    def __init__(self, c_hidden: int, spike_mode: str, spike_backend: str = "auto") -> None:
        super().__init__()
        self.spike_backend = str(spike_backend).lower()
        self.spike1 = SpikingRecurrentCell(spike_mode=spike_mode)
        self.conv1 = nn.Conv2d(c_hidden, c_hidden // 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hidden // 2)
        self.spike2 = SpikingRecurrentCell(spike_mode=spike_mode)
        self.conv2 = nn.Conv2d(c_hidden // 2, 1, 1, bias=True)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        state1, state2 = (None, None) if state is None else state

        x_seq, state1 = self.spike1(x_seq, state1)
        x_seq = _seq_to_ann_forward(x_seq, self.conv1, self.bn1)
        x_seq, state2 = self.spike2(x_seq, state2)
        x_seq = _seq_to_ann_forward(x_seq, self.conv2)
        return torch.sigmoid(x_seq), (state1, state2)


class SNN_c_RNN(nn.Module):
    """把 ``SPADSpikeNet`` 等价显式展开成 RNN 的版本。"""

    def __init__(
        self,
        C: int = 32,
        chunk_size: int = 128,
        spike_mode: str = "plif",
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

        self.stem = _StemRNN(c_enc, self.C, spike_mode, spike_backend=self.spike_backend)
        self.blocks = nn.ModuleList(
            [
                SpikeBlockRNN(self.C, spike_mode, spike_backend=self.spike_backend)
                for _ in range(num_blocks)
            ]
        )
        self.gate_head = _GateHeadRNN(self.C, spike_mode, spike_backend=self.spike_backend)
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
        """构造显式 RNN 状态容器。"""
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
        """处理单个 chunk，并显式返回新的 RNN 状态。"""
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


SPADSpikeRNN = SNN_c_RNN


__all__ = [
    "SNN_c_RNN",
    "SPADSpikeRNN",
    "SpikingRecurrentCell",
    "SpikeBlockRNN",
]
