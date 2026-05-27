"""SPAD Dense-Fog SNN Imaging Model

Input:  [B, 4096, P]   raw ToF timestamps (0 = invalid/untriggered)
Output: [B, 2, 64, 64] (depth, intensity)

Architecture: sinusoidal positional encoding → equal-width SpikeBlocks
              → EchoGate → Gated Moment → spatial refinement head
Supports chunked processing for large P (e.g., P=500).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
from spikingjelly1.clock_driven import functional
from spikingjelly1.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
    MultiStepIFNode,
)

from spikingjelly.activation_based import neuron

try:
    from utils.pointnet_utils import build_spike_node as _build_spike_node_cupy
    _HAS_CUPY_BUILDER = True
except Exception:
    _HAS_CUPY_BUILDER = False


def build_node(timestep, spike_mode="plif", tau=2.0, v_threshold=0.5):
    if _HAS_CUPY_BUILDER and torch.cuda.is_available():
        try:
            return _build_spike_node_cupy(timestep, spike_mode, tau=tau, v_threshold=v_threshold)
        except Exception:
            pass
    if spike_mode == "plif":
        return MultiStepParametricLIFNode(
            timestep=timestep, init_tau=tau, v_threshold=v_threshold,
            detach_reset=True, backend="torch",
        )
    elif spike_mode == "lif":
        return MultiStepLIFNode(
            timestep=timestep, tau=tau, v_threshold=v_threshold,
            detach_reset=True, backend="torch",
        )
    elif spike_mode == "if":
        return MultiStepIFNode(
            timestep=timestep, v_threshold=v_threshold,
            detach_reset=True, backend="torch",
        )
    else:
        raise ValueError(f"Unsupported spike_mode: {spike_mode}")


# ─── Encoding ────────────────────────────────────────────

def encode_tof(tof, valid, n_freq=8, t_max=150):
    """正弦位置编码: 将整数 tof 映射为多频率 sin/cos 特征.

    Args:
        tof:   [*, H, W] int/float, raw timestamps
        valid: [*, H, W] float, 1.0 if valid
        n_freq: number of frequency pairs
        t_max:  max valid bin

    Returns:
        [*, 2*n_freq+1, H, W]  all values in [-1, 1]
    """
    v = valid.unsqueeze(-3)                                # [*, 1, H, W]
    t = (tof.float() / t_max).unsqueeze(-3) * v           # [*, 1, H, W]
    channels = [v]
    for i in range(n_freq):
        freq = (i + 1) * math.pi
        channels.append(torch.sin(freq * t) * v)
        channels.append(torch.cos(freq * t) * v)
    return torch.cat(channels, dim=-3)                     # [*, 17, H, W]


class LearnableTofEmbedding(nn.Module):
    """可学习 LUT 编码: 将整数 ToF bin 查表映射为低维时间位置特征.

    核心是一个 nn.Embedding(t_max+1, embed_dim), embed_dim 可自由调整,
    与正弦编码完全独立. 训练时通过成像损失反向更新被访问的 LUT 表项.

    稳定性约束:
      - padding_idx=0: invalid (tof=0) 映射为全零向量, 不参与梯度
      - valid mask: 编码后 × valid, 无效像素不贡献信号
      - smoothness_loss(): 相邻 bin 平滑正则, 防止 LUT 学出尖锐跳变
      - norm_loss(): 范数一致性正则, 防止 PLIF 膜电位对某些 bin 偏置响应

    初始化策略:
      - "sinusoidal": 用正弦编码值填充 (截断或补零适配 embed_dim)
      - "rbf": 高斯径向基函数, 中心均匀分布在 [1, t_max]
      - "random": 标准正态随机初始化

    Args:
        t_max: 最大有效 tof bin (1~t_max 有效, 0=无效)
        embed_dim: 嵌入维度 (自由调整, 不必等于 17)
        init_mode: 初始化方式 ("sinusoidal" / "rbf" / "random")
        n_freq_init: 正弦初始化时的频率数 (仅 init_mode="sinusoidal" 时使用)
        max_norm: embedding 最大 L2 范数约束 (None=不限)
    """

    def __init__(self, t_max: int = 150, embed_dim: int = 16,
                 init_mode: str = "sinusoidal", n_freq_init: int = 8,
                 max_norm: float = None):
        super().__init__()
        self.t_max = t_max
        self.embed_dim = embed_dim
        # t_max+1 个条目: index 0 = invalid (padding), index 1~t_max = 有效 bin
        self.embedding = nn.Embedding(t_max + 1, embed_dim,
                                      padding_idx=0, max_norm=max_norm)

        if init_mode == "sinusoidal":
            self._init_sinusoidal(n_freq_init)
        elif init_mode == "rbf":
            self._init_rbf()
        elif init_mode == "random":
            self._init_random()
        else:
            raise ValueError(f"不支持的 init_mode: {init_mode}, 可选 sinusoidal/rbf/random")

    def _init_sinusoidal(self, n_freq: int):
        """正弦编码初始化: 用 sin/cos 值填充 LUT.

        若 embed_dim < 2*n_freq+1 则截断, 若 embed_dim > 则补零.
        提供与固定正弦编码相同的训练起点.
        """
        with torch.no_grad():
            weight = self.embedding.weight
            weight.zero_()
            for b in range(1, self.t_max + 1):
                t = b / self.t_max
                vec = [1.0]
                for i in range(n_freq):
                    freq = (i + 1) * math.pi
                    vec.append(math.sin(freq * t))
                    vec.append(math.cos(freq * t))
                vec = vec[:self.embed_dim]
                while len(vec) < self.embed_dim:
                    vec.append(0.0)
                weight[b] = torch.tensor(vec)

    def _init_rbf(self):
        """高斯 RBF 初始化: embed_dim 个高斯基函数, 中心均匀分布.

        每个维度对应一个中心 c_j, 值为 exp(-||t-c_j||^2 / (2σ^2)),
        σ 按相邻中心间距自动设定, 保证基函数有足够重叠.
        """
        with torch.no_grad():
            weight = self.embedding.weight
            weight.zero_()
            # embed_dim 个中心均匀分布在 [1, t_max]
            centers = torch.linspace(1.0, self.t_max, self.embed_dim)
            # σ = 中心间距的 0.8 倍, 保证相邻基函数有 ~60% 重叠
            sigma = (centers[1] - centers[0]).item() * 0.8 if self.embed_dim > 1 else self.t_max / 2.0
            for b in range(1, self.t_max + 1):
                # 各维度: exp(-(b - center_j)^2 / (2 * sigma^2))
                weight[b] = torch.exp(-((b - centers) ** 2) / (2 * sigma ** 2))

    def _init_random(self):
        """标准正态随机初始化, 缩放到合理范围."""
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
            self.embedding.weight[0].zero_()

    def forward(self, tof, valid):
        """查表编码.

        Args:
            tof:   [*, H, W] int/float, raw timestamps (0=invalid)
            valid: [*, H, W] float, 1.0 if valid

        Returns:
            [*, embed_dim, H, W] 编码后特征, invalid 位置为全零
        """
        # clamp 到 [0, t_max], 防止越界; 0 由 padding_idx 映射为零向量
        indices = tof.long().clamp(0, self.t_max)           # [*, H, W]
        # embedding 查表: [*, H, W] → [*, H, W, embed_dim]
        emb = self.embedding(indices)
        # 转置通道: [*, H, W, D] → [*, D, H, W]
        dims = list(range(emb.dim()))
        dims.insert(-2, dims.pop(-1))
        emb = emb.permute(*dims).contiguous()               # [*, D, H, W]
        # valid mask 清零无效像素
        emb = emb * valid.unsqueeze(-3)                     # [*, D, H, W]
        return emb

    def smoothness_loss(self) -> torch.Tensor:
        """相邻 bin 平滑正则: L_adj = mean(||emb[i+1] - emb[i]||^2), i ∈ [1, t_max-1].

        鼓励相邻时间 bin 的编码向量平滑过渡, 避免 LUT 过拟合产生尖锐跳变.
        """
        valid_emb = self.embedding.weight[1:]               # [t_max, embed_dim]
        diff = valid_emb[1:] - valid_emb[:-1]              # [t_max-1, embed_dim]
        return (diff ** 2).mean()

    def norm_loss(self) -> torch.Tensor:
        """范数一致性正则: 鼓励所有有效 bin 的 embedding 范数接近全局均值.

        避免某些 bin 的编码能量过大/过小导致 PLIF 响应偏差.
        """
        valid_emb = self.embedding.weight[1:]               # [t_max, embed_dim]
        norms = valid_emb.norm(dim=1)                       # [t_max]
        mean_norm = norms.mean()
        return ((norms - mean_norm) ** 2).mean()


# ─── Modules ─────────────────────────────────────────────

class MultiScaleDSConv(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw1 = nn.Conv2d(C, C, 3, padding=1, dilation=1, groups=C, bias=False)
        self.dw2 = nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False)
        self.dw4 = nn.Conv2d(C, C, 3, padding=4, dilation=4, groups=C, bias=False)
        self.pw = nn.Conv2d(C * 3, C, 1, bias=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        return self.bn(self.pw(torch.cat([self.dw1(x), self.dw2(x), self.dw4(x)], dim=1)))


class SpikeBlock(nn.Module):
    def __init__(self, C, timestep, spike_mode):
        super().__init__()
        self.spike_in = build_node(timestep, spike_mode)
        self.ms_dsconv = MultiScaleDSConv(C)
        self.spike_mid = build_node(timestep, spike_mode)
        self.pw = nn.Conv2d(C, C, 1, bias=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        identity = x
        x = self.spike_in(x)
        x = self.ms_dsconv(x)
        x = self.spike_mid(x)
        x = self.bn(self.pw(x))
        return x + identity


class SpatialRefineHead(nn.Module):
    """Lightweight CNN to smooth per-pixel gate noise. Residual: learns correction only."""

    def __init__(self, mid=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


# ─── Main Model ──────────────────────────────────────────

class SPADSpikeNet(nn.Module):
    """SPAD dense-fog SNN imaging model.

    Args:
        C:              working channel count
        chunk_size:     frames per chunk (= PLIF timestep within one chunk)
        spike_mode:     neuron type ("plif" / "lif" / "if")
        t_max:          maximum valid ToF bin
        n_freq:         sinusoidal encoding frequency count
        num_blocks:     number of SpikeBlocks
        encoding_mode:  "sinusoidal" (默认) 或 "lut" (可学习查表编码)
        embed_dim:      LUT 模式下的嵌入维度 (自由调整, 默认 16)
        lut_init:       LUT 初始化方式 ("sinusoidal" / "rbf" / "random")
    """

    def __init__(self, C=32, chunk_size=128, spike_mode="plif",
                 t_max=150, n_freq=8, num_blocks=3,
                 encoding_mode="sinusoidal", embed_dim=16,
                 lut_init="sinusoidal"):
        super().__init__()
        self.C = C
        self.chunk_size = chunk_size
        self.t_max = t_max
        self.n_freq = n_freq
        self.encoding_mode = encoding_mode

        if encoding_mode == "lut":
            self.tof_embedding = LearnableTofEmbedding(
                t_max=t_max, embed_dim=embed_dim,
                init_mode=lut_init, n_freq_init=n_freq,
            )
            C_enc = embed_dim
        else:
            self.tof_embedding = None
            C_enc = 2 * n_freq + 1  # 17

        self.stem = nn.Sequential(
            nn.Conv2d(C_enc, C, 1, bias=False),
            nn.BatchNorm2d(C),
            build_node(chunk_size, spike_mode),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.blocks = nn.ModuleList(
            [SpikeBlock(C, chunk_size, spike_mode) for _ in range(num_blocks)]
        )

        self.gate_head = nn.Sequential(
            build_node(chunk_size, spike_mode),
            nn.Conv2d(C, C // 2, 1, bias=False),
            nn.BatchNorm2d(C // 2),
            build_node(chunk_size, spike_mode),
            nn.Conv2d(C // 2, 1, 1, bias=True),
        )

        self.refine = SpatialRefineHead()

    def _forward_chunk(self, data_chunk):
        """Process one chunk: [T, B, H, W] -> gate [T, B, 1, H, W]."""
        T, B, H, W = data_chunk.shape

        valid = ((data_chunk >= 1) & (data_chunk <= self.t_max)).float()
        tof = data_chunk.float() * valid

        if self.encoding_mode == "lut":
            # LUT 编码: 用原始整数 tof 查表, valid mask 在 embedding 内部处理
            x = self.tof_embedding(data_chunk, valid)          # [T, B, D, H, W]
        else:
            x = encode_tof(tof, valid, self.n_freq, self.t_max)  # [T, B, 17, H, W]
        x = x.flatten(0, 1)                                    # [T*B, C_enc, H, W]

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        gate = torch.sigmoid(self.gate_head(x))                # [T*B, 1, H, W]
        gate = gate.view(T, B, 1, H, W)
        return gate, tof, valid

    def forward(self, raw_data):
        """
        Args:
            raw_data: [B, 4096, P]

        Returns:
            dict: depth [B,1,H,W], intensity [B,1,H,W], output [B,2,H,W],
                  gate [P,B,1,H,W], tof [P,B,H,W], valid [P,B,H,W]
        """
        B, _, P = raw_data.shape
        H, W = 64, 64
        T_chunk = self.chunk_size
        device = raw_data.device

        data = raw_data.view(B, H, W, P).permute(3, 0, 1, 2).contiguous()  # [P, B, H, W]

        weighted_sum = torch.zeros(B, 1, H, W, device=device)
        weight_sum = torch.zeros(B, 1, H, W, device=device)
        all_gates, all_tofs, all_valids = [], [], []

        n_chunks = math.ceil(P / T_chunk)
        for i in range(n_chunks):
            t0 = i * T_chunk
            t1 = min(t0 + T_chunk, P)
            chunk = data[t0:t1]                                 # [T_actual, B, H, W]

            T_actual = t1 - t0
            if T_actual < T_chunk:
                pad = torch.zeros(T_chunk - T_actual, B, H, W, device=device)
                chunk = torch.cat([chunk, pad], dim=0)

            gate, tof, valid = self._forward_chunk(chunk)       # [T_chunk, B, ...]

            gate = gate[:T_actual]
            tof = tof[:T_actual]
            valid = valid[:T_actual]

            tof_exp = tof.unsqueeze(2)
            v_exp = valid.unsqueeze(2)
            weighted_sum = weighted_sum + (gate * tof_exp * v_exp).sum(0)
            weight_sum = weight_sum + (gate * v_exp).sum(0)

            all_gates.append(gate)
            all_tofs.append(tof)
            all_valids.append(valid)

            # detach membrane potentials between chunks
            if i < n_chunks - 1:
                for m in self.modules():
                    if hasattr(m, "v") and isinstance(m.v, torch.Tensor):
                        m.v = m.v.detach()

        depth = weighted_sum / (weight_sum + 1e-6)
        intensity = weight_sum / P

        coarse = torch.cat([depth, intensity], dim=1)           # [B, 2, H, W]
        output = self.refine(coarse)                            # [B, 2, H, W]

        functional.reset_net(self)

        out = {
            "output": output,
            "depth": output[:, 0:1],
            "intensity": output[:, 1:2],
            "depth_coarse": depth,
            "intensity_coarse": intensity,
            "gate": torch.cat(all_gates, dim=0),
            "tof": torch.cat(all_tofs, dim=0),
            "valid": torch.cat(all_valids, dim=0),
        }

        # LUT 模式: 附加正则 loss 供训练时使用
        if self.tof_embedding is not None:
            out["lut_smooth"] = self.tof_embedding.smoothness_loss()
            out["lut_norm"] = self.tof_embedding.norm_loss()

        return out


# ─── Test ─────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, B, C = 500, 2, 32
    chunk = 128

    configs = [
        {"encoding_mode": "sinusoidal"},
        {"encoding_mode": "lut", "embed_dim": 16, "lut_init": "sinusoidal"},
        {"encoding_mode": "lut", "embed_dim": 16, "lut_init": "rbf"},
        {"encoding_mode": "lut", "embed_dim": 16, "lut_init": "random"},
        {"encoding_mode": "lut", "embed_dim": 32, "lut_init": "rbf"},
    ]

    for cfg in configs:
        label = f"{cfg['encoding_mode']}"
        if cfg["encoding_mode"] == "lut":
            label += f" dim={cfg['embed_dim']} init={cfg['lut_init']}"
        print(f"\n=== {label} (device={device}) ===")

        model = SPADSpikeNet(C=C, chunk_size=chunk, spike_mode="plif", **cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        fake_data = torch.randint(0, 160, (B, 4096, P), device=device).float()
        result = model(fake_data)

        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:20s} {str(list(v.shape)):25s} [{v.min():.4f}, {v.max():.4f}]")
        print("PASS")
