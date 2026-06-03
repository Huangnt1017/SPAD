"""SPAD 浓雾场景 SNN 成像模型（activation_based 新版后端）。

输入:
    ``raw_data`` 形状为 ``[B, 4096, P]``，其中 0 表示无效或未触发 ToF。

输出:
    字典形式结果，核心输出 ``output`` 形状为 ``[B, 2, 64, 64]``，
    通道 0 为深度，通道 1 为强度。

网络结构:
    正弦 / LUT ToF 编码 -> 等宽 SpikeBlock -> EchoGate -> Gated Moment
    -> 空间精修头。该版本使用 ``spikingjelly.activation_based``，
    神经元以 ``step_mode='m'`` 处理时间序列。
"""

import os
import math

# spikingjelly 导入前设置 CUDA_PATH, 否则 cupy backend 探测时找不到 CUDA headers
if "CUDA_PATH" not in os.environ:
    _default_cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    if os.path.isdir(_default_cuda):
        os.environ["CUDA_PATH"] = _default_cuda

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional


# ─── cupy backend 探测 ────────────────────────────────────

def _probe_cupy_backend() -> bool:
    """模块加载时探测 cupy backend 是否真正可用.

    仅 import cupy 不够: cupy 14.x 在缺少 pytest / CUDA_PATH 时
    import 可能成功但实际运算会抛异常. 因此构造一个 IFNode 并跑一次
    前向来确认 backend='cupy' 端到端可用.

    Returns:
        True 表示 cupy backend 可用, False 表示回退到 torch backend.
    """
    try:
        import cupy  # noqa: F401
        # 构造最轻量的节点做功能性验证
        test_node = neuron.IFNode(step_mode="m", backend="cupy")
        # 输入形状 [T=2, B=1]: activation_based 多步模式最小合法输入
        x = torch.zeros(2, 1, device="cuda")
        test_node(x)
        return True
    except Exception:
        return False


# 模块级常量: 仅在 CUDA 可用时才尝试探测, 避免 CPU 环境启动延迟
_CUPY_AVAILABLE: bool = _probe_cupy_backend() if torch.cuda.is_available() else False


# ─── 神经元工厂 ───────────────────────────────────────────

def build_node(
    spike_mode: str = "plif",
    tau: float = 2.0,
    v_threshold: float = 0.5,
) -> nn.Module:
    """构造 activation_based 脉冲神经元节点.

    新版 API 不需要 timestep 参数; step_mode='m' 表示多步模式,
    输入形状为 [T, B, *], 神经元内部自动沿 T 维展开.

    若 cupy backend 探测通过 (_CUPY_AVAILABLE=True), 自动使用
    backend='cupy' 加速 GPU 上的脉冲计算; 否则回退到 backend='torch'.

    Args:
        spike_mode:   神经元类型 ("plif" / "lif" / "if")
        tau:          膜时间常数 (lif/plif 有效)
        v_threshold:  发放阈值

    Returns:
        配置好 step_mode='m' 的神经元模块
    """
    backend = "cupy" if _CUPY_AVAILABLE else "torch"
    common = dict(v_threshold=v_threshold, detach_reset=True,
                  step_mode="m", backend=backend)
    if spike_mode == "plif":
        return neuron.ParametricLIFNode(init_tau=tau, **common)
    elif spike_mode == "lif":
        return neuron.LIFNode(tau=tau, **common)
    elif spike_mode == "if":
        return neuron.IFNode(**common)
    else:
        raise ValueError(f"不支持的 spike_mode: {spike_mode}, 可选 plif/lif/if")


# ─── 编码 ─────────────────────────────────────────────────

def encode_tof(
    tof: torch.Tensor,
    valid: torch.Tensor,
    n_freq: int = 8,
    t_max: int = 128,
) -> torch.Tensor:
    """正弦位置编码: 将整数 tof 映射为多频率 sin/cos 特征.

    Args:
        tof:    [*, H, W] int/float, raw timestamps
        valid:  [*, H, W] float, 1.0 if valid
        n_freq: 频率对数量
        t_max:  最大有效 bin

    Returns:
        [*, 2*n_freq+1, H, W]  所有值在 [-1, 1]
    """
    v = valid.unsqueeze(-3)                             # [*, 1, H, W]
    t = (tof.float() / t_max).unsqueeze(-3) * v        # [*, 1, H, W]
    channels = [v]
    for i in range(n_freq):
        freq = (i + 1) * math.pi
        channels.append(torch.sin(freq * t) * v)
        channels.append(torch.cos(freq * t) * v)
    return torch.cat(channels, dim=-3)                  # [*, 2*n_freq+1, H, W]


class LearnableTofEmbedding(nn.Module):
    """可学习 LUT 编码: 将整数 ToF bin 查表映射为低维时间位置特征.

    核心是 nn.Embedding(t_max+1, embed_dim), padding_idx=0 保证无效像素为零向量.
    训练时通过成像损失反向更新被访问的 LUT 表项.

    稳定性约束:
      - padding_idx=0: invalid (tof=0) 映射为全零向量, 不参与梯度
      - valid mask: 编码后 × valid, 无效像素不贡献信号
      - smoothness_loss(): 相邻 bin 平滑正则
      - norm_loss(): 范数一致性正则

    Args:
        t_max:       最大有效 tof bin (1~t_max 有效, 0=无效)
        embed_dim:   嵌入维度
        init_mode:   初始化方式 ("sinusoidal" / "rbf" / "random")
        n_freq_init: 正弦初始化时的频率数
        max_norm:    embedding 最大 L2 范数约束 (None=不限)
    """

    def __init__(
        self,
        t_max: int = 128,
        embed_dim: int = 16,
        init_mode: str = "sinusoidal",
        n_freq_init: int = 8,
        max_norm: float = None,
    ):
        super().__init__()
        self.t_max = t_max
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            t_max + 1, embed_dim, padding_idx=0, max_norm=max_norm
        )

        if init_mode == "sinusoidal":
            self._init_sinusoidal(n_freq_init)
        elif init_mode == "rbf":
            self._init_rbf()
        elif init_mode == "random":
            self._init_random()
        else:
            raise ValueError(
                f"不支持的 init_mode: {init_mode}, 可选 sinusoidal/rbf/random"
            )

    def _init_sinusoidal(self, n_freq: int) -> None:
        """正弦编码初始化: 用 sin/cos 值填充 LUT."""
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
                vec = vec[: self.embed_dim]
                while len(vec) < self.embed_dim:
                    vec.append(0.0)
                weight[b] = torch.tensor(vec)

    def _init_rbf(self) -> None:
        """高斯 RBF 初始化: embed_dim 个高斯基函数, 中心均匀分布."""
        with torch.no_grad():
            weight = self.embedding.weight
            weight.zero_()
            centers = torch.linspace(1.0, self.t_max, self.embed_dim)
            sigma = (
                (centers[1] - centers[0]).item() * 0.8
                if self.embed_dim > 1
                else self.t_max / 2.0
            )
            for b in range(1, self.t_max + 1):
                weight[b] = torch.exp(
                    -((b - centers) ** 2) / (2 * sigma ** 2)
                )

    def _init_random(self) -> None:
        """标准正态随机初始化, 缩放到合理范围."""
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
            self.embedding.weight[0].zero_()

    def forward(self, tof: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """查表编码.

        Args:
            tof:   [*, H, W] int/float, raw timestamps (0=invalid)
            valid: [*, H, W] float, 1.0 if valid

        Returns:
            [*, embed_dim, H, W] 编码后特征, invalid 位置为全零
        """
        indices = tof.long().clamp(0, self.t_max)       # [*, H, W]
        # [*, H, W] → [*, H, W, embed_dim]
        emb = self.embedding(indices)
        # [*, H, W, D] → [*, D, H, W]
        dims = list(range(emb.dim()))
        dims.insert(-2, dims.pop(-1))
        emb = emb.permute(*dims).contiguous()           # [*, D, H, W]
        emb = emb * valid.unsqueeze(-3)                 # [*, D, H, W]
        return emb

    def smoothness_loss(self) -> torch.Tensor:
        """相邻 bin 平滑正则: 鼓励相邻时间 bin 编码向量平滑过渡."""
        valid_emb = self.embedding.weight[1:]           # [t_max, embed_dim]
        diff = valid_emb[1:] - valid_emb[:-1]          # [t_max-1, embed_dim]
        return (diff ** 2).mean()

    def norm_loss(self) -> torch.Tensor:
        """范数一致性正则: 鼓励所有有效 bin 的 embedding 范数接近全局均值."""
        valid_emb = self.embedding.weight[1:]           # [t_max, embed_dim]
        norms = valid_emb.norm(dim=1)                   # [t_max]
        mean_norm = norms.mean()
        return ((norms - mean_norm) ** 2).mean()


# ─── 模块 ─────────────────────────────────────────────────

class MultiScaleDSConv(nn.Module):
    """多尺度深度可分离卷积: 三路膨胀卷积 (dilation=1/2/4) 拼接后 1×1 融合.

    dilation 只改变 3×3 卷积核在空间维的采样间隔, padding 与 dilation
    保持一致, 因此输出仍为原始 H×W 分辨率; 它不会下采样, 也不会跳过 P 维帧。

    Args:
        C: 输入/输出通道数
    """

    def __init__(self, C: int):
        super().__init__()
        self.dw1 = nn.Conv2d(C, C, 3, padding=1, dilation=1, groups=C, bias=False)
        self.dw2 = nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False)
        self.dw4 = nn.Conv2d(C, C, 3, padding=4, dilation=4, groups=C, bias=False)
        self.pw = nn.Conv2d(C * 3, C, 1, bias=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            [B, C, H, W]
        """
        # 三路膨胀卷积拼接: [B, C, H, W] × 3 → [B, 3C, H, W]
        return self.bn(self.pw(torch.cat([self.dw1(x), self.dw2(x), self.dw4(x)], dim=1)))


class SpikeBlock(nn.Module):
    """脉冲残差块: spike_in → MultiScaleDSConv → spike_mid → pw+BN + 残差.

    新版 API: 神经元 step_mode='m', 输入 [T, B, C, H, W];
    ANN 子模块 (DSConv/BN) 通过 seq_to_ann_forward 在时间维展开.

    Args:
        C:          通道数
        spike_mode: 神经元类型
    """

    def __init__(self, C: int, spike_mode: str):
        super().__init__()
        self.spike_in = build_node(spike_mode)
        self.ms_dsconv = MultiScaleDSConv(C)
        self.spike_mid = build_node(spike_mode)
        self.pw = nn.Conv2d(C, C, 1, bias=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]

        Returns:
            [T, B, C, H, W]
        """
        identity = x
        x = self.spike_in(x)                            # [T, B, C, H, W]
        # ANN 模块不感知时间维, 用 seq_to_ann_forward 展开 T
        x = functional.seq_to_ann_forward(x, self.ms_dsconv)   # [T, B, C, H, W]
        x = self.spike_mid(x)                           # [T, B, C, H, W]
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.pw, self.bn)
        )                                               # [T, B, C, H, W]
        return x + identity


class SpatialRefineHead(nn.Module):
    """置信度调制的轻量 CNN 精修头, 并将输出约束到物理有效范围.

    Args:
        mid: 中间通道数
        depth_range: depth 输出最大 ToF bin
    """

    def __init__(self, mid: int = 8, depth_range: float = 128.0):
        super().__init__()
        self.depth_range = float(depth_range)
        self.net = nn.Sequential(
            nn.Conv2d(3, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 3, padding=1, bias=False),
        )

    def forward(
        self,
        coarse: torch.Tensor,
        confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            coarse: [B, 2, H, W], ch0=depth, ch1=intensity
            confidence: [B, 1, H, W], 置信度越高允许越大的残差修正

        Returns:
            [B, 2, H, W]
        """
        if confidence is None:
            confidence = torch.ones_like(coarse[:, 0:1])
        confidence = confidence.clamp(0.0, 1.0)
        coarse_norm = torch.cat(
            [
                (coarse[:, 0:1] / self.depth_range).clamp(0.0, 1.0),
                coarse[:, 1:2].clamp(0.0, 1.0),
            ],
            dim=1,
        )
        residual = self.net(torch.cat([coarse_norm, confidence], dim=1))
        output_norm = (coarse_norm + residual * confidence).clamp(0.0, 1.0)
        depth = output_norm[:, 0:1] * self.depth_range
        intensity = output_norm[:, 1:2]
        return torch.cat([depth, intensity], dim=1)


# ─── Stem (ANN-only, 在 chunk 展开后调用) ─────────────────

class _Stem(nn.Module):
    """Stem 网络: 将编码特征映射到工作通道, 纯 ANN 无脉冲神经元.

    在 _forward_chunk 中对展平后的 [T*B, C_enc, H, W] 直接调用,
    不需要 seq_to_ann_forward 包装.

    Args:
        C_enc: 输入编码通道数
        C:     输出工作通道数
        spike_mode: 用于 stem 内部脉冲层的神经元类型
    """

    def __init__(self, C_enc: int, C: int, spike_mode: str):
        super().__init__()
        self.conv1 = nn.Conv2d(C_enc, C, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.spike = build_node(spike_mode)             # step_mode='m', 需要 [T,B,C,H,W]
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C_enc, H, W]

        Returns:
            [T, B, C, H, W]
        """
        # conv1+bn1: ANN, 展开时间维
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.conv1, self.bn1)
        )                                               # [T, B, C, H, W]
        x = self.spike(x)                               # [T, B, C, H, W]
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.conv2, self.bn2)
        )                                               # [T, B, C, H, W]
        return x


class _GateHead(nn.Module):
    """Gate 头: 脉冲 → 1×1 降维 → BN → 脉冲 → 1×1 → sigmoid.

    Args:
        C:          输入通道数
        spike_mode: 神经元类型
    """

    def __init__(self, C: int, spike_mode: str):
        super().__init__()
        self.spike1 = build_node(spike_mode)
        self.conv1 = nn.Conv2d(C, C // 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(C // 2)
        self.spike2 = build_node(spike_mode)
        self.conv2 = nn.Conv2d(C // 2, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]

        Returns:
            [T, B, 1, H, W]  sigmoid gate
        """
        x = self.spike1(x)                              # [T, B, C, H, W]
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.conv1, self.bn1)
        )                                               # [T, B, C//2, H, W]
        x = self.spike2(x)                              # [T, B, C//2, H, W]
        x = functional.seq_to_ann_forward(x, self.conv2)  # [T, B, 1, H, W]
        return torch.sigmoid(x)


# ─── 主模型 ───────────────────────────────────────────────

class SPADSpikeNet(nn.Module):
    """SPAD dense-fog SNN 成像模型 (activation_based API).

    使用 spikingjelly.activation_based, 神经元 step_mode='m',
    ANN 子模块通过 functional.seq_to_ann_forward 在时间维展开.

    Args:
        C:             工作通道数
        chunk_size:    每个 chunk 的帧数 (= 一次 forward 的时间步数 T)
        spike_mode:    神经元类型 ("plif" / "lif" / "if")
        t_max:         最大有效 ToF bin
        n_freq:        正弦编码频率对数量
        num_blocks:    SpikeBlock 数量
        encoding_mode: "sinusoidal" 或 "lut"
        embed_dim:     LUT 模式嵌入维度
        lut_init:      LUT 初始化方式 ("sinusoidal" / "rbf" / "random")
    """

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
    ):
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
            C_enc = 2 * n_freq + 1                      # 默认 17

        self.stem = _Stem(C_enc, C, spike_mode)
        self.blocks = nn.ModuleList(
            [SpikeBlock(C, spike_mode) for _ in range(num_blocks)]
        )
        self.gate_head = _GateHead(C, spike_mode)
        self.refine = SpatialRefineHead(depth_range=t_max)

    def _encode_chunk(
        self, chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对一个 chunk 做 valid mask + 编码.

        Args:
            chunk: [T, B, H, W] raw timestamps

        Returns:
            x:     [T, B, C_enc, H, W] 编码特征
            tof:   [T, B, H, W] masked float timestamps
            valid: [T, B, H, W] 有效像素 mask
        """
        valid = ((chunk >= 1) & (chunk <= self.t_max)).float()
        tof = chunk.float() * valid

        if self.encoding_mode == "lut":
            # LUT 查表: [T, B, H, W] → [T, B, embed_dim, H, W]
            x = self.tof_embedding(chunk, valid)
        else:
            # 正弦编码: [T, B, H, W] → [T, B, 2*n_freq+1, H, W]
            x = encode_tof(tof, valid, self.n_freq, self.t_max)
        return x, tof, valid

    def _forward_chunk(
        self, data_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理一个 chunk: [T, B, H, W] → gate [T, B, 1, H, W].

        神经元在 chunk 内保持膜电位状态; chunk 间通过 detach 截断梯度.
        """
        x, tof, valid = self._encode_chunk(data_chunk)  # [T, B, C_enc, H, W]

        x = self.stem(x)                                # [T, B, C, H, W]
        for block in self.blocks:
            x = block(x)                                # [T, B, C, H, W]

        gate = self.gate_head(x)                        # [T, B, 1, H, W]
        return gate, tof, valid

    def forward(self, raw_data: torch.Tensor) -> dict:
        """
        Args:
            raw_data: [B, N, P]  raw ToF timestamps, N = H*W (默认 4096 = 64×64)

        Returns:
            dict:
                output:            [B, 2, H, W]  精修后深度+强度
                depth:             [B, 1, H, W]
                intensity:         [B, 1, H, W]
                depth_coarse:      [B, 1, H, W]
                intensity_coarse:  [B, 1, H, W]
                gate:              [P, B, 1, H, W]
                tof:               [P, B, H, W]
                valid:             [P, B, H, W]
                lut_smooth:        scalar (仅 encoding_mode="lut")
                lut_norm:          scalar (仅 encoding_mode="lut")
        """
        B, N, P = raw_data.shape
        # N = H*W, 假设正方形; 支持任意分辨率 (如 16×16 用于测试, 64×64 用于生产)
        H = W = int(N ** 0.5)
        T_chunk = self.chunk_size
        device = raw_data.device

        # [B, 4096, P] → [B, H, W, P] → [P, B, H, W]
        data = raw_data.view(B, H, W, P).permute(3, 0, 1, 2).contiguous()

        weighted_sum = torch.zeros(B, 1, H, W, device=device)
        weight_sum = torch.zeros(B, 1, H, W, device=device)
        all_gates, all_tofs, all_valids = [], [], []

        n_chunks = math.ceil(P / T_chunk)
        for i in range(n_chunks):
            t0 = i * T_chunk
            t1 = min(t0 + T_chunk, P)
            chunk = data[t0:t1]                         # [T_actual, B, H, W]

            T_actual = t1 - t0
            if T_actual < T_chunk:
                # 末尾 chunk 补零帧, 保证神经元 timestep 一致
                pad = torch.zeros(T_chunk - T_actual, B, H, W, device=device)
                chunk = torch.cat([chunk, pad], dim=0)  # [T_chunk, B, H, W]

            gate, tof, valid = self._forward_chunk(chunk)  # [T_chunk, B, ...]

            # 截掉补零帧
            gate = gate[:T_actual]                      # [T_actual, B, 1, H, W]
            tof = tof[:T_actual]                        # [T_actual, B, H, W]
            valid = valid[:T_actual]                    # [T_actual, B, H, W]

            tof_exp = tof.unsqueeze(2)                  # [T_actual, B, 1, H, W]
            v_exp = valid.unsqueeze(2)                  # [T_actual, B, 1, H, W]
            # 加权累积: Σ gate * tof * valid
            weighted_sum = weighted_sum + (gate * tof_exp * v_exp).sum(0)
            weight_sum = weight_sum + (gate * v_exp).sum(0)

            all_gates.append(gate)
            all_tofs.append(tof)
            all_valids.append(valid)

            # chunk 间截断膜电位梯度, 防止 BPTT 跨 chunk 爆炸
            if i < n_chunks - 1:
                functional.detach_net(self)

        depth = weighted_sum / (weight_sum + 1e-6)      # [B, 1, H, W]
        intensity = weight_sum / P                      # [B, 1, H, W]
        confidence = (weight_sum / (weight_sum + 1.0)).clamp(0.0, 1.0)

        # [B, 1, H, W] × 2 → [B, 2, H, W]
        coarse = torch.cat([depth, intensity], dim=1)
        output = self.refine(coarse, confidence)        # [B, 2, H, W]

        functional.reset_net(self)

        out = {
            "output": output,
            "depth": output[:, 0:1],
            "intensity": output[:, 1:2],
            "depth_coarse": depth,
            "intensity_coarse": intensity,
            "confidence": confidence,
            "gate": torch.cat(all_gates, dim=0),        # [P, B, 1, H, W]
            "tof": torch.cat(all_tofs, dim=0),          # [P, B, H, W]
            "valid": torch.cat(all_valids, dim=0),      # [P, B, H, W]
        }

        if self.tof_embedding is not None:
            out["lut_smooth"] = self.tof_embedding.smoothness_loss()
            out["lut_norm"] = self.tof_embedding.norm_loss()

        return out


def _benchmark_forward_5d_full_network(
    model: SPADSpikeNet,
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the full learnable network path for benchmark input [T, B, C, H, W].

    The normal SPADSpikeNet.forward() takes raw ToF data [B, H*W, P]. This
    benchmark starts from an already encoded feature tensor [T, B, C, H, W],
    then exercises stem, all SpikeBlocks, gate_head, temporal aggregation,
    and refine head.
    """
    T = x.shape[0]

    x = model.stem(x)
    for block in model.blocks:
        x = block(x)

    gate = model.gate_head(x)  # [T, B, 1, H, W]
    time_bins = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
    time_bins = time_bins.view(T, 1, 1, 1, 1)

    weight_sum = gate.sum(0)  # [B, 1, H, W]
    depth = (gate * time_bins).sum(0) / (weight_sum + 1e-6)
    intensity = weight_sum / T
    confidence = (weight_sum / (weight_sum + 1.0)).clamp(0.0, 1.0)
    output = model.refine(torch.cat([depth, intensity], dim=1), confidence)

    return {
        "output": output,
        "depth_coarse": depth,
        "intensity_coarse": intensity,
        "confidence": confidence,
        "gate": gate,
    }


def _benchmark_is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _benchmark_cleanup_cuda(model=None) -> None:
    if model is not None:
        try:
            functional.reset_net(model)
        except Exception:
            pass
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def run_5d_memory_benchmark() -> None:
    """Benchmark CUDA memory for input shape [T, B, C, 64, 64]."""
    import csv

    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark measures CUDA memory only.")
        return

    device = "cuda"
    H = W = 64
    t_values = list(range(50, 501, 50))
    b_values = [4, 8, 16, 32]
    c_values = [4, 8, 16, 24, 32]
    measure_backward = True
    physical_limit_gb = 12
    physical_limit_mb = physical_limit_gb * 1024

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"cupy backend: {'available (backend=cupy)' if _CUPY_AVAILABLE else 'unavailable, fallback to torch'}")
    print("Benchmark input: [T, B, C, 64, 64]")
    print("Network path: stem -> SpikeBlocks -> gate_head -> temporal aggregation -> refine")
    print(f"Mode: {'forward + backward' if measure_backward else 'forward only'}")
    print(f"Stop rule: peak_allocated > {physical_limit_gb} GB, skip larger T for current B/C")

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "snn_5d_memory_benchmark.csv",
    )
    rows = []

    for C in c_values:
        for B in b_values:
            print(f"\n=== C={C:2d}, B={B:2d}, H=W={H} ===")
            for T in t_values:
                model = None
                x = None
                result = None
                loss = None
                try:
                    _benchmark_cleanup_cuda()
                    torch.cuda.reset_peak_memory_stats()

                    # encoding_mode='lut' with embed_dim=C makes stem input
                    # channels equal to the benchmark tensor channel count C.
                    model = SPADSpikeNet(
                        C=C,
                        chunk_size=T,
                        spike_mode="plif",
                        t_max=max(500, T),
                        num_blocks=3,
                        encoding_mode="lut",
                        embed_dim=C,
                        lut_init="sinusoidal",
                    ).to(device)
                    model.train()
                    n_params = sum(p.numel() for p in model.parameters())

                    x = torch.randn(T, B, C, H, W, device=device)
                    input_mb = x.numel() * x.element_size() / 1024**2

                    result = _benchmark_forward_5d_full_network(model, x)
                    loss = result["output"].mean()
                    if measure_backward:
                        loss.backward()

                    torch.cuda.synchronize()
                    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
                    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
                    over_physical_limit = peak_allocated_mb > physical_limit_mb
                    status = "OVER_12GB" if over_physical_limit else "PASS"

                    rows.append({
                        "T": T,
                        "B": B,
                        "C": C,
                        "H": H,
                        "W": W,
                        "params": n_params,
                        "input_mb": round(input_mb, 2),
                        "peak_allocated_mb": round(peak_allocated_mb, 2),
                        "peak_reserved_mb": round(peak_reserved_mb, 2),
                        "status": status,
                    })
                    print(
                        f"T={T:3d}  input={input_mb:8.1f} MB  "
                        f"peak_alloc={peak_allocated_mb:9.1f} MB  "
                        f"peak_reserved={peak_reserved_mb:9.1f} MB  {status}"
                    )
                    if over_physical_limit:
                        print(
                            f"T={T:3d}  peak_allocated exceeds {physical_limit_gb} GB, "
                            "skip larger T for this B/C"
                        )
                        break

                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if not _benchmark_is_oom_error(exc):
                        raise
                    rows.append({
                        "T": T,
                        "B": B,
                        "C": C,
                        "H": H,
                        "W": W,
                        "params": "",
                        "input_mb": "",
                        "peak_allocated_mb": "",
                        "peak_reserved_mb": "",
                        "status": "OOM",
                    })
                    print(f"T={T:3d}  OOM, skip larger T for this B/C")
                    break

                finally:
                    del loss, result, x
                    if model is not None:
                        _benchmark_cleanup_cuda(model)
                    del model

    fieldnames = [
        "T",
        "B",
        "C",
        "H",
        "W",
        "params",
        "input_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark results to: {csv_path}")


# ─── 显存测试 ─────────────────────────────────────────────

if __name__ == "__main__":
    run_5d_memory_benchmark()

if False and __name__ == "__main__":
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"cupy backend: {'可用 (backend=cupy)' if _CUPY_AVAILABLE else '不可用, 回退到 torch'}")

    # 小尺寸输入: H=W=16 (256像素), P=16帧, chunk=16
    # 用于快速验证 forward/backward 正确性, 不测显存上限
    H, W, P, C = 16, 16, 16, 8
    chunk = 16

    configs = [
        {"encoding_mode": "sinusoidal"},
        {"encoding_mode": "lut", "embed_dim": 16, "lut_init": "sinusoidal"},
        {"encoding_mode": "lut", "embed_dim": 16, "lut_init": "rbf"},
        {"encoding_mode": "lut", "embed_dim": 32, "lut_init": "rbf"},
    ]

    for cfg in configs:
        label = cfg["encoding_mode"]
        if cfg["encoding_mode"] == "lut":
            label += f" dim={cfg['embed_dim']} init={cfg['lut_init']}"
        print(f"\n=== {label} ===")

        for batch_size in [4, 8, 16, 32]:
            try:
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    gc.collect()

                model = SPADSpikeNet(C=C, chunk_size=chunk, spike_mode="plif",
                                     **cfg).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                # [B, H*W, P] 小尺寸输入
                fake_data = torch.randint(0, 140, (batch_size, H * W, P), device=device).float()
                result = model(fake_data)
                loss = result["output"].mean()
                if "lut_smooth" in result:
                    loss = loss + 0.01 * result["lut_smooth"] + 0.005 * result["lut_norm"]
                loss.backward()

                if device == "cuda":
                    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
                    print(f"  B={batch_size:2d}  params={n_params:,}  peak={peak_mb:.0f} MB  PASS")
                else:
                    print(f"  B={batch_size:2d}  params={n_params:,}  PASS (CPU)")

                del model, fake_data, result, loss
            except torch.cuda.OutOfMemoryError:
                print(f"  B={batch_size:2d}  OOM — 停止扫描")
                break
