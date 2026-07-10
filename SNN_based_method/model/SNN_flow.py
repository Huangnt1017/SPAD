"""SPAD 跨帧流式 SNN 模型。

``SNNFlowNet`` 继承 ``SPADSpikeNet`` 的全部子网络 (stem / blocks / gate_head /
refine / 编码层), 但在 ``__init__`` 末尾把所有 BatchNorm2d 替换为 GroupNorm,
以消除 ``seq_to_ann_forward`` 摊平 [T,B,..]→[T*B,..] 后 BN batch 统计带来的
**chunk 内跨帧未来泄露** (训练模式下实测可达数十 bin 的深度偏差)。换 GN 后
归一化只在每样本 (C,H,W) 上做, 与 T/batch 维无关, 任意 chunk 切法、train/eval
都严格因果。代价: 因 BN→GN 参数结构不同, **不能再直接加载 ``model_backend="new"``
的权重**, flow 模型需独立训练。

与父类的核心区别:

- ``SPADSpikeNet.forward`` 把一组 P 页一次性吞入, 开头 reset 膜电位、结尾再 reset,
  即一次 forward = 一个完整样本的独立推理。
- ``SNNFlowNet`` 把膜电位和累积量 (weighted_sum / weight_sum / gate_hist /
  raw_hist) 持久化到 ``stream_step`` 之间, 因此可以**每来一(批)页就更新状态、
  随时读出当前深度/强度估计**, 不必等全部 P 页到齐。这对应硬件节奏
  (SPAD 25000 frame/s 逐帧到达), 价值在低延迟与恒定显存。

针对浓雾三死结的修复 (与训练侧 loss 修改配套):

1. 死结 1 (逐光子 gate 看不到直方图形状): ``use_stream_context=True`` 把 raw
   直方图的 running 统计 (雾峰位置/密度/谷后占比/集中度) 作为 4 个额外通道
   逐 chunk 注入 stem, 严格因果; 配合 ``spike_mode='plif_mt'`` 的 per-channel
   多时间尺度 PLIF (见 SNN_new.build_node) 和光子级 gate 监督
   (loss.GatePhotonSupervisionLoss) 使用。
2. 死结 2 (argmax 梯度饥饿): 非 valley 路径的 ``_finalize_gated_peak_maps``
   改用温度 softargmax (见 SNN_new), 梯度可达全部 bin。
3. 死结 3 (强度量纲失配): ``_ValleyHumpHead`` v2 用谷后 hump prominence 与
   雾峰高度的 log 对比度 + 可学习标定输出 intensity, 对 P 和雾级近似不变。

跨帧流式 (page-streaming) 已实现; 周期内流式 (把 ToF 当激光周期内的脉冲发放时刻,
让膜电位沿 ToF 轴做匹配滤波) 仅预留扩展点 ``_encode_frames`` + ``intra_period_mode``,
未在本类实现具体动力学。

API 速览:
    stream_reset(B, H, W, device): 清零累积量 + reset 膜电位, 开始一段新序列。
    stream_step(frames): 喂入 [T, B, H, W] 新页 (T 可为 1), 累积状态, 返回当前估计。
    stream_readout(): 用当前累积量算一次估计, 不改变状态。
    forward(raw_data): 与父类签名/返回一致的批量接口, 内部走流式循环, 供训练复用。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional

from SNN_based_method.model.SNN_new import (
    SPADSpikeNet,
    _accumulate_gate_hist,
    _collect_module_spike_stats,
    _finalize_gated_peak_maps,
    _reset_module_spike_stats,
    build_node,
)


def _pick_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """选不超过 ``max_groups`` 且能整除 ``num_channels`` 的最大约数, 兜底为 1。"""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _replace_bn_with_gn(root: nn.Module) -> int:
    """把 ``root`` 子树里所有 BatchNorm2d 原地替换为 GroupNorm, 返回替换数。

    GroupNorm 只在每个样本自身的 (C, H, W) 上做统计, 与 T 维和 batch 维都无关。
    因此 ``seq_to_ann_forward`` 把 [T, B, C, H, W] 摊平成 [T*B, C, H, W] 后,
    每一帧独立归一化, 不再借由 BN 的 batch 统计混入同 chunk 的未来帧 ——
    无论 chunk 怎么切、train 还是 eval, 都严格因果且行为一致。
    """
    replaced = 0
    for name, child in list(root.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                num_groups=_pick_num_groups(num_channels),
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine,
            )
            gn = gn.to(child.weight.device if child.affine else next(root.parameters()).device)
            setattr(root, name, gn)
            replaced += 1
        else:
            replaced += _replace_bn_with_gn(child)
    return replaced


class _SlimStem(nn.Module):
    """瘦身版 Stem: 把父类 conv2 的 dense 3x3 换成 depthwise 3x3 + pointwise 1x1。

    父类 ``_Stem`` 的 conv2 是 (C, C, 3, 3) dense 卷积, 单层就占全模型约 57% 参数,
    而它只承担"工作通道内的局部空间上下文 + 通道混合"。这里拆成 depthwise(空间)
    + pointwise(通道) 两步, 保住 3x3 感受野和全通道混合, 参数从 C*C*9 降到
    C*9 + C*C, 把省下的预算让给 backbone 和 gate head。

    结构 (BN 会在构造后被 _replace_bn_with_gn 统一换成 GroupNorm 以保持因果):
        [T, B, C_enc, H, W]
          -> Conv1x1 + BN              (融合 ToF 编码通道)
          -> spike
          -> DWConv3x3 + PWConv1x1 + BN (局部空间 + 通道混合)
          -> [T, B, C, H, W]
    """

    def __init__(
        self,
        c_enc: int,
        channels: int,
        spike_mode: str,
        spike_tau: float,
        spike_v_threshold: float,
        spike_v_reset: float | None,
        spike_backend: str,
        spike_tau_max: float | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_enc, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.spike = build_node(
            spike_mode,
            tau=spike_tau,
            v_threshold=spike_v_threshold,
            v_reset=spike_v_reset,
            spike_backend=spike_backend,
            channels=channels,
            tau_max=spike_tau_max,
        )
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.conv1, self.bn1)
        )
        x = self.spike(x)
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.dw, self.pw, self.bn2)
        )
        return x


class _FatGateHead(nn.Module):
    """加宽 + 加空间上下文的 gate 头。

    父类 ``_GateHead`` 全是 1x1 卷积, 中间维 C//2, gate 决策**完全没有空间上下文**——
    每个像素的 gate 只看自己。但目标回波是空间相干的 (目标是连片形状, 雾是弥散背景),
    让 gate 看邻域能直接帮选择性 (前几轮诊断的 SNN gate 选择性瓶颈)。这里:
      - 中间维从 C//2 加宽到 C;
      - 插入一个 depthwise 3x3, 给 gate 决策引入局部邻域。

    结构 (BN -> GroupNorm 同上):
        [T, B, C, H, W]
          -> spike1
          -> Conv1x1(C->C) + BN
          -> spike2
          -> DWConv3x3 + BN
          -> Conv1x1(C->1, bias)
          -> sigmoid -> [T, B, 1, H, W]
    """

    def __init__(
        self,
        channels: int,
        spike_mode: str,
        spike_tau: float,
        spike_v_threshold: float,
        spike_v_reset: float | None,
        spike_backend: str,
        spike_tau_max: float | None = None,
    ) -> None:
        super().__init__()
        self.spike1 = build_node(
            spike_mode,
            tau=spike_tau,
            v_threshold=spike_v_threshold,
            v_reset=spike_v_reset,
            spike_backend=spike_backend,
            channels=channels,
            tau_max=spike_tau_max,
        )
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.spike2 = build_node(
            spike_mode,
            tau=spike_tau,
            v_threshold=spike_v_threshold,
            v_reset=spike_v_reset,
            spike_backend=spike_backend,
            channels=channels,
            tau_max=spike_tau_max,
        )
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spike1(x)
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.conv1, self.bn1)
        )
        x = self.spike2(x)
        x = functional.seq_to_ann_forward(
            x, nn.Sequential(self.dw, self.bn2)
        )
        x = functional.seq_to_ann_forward(x, self.conv2)
        return torch.sigmoid(x)


class _ValleyHumpHead(nn.Module):
    """谷后 hump 物理检测头 v2: 雾峰 → 有界软谷 → 谷后窗口内基线扣除质心。

    复现离线探针验证过的规则。浓雾后向散射服从 Beer-Lambert 单调衰减 (早峰),
    目标是雾峰之后、衰减谷底之后重新隆起的第二峰。逐光子 gate 因在累积前决策、
    看不到直方图形状, 只能学到 "全开" 平凡解 → depth 被锁死在雾峰; 本头改在
    **累积后的直方图**上做选择, 给出物理量纲不变的 depth。

    v2 相比 v1 的四处修正:
      1. **雾峰/谷位从 raw 直方图估计** (不乘 gate)。v1 用 gate 直方图估计雾峰,
         其工作前提是 gate 全开; 一旦光子级监督教会 gate 压雾, 雾峰会从 gate
         直方图上消失, 谷位随之漂移。raw 直方图不受 gate 影响, 两者彻底解耦:
         gate 负责提升谷后 hump 的信噪比, raw 负责稳定的雾结构定位。
      2. **有界谷偏移**: v1 用 softplus 无上界且可塌缩到 0 (退回雾峰)。改为
         ``offset = min + (max-min)·sigmoid(·)``, 物理初始化 ≈ offset_init。
      3. **谷后有限窗口 + 基线扣除**: v1 在全 bin 范围做 softargmax, 平坦远尾
         会把质心拉偏。改为 [valley, valley+hump_window] 软窗口, 窗内先扣除
         均值基线再取 prominence, 背景像素的平坦雾尾 prominence≈0。
      4. **对比度量纲 intensity** (死结 3): v1 的 peak/P 在浓雾下物理上限只有
         ~0.02-0.2, 够不着 0-1 的 label 置信图。改为 log 域雾峰对比度 + 可学习
         仿射标定: ``sigmoid(a·(log1p(hump) - log1p(fog)) + c)``, 对 P 和雾级
         近似不变, 背景像素 (无 hump) 自然趋 0。

    全流程可微; depth 始终是直方图 bin 的可微质心, 网络只学 "雾衰描述 + 谷位置",
    不直接回归深度, 符合 gate-moment 的物理量纲不变原则。

    Args:
        t_max: ToF 最大 bin (有效 bin 为 1..t_max)。
        spatial_pool: 空间聚合核大小 (奇数), 提目标 SNR。
        valley_hidden: 谷偏移预测网络的隐藏通道数。
        fog_sharpness: 雾峰 softargmax 温度 (大=更尖锐)。
        hump_sharpness: 谷后 hump softargmax 温度。
        gate_beta_init: 软谷门陡度初值 (可学习, softplus 参数化保正)。
        valley_offset_init: 谷相对雾峰的初始偏移 (bin), 物理初始化用。
        valley_offset_min: 谷偏移下界 (bin), 防止塌缩回雾峰。
        valley_offset_max: 谷偏移上界 (bin), 防止越过目标窗。
        hump_window: 谷后搜峰窗口长度 (bin), 之外的远尾不参与质心。
    """

    def __init__(
        self,
        t_max: int,
        spatial_pool: int = 5,
        valley_hidden: int = 16,
        fog_sharpness: float = 6.0,
        hump_sharpness: float = 8.0,
        gate_beta_init: float = 2.0,
        valley_offset_init: float = 11.0,
        valley_offset_min: float = 3.0,
        valley_offset_max: float = 40.0,
        hump_window: float = 48.0,
    ) -> None:
        super().__init__()
        self.t_max = int(t_max)
        if spatial_pool % 2 == 0:
            raise ValueError("spatial_pool must be odd")
        if not valley_offset_min < valley_offset_init < valley_offset_max:
            raise ValueError(
                "valley offset bounds must satisfy min < init < max, got "
                f"min={valley_offset_min}, init={valley_offset_init}, max={valley_offset_max}"
            )
        if hump_window <= 0:
            raise ValueError(f"hump_window must be positive, got {hump_window}")
        if gate_beta_init <= 0:
            raise ValueError(f"gate_beta_init must be positive, got {gate_beta_init}")
        self.spatial_pool = int(spatial_pool)
        self.fog_sharpness = float(fog_sharpness)
        self.hump_sharpness = float(hump_sharpness)
        self.offset_min = float(valley_offset_min)
        self.offset_max = float(valley_offset_max)
        self.hump_window = float(hump_window)
        self.eps = 1e-6

        # bins 值 1..t_max, 形状 [1, t_max, 1, 1] 便于沿时间维广播。
        self.register_buffer(
            "bins",
            torch.arange(1, self.t_max + 1, dtype=torch.float32).view(1, self.t_max, 1, 1),
        )

        # 软谷门陡度: softplus 参数化保正, 初始化使 softplus(raw) = gate_beta_init。
        self.gate_beta_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(gate_beta_init))))
        )

        # intensity 对比度标定 (死结 3): sigmoid(a·(log1p(hump) - log1p(fog)) + c)。
        self.intensity_scale = nn.Parameter(torch.tensor(1.0))
        self.intensity_bias = nn.Parameter(torch.tensor(0.0))

        # 谷偏移预测: 从归一化 raw 直方图 (t_max 通道) → 1 偏移比例。
        self.valley_net = nn.Sequential(
            nn.Conv2d(self.t_max, valley_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(valley_hidden, 1, 1, bias=True),
        )
        # 物理初始化: 末层权重设为极小随机 (非纯零)、bias 使 sigmoid 输出对应
        # offset_init。bias 主导 → 未训练前向的谷偏移 ≈ valley_offset_init, 谷后
        # hump 一开始就落在目标 bin (探针验证)。但**不能纯零**: 零权重会在反向时
        # 把梯度挡在末层之前, 使 valley_net 上游 grad 恒为 0 (冒烟实测)。
        last_conv = self.valley_net[-1]
        nn.init.normal_(last_conv.weight, mean=0.0, std=1e-3)
        init_ratio = (float(valley_offset_init) - self.offset_min) / (
            self.offset_max - self.offset_min
        )
        # logit(init_ratio): sigmoid 反函数, 保证初始 offset ≈ valley_offset_init
        nn.init.constant_(
            last_conv.bias, math.log(init_ratio / (1.0 - init_ratio))
        )

    def forward(
        self,
        gate_hist: torch.Tensor,
        raw_hist: torch.Tensor,
        weight_sum: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            gate_hist: [t_max+1, B*H*W] gate 加权 ToF 直方图, bin0 保留给无效 ToF。
            raw_hist: [t_max+1, B*H*W] 不加 gate 的 valid 光子计数直方图 (雾结构定位用)。
            weight_sum: [B, 1, H, W], gate·valid 累积量, 用于 support。

        Returns:
            dict: depth/intensity/confidence/support/selectivity (均 [B,1,H,W]),
            以及 ``stats`` 子 dict (fog_peak/offset/valley/beta 等标量诊断量, 已 detach)。
        """
        if gate_hist.dim() != 2 or raw_hist.dim() != 2:
            raise ValueError("gate_hist / raw_hist must have shape [t_max+1, B*H*W]")
        if weight_sum.dim() != 4 or weight_sum.shape[1] != 1:
            raise ValueError("weight_sum must have shape [B, 1, H, W]")
        batch_size, _, height, width = weight_sum.shape
        t_max = self.t_max
        pool = self.spatial_pool

        # 丢掉无效 bin0; [t_max, B*H*W] → [B, t_max, H, W]
        raw = (
            raw_hist[1:].to(weight_sum.dtype)
            .reshape(t_max, batch_size, height, width)
            .permute(1, 0, 2, 3)
        )
        gated = (
            gate_hist[1:].to(weight_sum.dtype)
            .reshape(t_max, batch_size, height, width)
            .permute(1, 0, 2, 3)
        )

        # 1. 空间聚合 (固定 avgpool, 保持尺寸); 目标成片 → SNR 提升, 雾随机被平滑。
        raw_pooled = F.avg_pool2d(raw, pool, stride=1, padding=pool // 2)
        gated_pooled = F.avg_pool2d(gated, pool, stride=1, padding=pool // 2)

        bins = self.bins.to(raw_pooled.dtype)                   # [1, t_max, 1, 1]

        # 2. 雾峰软定位 (raw 直方图, 与 gate 解耦): 逐像素归一化后 softargmax。
        raw_norm = raw_pooled / (raw_pooled.amax(dim=1, keepdim=True) + self.eps)
        w_fog = F.softmax(raw_norm * self.fog_sharpness, dim=1)
        fog_peak = (w_fog * bins).sum(dim=1, keepdim=True)      # [B, 1, H, W]

        # 3. 有界软谷: valley = fog_peak + (min + span·sigmoid(net(raw 形状)))。
        offset_ratio = torch.sigmoid(self.valley_net(raw_norm))  # [B, 1, H, W] ∈ (0,1)
        offset = self.offset_min + (self.offset_max - self.offset_min) * offset_ratio
        valley = fog_peak + offset

        # 4. 谷后有限软窗口: 左沿压雾段, 右沿截远尾, beta 可学习。
        gate_beta = F.softplus(self.gate_beta_raw)
        window = torch.sigmoid(gate_beta * (bins - valley)) * torch.sigmoid(
            gate_beta * (valley + self.hump_window - bins)
        )                                                       # [B, t_max, H, W]
        masked = gated_pooled * window

        # 5. 窗内基线扣除 → prominence: 平坦雾尾被扣平, 只有真 hump 突出。
        window_mass = masked.sum(dim=1, keepdim=True)
        window_len = window.sum(dim=1, keepdim=True) + self.eps
        baseline = window_mass / window_len
        prominence = F.relu(masked - baseline)                  # [B, t_max, H, W]

        # 6. 谷后 hump 质心: prominence 归一化 + log(window) 屏蔽窗外 → softargmax。
        prom_norm = prominence / (prominence.amax(dim=1, keepdim=True) + self.eps)
        hump_logits = prom_norm * self.hump_sharpness + torch.log(window + self.eps)
        w_hump = F.softmax(hump_logits, dim=1)
        depth = (w_hump * bins).sum(dim=1, keepdim=True)        # [B, 1, H, W]

        # 7. 对比度 intensity (死结 3): 谷后 hump 突出量 vs 谷前雾峰高度, log 域
        #    可学习仿射标定到 label 的 0-1 置信量纲; 对 P 与雾级近似不变。
        hump_peak = prominence.amax(dim=1, keepdim=True)        # [B, 1, H, W]
        fog_mask = 1.0 - torch.sigmoid(gate_beta * (bins - valley))
        fog_height = (raw_pooled * fog_mask).amax(dim=1, keepdim=True)
        contrast = torch.log1p(hump_peak) - torch.log1p(fog_height)
        intensity = torch.sigmoid(
            self.intensity_scale * contrast + self.intensity_bias
        ).clamp(0.0, 1.0)

        # 8. 置信/选择性: hump 计数支撑 refine 门控; 谷后 gate 质量占比诊断选择性。
        support = (weight_sum / (weight_sum + 1.0)).clamp(0.0, 1.0)
        post_mass = masked.sum(dim=1, keepdim=True)
        total_mass = gated_pooled.sum(dim=1, keepdim=True) + self.eps
        selectivity = (post_mass / total_mass).clamp(0.0, 1.0)
        confidence = (hump_peak / (hump_peak + 1.0)).clamp(0.0, 1.0)

        # 诊断统计 (detached 标量): 供训练日志观察谷位是否塌缩/漂移。
        stats = {
            "fog_peak_mean": fog_peak.detach().mean(),
            "offset_mean": offset.detach().mean(),
            "valley_mean": valley.detach().mean(),
            "gate_beta": gate_beta.detach(),
            "hump_peak_mean": hump_peak.detach().mean(),
            "fog_height_mean": fog_height.detach().mean(),
        }

        return {
            "depth": depth,
            "intensity": intensity,
            "confidence": confidence,
            "support": support,
            "selectivity": selectivity,
            "stats": stats,
        }


class SNNFlowNet(SPADSpikeNet):
    """跨帧流式 SNN: 状态持久化到 step 之间, 支持随时读出当前估计。

    Args (除父类参数外):
        state_detach_interval: 流式累积量 (weighted_sum/weight_sum/gate_hist) 每隔
            多少个 chunk 做一次 detach, 用于截断时间反传 (TBPTT)。0 表示不截断累积量
            (与父类 forward 行为等价, 仅 detach 膜电位)。长序列单帧流式训练时设为正值
            可控制显存与梯度链长度。
        intra_period_mode: 预留开关。False (默认) 走跨帧流式; True 时调用
            ``_encode_frames`` 的周期内编码分支 (当前未实现, 仅占位抛错)。
        use_stream_context: 是否把 raw 直方图的 running 统计 (雾峰位置/光子密度/
            谷后占比/峰值集中度) 作为 4 个额外输入通道逐 chunk 注入 stem (死结 1
            修复之一)。上下文只来自**当前 chunk 之前**已累积的页, 严格因果;
            且从无梯度的 raw 计数计算, 不延长 BPTT 链。
        use_valley_hump: 是否用谷后 hump 物理检测头替代 gated-moment 输出口径。
        valley_spatial_pool: 谷后 hump 头的空间聚合核大小。
        valley_offset_min/valley_offset_max: 谷偏移的可行域 (bin)。
        valley_offset_init: 谷偏移初值 (bin), 物理初始化。
        valley_gate_beta_init: 软谷门陡度初值 (可学习)。
        valley_hump_window: 谷后搜峰窗口长度 (bin)。
    """

    # 流式上下文通道: [雾峰位置/t_max, 每页光子密度, 谷后光子占比, 峰值集中度]
    NUM_STREAM_CONTEXT_CHANNELS = 4

    def __init__(
        self,
        *args,
        state_detach_interval: int = 0,
        intra_period_mode: bool = False,
        use_stream_context: bool = True,
        use_valley_hump: bool = True,
        valley_spatial_pool: int = 5,
        valley_offset_min: float = 3.0,
        valley_offset_max: float = 40.0,
        valley_offset_init: float = 11.0,
        valley_gate_beta_init: float = 2.0,
        valley_hump_window: float = 48.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.state_detach_interval = int(state_detach_interval)
        if self.state_detach_interval < 0:
            raise ValueError("state_detach_interval must be non-negative")
        self.intra_period_mode = bool(intra_period_mode)
        self.use_stream_context = bool(use_stream_context)
        # 谷后 hump 物理检测头: 在累积后的直方图上做 "雾峰→软谷→谷后质心",
        # 解决逐光子 gate 看不到直方图形状、只能学全开 → depth 锁死雾峰的结构死锁。
        self.use_valley_hump = bool(use_valley_hump)
        # 流式状态; None 表示尚未 stream_reset。不进 state_dict (随 batch 形状变化)。
        self._stream_state: Optional[dict] = None

        # 参数重分配: 把容量从 stem 的 dense 3x3 (占父类 57% 参数) 移到
        # backbone 和 gate_head。stem 改用 DW+PW 分解保留感受野与通道混合;
        # gate_head 加宽到 C 并加一个 depthwise 3x3, 让 gate 能看空间邻域 ——
        # 目标回波是空间相干的, 这直接作用在 gate 选择性瓶颈上。
        spike_kwargs = dict(
            spike_mode=self.spike_mode,
            spike_tau=self.spike_tau,
            spike_v_threshold=self.spike_v_threshold,
            spike_v_reset=self.spike_v_reset,
            spike_backend=self.spike_backend,
            spike_tau_max=self.spike_tau_max,
        )
        c_enc = (
            self.tof_embedding.embed_dim
            if self.tof_embedding is not None
            else 2 * self.n_freq + 1
        )
        if self.use_stream_context:
            c_enc += self.NUM_STREAM_CONTEXT_CHANNELS
        self.stem = _SlimStem(c_enc, self.C, **spike_kwargs)
        self.gate_head = _FatGateHead(self.C, **spike_kwargs)

        # 把 (含上面新模块的) 所有 BatchNorm2d 换成 GroupNorm, 消除 chunk 内跨帧统计泄露。
        self.num_gn_replaced = _replace_bn_with_gn(self)

        # 谷后 hump 头在 GN 替换之后构造 (它内部无 BN, 不需再换)。
        if self.use_valley_hump:
            self.valley_hump = _ValleyHumpHead(
                t_max=self.t_max,
                spatial_pool=int(valley_spatial_pool),
                gate_beta_init=float(valley_gate_beta_init),
                valley_offset_init=float(valley_offset_init),
                valley_offset_min=float(valley_offset_min),
                valley_offset_max=float(valley_offset_max),
                hump_window=float(valley_hump_window),
            )
        else:
            self.valley_hump = None

    # ── 周期内流式扩展点 (预留, 未实现) ──────────────────────────
    def _encode_frames(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对一批页做编码, 返回 (x, tof, valid)。

        跨帧流式直接复用父类 ``_encode_chunk`` (ToF 作为编码通道值)。
        ``intra_period_mode=True`` 是为周期内流式预留的分支: 未来在此把每个像素的
        ToF 值散射成激光周期时间轴上的脉冲张量, 让膜电位沿 ToF 轴积分做匹配滤波。
        当前未实现, 显式抛错以免静默走错路径。
        """
        if not self.intra_period_mode:
            return self._encode_chunk(frames)
        raise NotImplementedError(
            "intra_period_mode (周期内流式) 尚未实现; 这是为 ToF 轴脉冲编码预留的扩展点。"
        )

    # ── 流式状态管理 ────────────────────────────────────────────
    def stream_reset(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | str,
    ) -> None:
        """清零累积量并 reset 膜电位, 开始一段新的流式序列。"""
        device = torch.device(device)
        functional.reset_net(self)
        _reset_module_spike_stats(self)
        self._stream_state = {
            "weighted_sum": torch.zeros(batch_size, 1, height, width, device=device),
            "weight_sum": torch.zeros(batch_size, 1, height, width, device=device),
            "gate_hist": torch.zeros(
                self.t_max + 1, batch_size * height * width, device=device
            ),
            # raw 光子计数直方图 (不乘 gate): 雾结构定位 + 流式上下文用, 无梯度。
            "raw_hist": torch.zeros(
                self.t_max + 1, batch_size * height * width, device=device
            ),
            "num_pages": 0,
            "chunk_count": 0,
            "B": int(batch_size),
            "H": int(height),
            "W": int(width),
        }

    def _require_state(self) -> dict:
        if self._stream_state is None:
            raise RuntimeError("call stream_reset(...) before stream_step/stream_readout")
        return self._stream_state

    @torch.no_grad()
    def _compute_stream_context(self) -> torch.Tensor:
        """从**已累积**的 raw 直方图算 4 通道流式上下文 (死结 1 修复之一)。

        逐光子 gate 在决策时刻看不到直方图形状 —— 能区分雾/目标的唯一统计量。
        这里把过去页的 raw 统计压成 4 个逐像素通道, 与 ToF 编码一起进 stem,
        让 gate 决策第一次"看得见"本样本的雾结构:

            ch0 雾峰位置估计 / t_max     (ToF-shift 增强下的关键不变量锚点)
            ch1 每页光子密度            (雾浓度 proxy)
            ch2 雾峰后光子占比          (谷后证据存在性)
            ch3 峰值集中度 (峰高/总量)   (直方图尖锐程度)

        仅使用当前 chunk **之前**的页 → 严格因果; raw 计数不含参数 → 无梯度,
        不延长 BPTT 链。第一个 chunk (num_pages=0) 返回全零。

        Returns:
            [B, 4, H, W] 上下文张量 (与 weight_sum 同 device/dtype)。
        """
        state = self._require_state()
        B, H, W = state["B"], state["H"], state["W"]
        ref = state["weight_sum"]
        if state["num_pages"] <= 0:
            return torch.zeros(
                B, self.NUM_STREAM_CONTEXT_CHANNELS, H, W,
                device=ref.device, dtype=ref.dtype,
            )

        t_max = self.t_max
        eps = 1e-6
        # [t_max, B*H*W] → [B, t_max, H, W]
        raw = (
            state["raw_hist"][1:]
            .reshape(t_max, B, H, W)
            .permute(1, 0, 2, 3)
            .to(ref.dtype)
        )
        pooled = F.avg_pool2d(raw, 5, stride=1, padding=2)
        total = pooled.sum(dim=1, keepdim=True)                  # [B, 1, H, W]
        # [t_max] → [1, t_max, 1, 1] 广播
        bins = torch.arange(
            1, t_max + 1, device=raw.device, dtype=raw.dtype
        ).view(1, t_max, 1, 1)

        peak_height = pooled.amax(dim=1, keepdim=True)
        pooled_norm = pooled / (peak_height + eps)
        w_fog = F.softmax(pooled_norm * 6.0, dim=1)
        fog_peak = (w_fog * bins).sum(dim=1, keepdim=True)       # [B, 1, H, W]

        # 雾峰后 8 bin 起算"晚到"光子 (谷/目标方向), 软划分保持平滑。
        late_mask = torch.sigmoid(bins - (fog_peak + 8.0))
        late_frac = (pooled * late_mask).sum(dim=1, keepdim=True) / (total + eps)

        density = total / float(max(1, state["num_pages"]))
        concentration = peak_height / (total + eps)

        # [B, 1, H, W] × 4 → [B, 4, H, W]
        return torch.cat(
            [fog_peak / float(t_max), density.clamp(0.0, 1.0),
             late_frac.clamp(0.0, 1.0), concentration.clamp(0.0, 1.0)],
            dim=1,
        )

    def stream_step(
        self,
        frames: torch.Tensor,
        *,
        collect_sequence: bool = False,
    ) -> dict:
        """喂入一批新到达的页, 累积状态并返回当前估计。

        Args:
            frames: [T, B, H, W] 新页 (T 可为 1 表示单帧)。B/H/W 必须与 stream_reset 一致。
            collect_sequence: 是否在返回中附带本 step 的 gate/tof/valid 序列。

        Returns:
            stream_readout() 的结果 dict; collect_sequence=True 时额外含
            ``gate``/``tof``/``valid`` (仅本 step)。
        """
        if frames.dim() != 4:
            raise ValueError("frames must have shape [T, B, H, W]")
        state = self._require_state()
        B, H, W = state["B"], state["H"], state["W"]
        if frames.shape[1:] != (B, H, W):
            raise ValueError(
                f"frames batch/spatial {tuple(frames.shape[1:])} "
                f"!= stream_reset ({B}, {H}, {W})"
            )

        x, tof, valid = self._encode_frames(frames)     # [T, B, C_enc/.., H, W]
        if self.use_stream_context:
            # 上下文取自当前 chunk 之前的累积 (严格因果), 对 chunk 内所有页相同。
            # [B, 4, H, W] → [T, B, 4, H, W] 后与编码特征在通道维拼接
            context = self._compute_stream_context()
            context = context.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
            x = torch.cat([x, context.to(x.dtype)], dim=2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        gate = self.gate_head(x)                         # [T, B, 1, H, W]

        T_actual = frames.shape[0]
        tof_exp = tof.unsqueeze(2)                       # [T, B, 1, H, W]
        v_exp = valid.unsqueeze(2)
        state["weighted_sum"] = state["weighted_sum"] + (gate * tof_exp * v_exp).sum(0)
        state["weight_sum"] = state["weight_sum"] + (gate * v_exp).sum(0)
        state["gate_hist"] = _accumulate_gate_hist(
            state["gate_hist"], gate, tof, valid, self.t_max
        )
        # raw 直方图: gate 恒 1 的 valid 计数, detach 保证不挂梯度。
        with torch.no_grad():
            state["raw_hist"] = _accumulate_gate_hist(
                state["raw_hist"], torch.ones_like(gate), tof, valid, self.t_max
            )
        state["num_pages"] += int(T_actual)
        state["chunk_count"] += 1

        # TBPTT: 周期性 detach 膜电位与累积量, 截断跨 step 的梯度链。
        interval = self.state_detach_interval
        if interval > 0 and state["chunk_count"] % interval == 0:
            functional.detach_net(self)
            state["weighted_sum"] = state["weighted_sum"].detach()
            state["weight_sum"] = state["weight_sum"].detach()
            state["gate_hist"] = state["gate_hist"].detach()

        out = self.stream_readout()
        if collect_sequence:
            out["gate"] = gate
            out["tof"] = tof
            out["valid"] = valid
        return out

    def stream_readout(self) -> dict:
        """用当前累积量算一次深度/强度估计, 不改变状态。"""
        state = self._require_state()
        B, H, W = state["B"], state["H"], state["W"]
        num_pages = max(1, int(state["num_pages"]))

        valley_stats = None
        if self.valley_hump is not None:
            # 谷后 hump 头: 雾峰/谷位从 raw 直方图定位 (与 gate 解耦),
            # hump 质心/峰高从 gate 直方图读出 (gate 学到选择性后信噪比更高)。
            maps = self.valley_hump(
                gate_hist=state["gate_hist"],
                raw_hist=state["raw_hist"],
                weight_sum=state["weight_sum"],
            )
            valley_stats = maps.pop("stats", None)
        else:
            maps = _finalize_gated_peak_maps(
                weighted_sum=state["weighted_sum"],
                weight_sum=state["weight_sum"],
                gate_hist=state["gate_hist"],
                num_pages=num_pages,
                depth_peak_half_width=self.depth_peak_half_width,
                softargmax_sharpness=self.depth_softargmax_sharpness,
            )
        depth = maps["depth"]
        intensity = maps["intensity"]
        confidence_map = maps["confidence"]
        coarse = torch.cat([depth, intensity], dim=1)
        output = self.refine(coarse, confidence_map)

        out = {
            "output": output,
            "depth": output[:, 0:1],
            "intensity": output[:, 1:2],
            "depth_coarse": depth,
            "intensity_coarse": intensity,
            "confidence": confidence_map,
            "support": maps["support"],
            "selectivity": maps["selectivity"],
            "num_pages": num_pages,
        }
        if valley_stats is not None:
            out["valley_stats"] = valley_stats
        out["gate_hist"] = (
            state["gate_hist"].detach()
            .reshape(self.t_max + 1, B, H, W)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        return out

    # ── 训练/批量接口 (与父类签名一致, 内部走流式循环) ───────────
    def forward(
        self,
        raw_data: torch.Tensor,
        *,
        return_sequence: bool | None = None,
    ) -> dict:
        """批量接口: 内部按 chunk_size 流式循环, 返回与父类一致的 dict。

        训练时复用本路径, 因此 loss 无需区分 backend。每个 chunk 之后按
        ``state_detach_interval`` 截断梯度 (0 表示仅 detach 膜电位, 等价父类)。
        """
        should_return_sequence = (
            self.return_sequence if return_sequence is None else bool(return_sequence)
        )
        B, N, P = raw_data.shape
        H = W = int(N ** 0.5)
        device = raw_data.device
        T_chunk = self.chunk_size

        # [B, N, P] → [P, B, H, W]
        data = raw_data.view(B, H, W, P).permute(3, 0, 1, 2).contiguous()

        self.stream_reset(B, H, W, device)
        all_gates, all_tofs, all_valids = [], [], []

        n_chunks = math.ceil(P / T_chunk)
        for i in range(n_chunks):
            t0 = i * T_chunk
            t1 = min(t0 + T_chunk, P)
            chunk = data[t0:t1]                          # [T_actual, B, H, W]

            step_out = self.stream_step(
                chunk, collect_sequence=should_return_sequence
            )
            if should_return_sequence:
                all_gates.append(step_out["gate"])
                all_tofs.append(step_out["tof"])
                all_valids.append(step_out["valid"])

            # state_detach_interval=0 时, stream_step 不会 detach 膜电位; 这里补一次
            # 与父类等价的 chunk 间膜电位 detach (但保留累积量加法链, 让强度梯度可回传)。
            if self.state_detach_interval == 0 and i < n_chunks - 1:
                functional.detach_net(self)

        out = self.stream_readout()
        out.pop("num_pages", None)

        spike_stats = _collect_module_spike_stats(self)
        functional.reset_net(self)
        self._stream_state = None

        if should_return_sequence:
            out["gate"] = torch.cat(all_gates, dim=0)    # [P, B, 1, H, W]
            out["tof"] = torch.cat(all_tofs, dim=0)      # [P, B, H, W]
            out["valid"] = torch.cat(all_valids, dim=0)  # [P, B, H, W]

        if self.tof_embedding is not None:
            out["lut_smooth"] = self.tof_embedding.smoothness_loss()
            out["lut_norm"] = self.tof_embedding.norm_loss()
        if spike_stats:
            out["spike_stats"] = spike_stats

        return out
