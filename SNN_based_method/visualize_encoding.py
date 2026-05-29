"""ToF 编码可视化工具。

功能:
1. 可视化单帧 raw ToF 热力图。
2. 逐通道查看单帧编码结果，例如 valid + 多组 sin/cos 特征。
3. 分析多帧累积聚合时 depth / intensity 的收敛过程。
4. 对比不同 ToF bin 经过编码后的频率响应和区分度。

用法示例:
    python SNN_based_method/visualize_encoding.py --data_path <txt文件路径> --n_freq 8 --t_max 150
    python SNN_based_method/visualize_encoding.py --raw_path <raw文件路径> --pages_per_group 500 --time_threshold 150 --plot_group_num 1
"""

import argparse
import math
import os
import sys
import re
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False



FREQ_PRESETS = {
    "A": [1, 2, 3, 4, 5, 6, 7, 8],
    "B": [1, 2, 4, 6, 8, 12, 16, 24],
    "C": [1, 2, 3, 4, 6, 8, 12, 16],
    "D": [1, 2, 3, 4, 5, 6, 8, 12],
    "E": [1, 2, 3, 5, 8, 12, 16, 24],
}


# ─── 数据加载 ───────

def load_raw_data(data_path: str) -> np.ndarray:
    """从 raw.txt 文件加载 SPAD 数据.

    Args:
        data_path: txt 文件路径, 头部 # 开头为注释, 每行是一个像素 4096 values × P columns

    Returns:
        data: [4096, P] int array, 0 表示无效
    """
    rows = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = list(map(int, line.split()))
            rows.append(vals)
    data = np.array(rows, dtype=np.int32)
    print(f"加载数据: shape={data.shape} (像素数={data.shape[0]}, 帧数P={data.shape[1]})")
    return data


def load_raw_group_from_binary(
    raw_path: str,
    pages_per_group: int,
    time_threshold: int,
    plot_group_num: int,
    total_pages: Optional[int] = None,
) -> np.ndarray:
    """直接读取 raw 文件中的某一组, 返回 [4096, P] 数据.

    Args:
        raw_path: 原始 raw 文件路径
        pages_per_group: 每组页数
        time_threshold: 有效 ToF 上限 (超过置 0)
        plot_group_num: 要读取的组号 (从 1 开始)
        total_pages: 可选, 限制读取的总页数
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw 文件不存在: {raw_path}")
    if pages_per_group <= 0:
        raise ValueError("pages_per_group must be a positive integer")
    if time_threshold <= 0:
        raise ValueError("time_threshold must be a positive integer")
    if plot_group_num <= 0:
        raise ValueError("plot_group_num must be a positive integer")

    num_pixels = 64 * 64
    file_size = os.path.getsize(raw_path)
    total_values = file_size // 2
    max_pages = total_values // num_pixels
    if max_pages == 0:
        return np.empty((0, 4096), dtype=np.int32)

    if total_pages is None:
        total_pages = max_pages - (max_pages % pages_per_group)

    if total_pages <= 0:
        raise ValueError("No complete groups are available in this raw file")
    if total_pages > max_pages:
        raise ValueError(f"total_pages {total_pages} exceeds file pages {max_pages}")
    if total_pages % pages_per_group != 0:
        raise ValueError("total_pages must be divisible by pages_per_group")

    num_groups = total_pages // pages_per_group
    if plot_group_num > num_groups:
        raise IndexError(f"plot_group_num {plot_group_num} exceeds available groups {num_groups}")

    values_per_group = pages_per_group * num_pixels
    byte_offset = (plot_group_num - 1) * values_per_group * 2
    data = np.fromfile(raw_path, dtype=np.uint16, count=values_per_group, offset=byte_offset)
    if data.size != values_per_group:
        raise IOError(
            f"File too short: expected {values_per_group} values, got {data.size}"
        )

    frames = data.reshape((pages_per_group, 64, 64))
    flat_frames = frames.reshape((pages_per_group, num_pixels))
    flat_frames[flat_frames > time_threshold] = 0
    grouped = flat_frames.T.astype(np.int32, copy=False)
    print(
        f"加载 raw 组: group={plot_group_num}/{num_groups}, "
        f"shape={grouped.shape} (P={grouped.shape[1]})"
    )
    return grouped


def parse_freqs_arg(freqs_arg: Optional[str], n_freq: int) -> List[int]:
    """Parse frequency list or preset name into a list of ints."""
    if not freqs_arg:
        return list(range(1, n_freq + 1))

    key = freqs_arg.strip().upper()
    if key in FREQ_PRESETS:
        return list(FREQ_PRESETS[key])

    tokens = [t for t in re.split(r"[\s,]+", freqs_arg.strip()) if t]
    freqs = [int(t) for t in tokens]
    if not freqs:
        raise ValueError("freqs_arg is empty")
    return freqs


def encode_tof_with_freqs(
    tof: torch.Tensor,
    valid: torch.Tensor,
    freqs: List[int],
    t_max: int,
) -> torch.Tensor:
    """Encode a single frame using an explicit frequency list."""
    v = valid.float()
    t = tof.float() / float(t_max)
    channels = [v]
    for freq in freqs:
        phase = math.pi * float(freq) * t
        channels.append(torch.sin(phase) * v)
        channels.append(torch.cos(phase) * v)
    return torch.stack(channels, dim=0)


def data_to_frames(data: np.ndarray, h: int = 64, w: int = 64) -> torch.Tensor:
    """将 [4096, P] 数据转为 [P, H, W] 帧序列.

    Args:
        data: [4096, P] raw timestamps
        h, w: 空间分辨率

    Returns:
        frames: [P, H, W] tensor
    """
    assert data.shape[0] == h * w, f"像素数 {data.shape[0]} != {h}*{w}"
    # [4096, P] → [H, W, P] → [P, H, W]
    frames = torch.from_numpy(data).float().reshape(h, w, -1).permute(2, 0, 1)
    return frames


# ─── 可视化函数 ─────────────────────────────────────────────────

def plot_single_frame_raw(
    frame: torch.Tensor,
    frame_idx: int,
    save_dir: Optional[str],
    target_bin: int = 60,
):
    """可视化单帧原始 tof 热力图.

    Args:
        frame: [H, W] raw tof values
        frame_idx: 帧号
        save_dir: 保存目录
        target_bin: 目标真实距离对应的 tof bin, 用于在分布图上标注
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 原始 tof (含 0)
    im0 = axes[0].imshow(frame.numpy(), cmap="turbo", vmin=0, vmax=150)
    axes[0].set_title(f"原始 ToF (帧 #{frame_idx})")
    plt.colorbar(im0, ax=axes[0], label="bin")

    # 有效性 mask
    valid = ((frame >= 1) & (frame <= 150)).float()
    axes[1].imshow(valid.numpy(), cmap="gray", vmin=0, vmax=1)
    valid_ratio = valid.mean().item() * 100
    axes[1].set_title(f"有效像素 mask ({valid_ratio:.1f}%)")

    # 仅有效像素的 tof 分布, 标注雾区和目标区
    valid_tof = frame[valid.bool()].numpy()
    if len(valid_tof) > 0:
        axes[2].hist(valid_tof, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[2].axvline(np.median(valid_tof), color="red", linestyle="--",
                        label=f"中位数={np.median(valid_tof):.1f} (雾散射主峰)")
        axes[2].axvline(target_bin, color="lime", linestyle="-", linewidth=2,
                        label=f"目标回波 ≈bin {target_bin}")
        # 雾区 / 目标区背景色
        axes[2].axvspan(30, 50, alpha=0.12, color="red", label="雾后向散射区")
        axes[2].axvspan(target_bin - 5, target_bin + 5, alpha=0.12, color="lime",
                        label="目标信号区")
        axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel("ToF bin")
    axes[2].set_ylabel("像素数")
    axes[2].set_title("有效 ToF 分布 (雾 vs 目标)")

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"01_raw_frame_{frame_idx}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


def plot_encoded_channels(
    frame: torch.Tensor,
    frame_idx: int,
    freqs: List[int],
    t_max: int,
    save_dir: Optional[str],
):
    """可视化单帧编码后的全部 2*n_freq+1 通道.

    Args:
        frame: [H, W] raw tof
        freqs: 频率列表
        t_max: 最大 bin
        save_dir: 保存目录
    """
    valid = ((frame >= 1) & (frame <= t_max)).float()      # [H, W]
    tof = frame.float() * valid

    encoded = encode_tof_with_freqs(tof, valid, freqs=freqs, t_max=t_max)  # [C, H, W]
    n_channels = encoded.shape[0]

    # 通道名称
    names = ["valid"]
    for freq in freqs:
        names.append(f"sin({freq}π·t)")
        names.append(f"cos({freq}π·t)")

    # 布局: 3行6列 = 18格, 前17有内容
    n_cols = 6
    n_rows = math.ceil(n_channels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    axes = axes.flatten()

    for ch_idx in range(n_channels):
        ch_data = encoded[ch_idx].numpy()
        if ch_idx == 0:
            # valid 通道用灰度
            axes[ch_idx].imshow(ch_data, cmap="gray", vmin=0, vmax=1)
        else:
            # sin/cos 通道用发散色阶
            axes[ch_idx].imshow(ch_data, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[ch_idx].set_title(names[ch_idx], fontsize=9)
        axes[ch_idx].axis("off")

    # 隐藏多余子图
    for idx in range(n_channels, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"正弦编码 {n_channels} 通道 (帧 #{frame_idx}, freqs={freqs})",
        fontsize=13,
    )
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"02_encoded_channels_frame_{frame_idx}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


def plot_frequency_response(freqs: List[int], t_max: int, save_dir: Optional[str]):
    """展示不同 tof 值经正弦编码后的频谱响应曲线.

    横轴: tof bin [1, t_max], 纵轴: 各频率通道输出值.
    帮助理解编码对不同距离的区分能力.

    Args:
        n_freq: 频率对数
        t_max: 最大 bin
        save_dir: 保存目录
    """
    tof_vals = torch.arange(1, t_max + 1).float()
    t_norm = tof_vals / t_max                                   # [t_max]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # sin 通道
    for freq in freqs:
        y = torch.sin(math.pi * float(freq) * t_norm).numpy()
        axes[0].plot(tof_vals.numpy(), y, label=f"sin({freq}π·t)", alpha=0.8)
    axes[0].set_ylabel("输出值")
    axes[0].set_title("sin 通道频率响应")
    axes[0].legend(loc="upper right", fontsize=7, ncol=4)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.1, 1.1)

    # cos 通道
    for freq in freqs:
        y = torch.cos(math.pi * float(freq) * t_norm).numpy()
        axes[1].plot(tof_vals.numpy(), y, label=f"cos({freq}π·t)", alpha=0.8)
    axes[1].set_xlabel("ToF bin")
    axes[1].set_ylabel("输出值")
    axes[1].set_title("cos 通道频率响应")
    axes[1].legend(loc="upper right", fontsize=7, ncol=4)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.1, 1.1)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "03_frequency_response.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


def plot_multi_frame_aggregation(
    frames: torch.Tensor,
    t_max: int,
    save_dir: Optional[str],
    target_bin: int = 60,
    fog_range: tuple = (30, 50),
):
    """可视化多帧聚合效果: 随帧数递增, depth 和 intensity 如何收敛.

    模拟 Gated Moment 中 gate=valid 的简化版 (无模型权重, 仅用有效性加权),
    观察累积统计量的演变趋势.
    同时统计雾区 / 目标区光子占比随帧数变化, 展示信噪比演变.

    Args:
        frames: [P, H, W] 全部帧
        t_max: 最大有效 bin
        save_dir: 保存目录
        target_bin: 目标真实距离 bin
        fog_range: 雾后向散射的 bin 区间 (lo, hi)
    """
    P, H, W = frames.shape
    fog_lo, fog_hi = fog_range
    target_lo, target_hi = target_bin - 5, target_bin + 5

    # 选取若干快照帧数
    checkpoints = [1, 5, 10, 50, 100, min(200, P), min(500, P), P]
    checkpoints = sorted(set(cp for cp in checkpoints if cp <= P))

    # 累积计算 (gate=1 for valid, depth = weighted mean of tof)
    weighted_sum = torch.zeros(H, W)
    weight_sum = torch.zeros(H, W)
    snapshots_depth = []
    snapshots_intensity = []
    snapshot_labels = []

    # 逐帧统计雾 / 目标光子数 (每隔一定帧采样, 避免太密)
    sample_interval = max(1, P // 200)
    fog_counts_cum = []
    target_counts_cum = []
    total_counts_cum = []
    frame_axes = []
    fog_count_running = 0
    target_count_running = 0
    total_count_running = 0

    for p_idx in range(P):
        frame = frames[p_idx]                                   # [H, W]
        valid = ((frame >= 1) & (frame <= t_max)).float()
        tof = frame.float() * valid

        weighted_sum += tof * valid
        weight_sum += valid

        # 统计光子归属
        is_fog = ((frame >= fog_lo) & (frame <= fog_hi)).sum().item()
        is_target = ((frame >= target_lo) & (frame <= target_hi)).sum().item()
        is_valid = valid.sum().item()
        fog_count_running += is_fog
        target_count_running += is_target
        total_count_running += is_valid

        frame_count = p_idx + 1
        if frame_count in checkpoints:
            depth = weighted_sum / (weight_sum + 1e-6)
            intensity = weight_sum / frame_count
            snapshots_depth.append(depth.clone().numpy())
            snapshots_intensity.append(intensity.clone().numpy())
            snapshot_labels.append(f"P={frame_count}")

        if frame_count % sample_interval == 0 or frame_count == P:
            frame_axes.append(frame_count)
            fog_counts_cum.append(fog_count_running)
            target_counts_cum.append(target_count_running)
            total_counts_cum.append(total_count_running)

    n_snaps = len(snapshots_depth)

    # ─── 图1: Depth / Intensity 聚合快照 ──
    fig, axes = plt.subplots(2, n_snaps, figsize=(n_snaps * 3, 6.5))
    if n_snaps == 1:
        axes = axes.reshape(2, 1)

    for idx in range(n_snaps):
        d = snapshots_depth[idx]
        valid_mask = d > 0
        vmin_d = np.percentile(d[valid_mask], 2) if valid_mask.any() else 0
        vmax_d = np.percentile(d[valid_mask], 98) if valid_mask.any() else t_max

        im = axes[0, idx].imshow(d, cmap="turbo", vmin=vmin_d, vmax=vmax_d)
        axes[0, idx].set_title(snapshot_labels[idx], fontsize=10)
        axes[0, idx].axis("off")

        i_data = snapshots_intensity[idx]
        axes[1, idx].imshow(i_data, cmap="inferno", vmin=0,
                            vmax=max(i_data.max(), 0.01))
        axes[1, idx].axis("off")

    axes[0, 0].set_ylabel("Depth (被雾主导→偏低)", fontsize=9)
    axes[1, 0].set_ylabel("Intensity (高=雾, 非目标)", fontsize=9)

    fig.suptitle("多帧聚合 (无模型, 纯统计加权 → depth 被雾后向散射拉偏)", fontsize=12)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "04_multi_frame_aggregation.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")

    # ─── 图2: 雾/目标光子占比随帧数变化 ──
    frame_axes = np.array(frame_axes)
    fog_ratio = np.array(fog_counts_cum) / (np.array(total_counts_cum) + 1e-6) * 100
    target_ratio = np.array(target_counts_cum) / (np.array(total_counts_cum) + 1e-6) * 100

    fig2, ax_left = plt.subplots(figsize=(10, 5))

    ax_left.fill_between(frame_axes, fog_ratio, alpha=0.3, color="red", label=f"雾区 [{fog_lo}-{fog_hi}] bin")
    ax_left.plot(frame_axes, fog_ratio, color="red", linewidth=1.5)
    ax_left.fill_between(frame_axes, target_ratio, alpha=0.3, color="green", label=f"目标区 [{target_lo}-{target_hi}] bin")
    ax_left.plot(frame_axes, target_ratio, color="green", linewidth=1.5)
    ax_left.set_xlabel("累积帧数 P")
    ax_left.set_ylabel("光子占比 (%)")
    ax_left.set_title("雾光子 vs 目标光子占比随帧数累积趋势")
    ax_left.legend(loc="center right", fontsize=9)
    ax_left.grid(True, alpha=0.3)

    # 右轴: 雾/目标比 (信噪比的倒数)
    ax_right = ax_left.twinx()
    fog_target_ratio = np.array(fog_counts_cum) / (np.array(target_counts_cum) + 1e-6)
    ax_right.plot(frame_axes, fog_target_ratio, color="orange", linestyle="--",
                  linewidth=1.5, label="雾/目标光子比 (越高越难)")
    ax_right.set_ylabel("雾/目标光子比", color="orange")
    ax_right.tick_params(axis="y", labelcolor="orange")
    ax_right.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_dir:
        path2 = os.path.join(save_dir, "04b_fog_vs_target_ratio.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  保存: {path2}")


def plot_target_bin_intensity(
    frames: torch.Tensor,
    target_bin: int,
    save_dir: Optional[str],
):
    """按帧比例聚合指定深度 bin 的强度图 (其余置零)."""
    if target_bin <= 0:
        raise ValueError("target_bin must be a positive integer")

    total_frames = frames.shape[0]
    if total_frames == 0:
        return

    percent_points = [1, 5, 10, 20, 30, 50, 75, 100]
    frame_counts = [max(1, int(round(total_frames * p / 100.0))) for p in percent_points]
    frame_counts = sorted(set(min(total_frames, c) for c in frame_counts))

    n_cols = 4
    n_rows = math.ceil(len(frame_counts) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, frame_count in enumerate(frame_counts):
        row = idx // n_cols
        col = idx % n_cols
        subset = frames[:frame_count]
        intensity_map = (subset == float(target_bin)).sum(dim=0).float()
        ax = axes[row, col]
        im = ax.imshow(intensity_map.numpy(), cmap="inferno", vmin=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        percent = frame_count / total_frames * 100.0
        ax.set_title(f"{percent:.0f}% (P={frame_count})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(frame_counts), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(f"Target-bin Intensity (bin={target_bin})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        path = os.path.join(save_dir, f"07_target_bin_intensity_{target_bin}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


def plot_encoding_discrimination(freqs: List[int], t_max: int, save_dir: Optional[str]):
    """展示编码的距离区分能力: 不同 tof 对应的编码向量之间的余弦相似度.

    理想情况下相近 tof 相似度高, 差距大的 tof 相似度低 → 编码区分度好.

    Args:
        n_freq: 频率对数
        t_max: 最大 bin
        save_dir: 保存目录
    """
    # 对 [1, t_max] 中均匀采样的 tof 值编码
    tof_vals = torch.arange(1, t_max + 1).float()              # [t_max]
    t_norm = tof_vals / t_max

    # 构建编码矩阵 [t_max, 2*n_freq+1]
    vectors = [torch.ones(t_max)]                               # valid 通道恒为 1
    for freq in freqs:
        phase = math.pi * float(freq) * t_norm
        vectors.append(torch.sin(phase))
        vectors.append(torch.cos(phase))
    enc_matrix = torch.stack(vectors, dim=1)                    # [t_max, 17]

    # 归一化后计算余弦相似度矩阵
    enc_normed = enc_matrix / (enc_matrix.norm(dim=1, keepdim=True) + 1e-8)
    cosine_sim = (enc_normed @ enc_normed.T).numpy()            # [t_max, t_max]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 相似度矩阵热力图
    im = axes[0].imshow(cosine_sim, cmap="RdBu_r", vmin=-1, vmax=1,
                        extent=[1, t_max, t_max, 1])
    axes[0].set_xlabel("ToF bin")
    axes[0].set_ylabel("ToF bin")
    axes[0].set_title("编码向量余弦相似度矩阵")
    plt.colorbar(im, ax=axes[0])

    # 选几个参考 tof, 画其与其它 tof 的相似度曲线
    ref_tofs = [10, 30, 50, 75, 100, 130]
    for ref in ref_tofs:
        if ref <= t_max:
            axes[1].plot(tof_vals.numpy(), cosine_sim[ref - 1], label=f"ref={ref}",
                         alpha=0.8)
    axes[1].set_xlabel("ToF bin")
    axes[1].set_ylabel("余弦相似度")
    axes[1].set_title("参考点与其他 tof 的编码相似度")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.05, 1.05)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "05_encoding_discrimination.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


def plot_fog_vs_target_encoding(
    freqs: List[int],
    t_max: int,
    target_bin: int,
    save_dir: Optional[str],
    fog_bin: int = 40,
):
    """对比雾 (bin≈40) 和目标 (bin≈60) 的编码向量, 展示编码如何区分两者.

    左图: 两个 bin 的 17 维编码向量条形图对比
    右图: 编码向量差异 (目标 - 雾), 突出哪些通道贡献最大区分度

    Args:
        n_freq: 频率对数
        t_max: 最大 bin
        target_bin: 目标 tof bin
        save_dir: 保存目录
        fog_bin: 雾主峰 tof bin
    """
    # 计算两个 bin 的编码
    t_fog = fog_bin / t_max
    t_target = target_bin / t_max

    names = ["valid"]
    fog_vec = [1.0]
    target_vec = [1.0]
    for freq in freqs:
        names.append(f"sin({freq}π)")
        names.append(f"cos({freq}π)")
        phase = math.pi * float(freq)
        fog_vec.extend([math.sin(phase * t_fog), math.cos(phase * t_fog)])
        target_vec.extend([math.sin(phase * t_target), math.cos(phase * t_target)])

    fog_vec = np.array(fog_vec)
    target_vec = np.array(target_vec)
    diff = target_vec - fog_vec
    n_ch = len(names)
    x = np.arange(n_ch)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上图: 两个 bin 的编码向量对比
    bar_w = 0.35
    axes[0].bar(x - bar_w / 2, fog_vec, bar_w, color="red", alpha=0.7,
                label=f"雾 bin={fog_bin}")
    axes[0].bar(x + bar_w / 2, target_vec, bar_w, color="green", alpha=0.7,
                label=f"目标 bin={target_bin}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("编码值")
    axes[0].set_title(f"编码向量对比: 雾 (bin={fog_bin}) vs 目标 (bin={target_bin})")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].axhline(0, color="black", linewidth=0.5)

    # 下图: 编码差异 (目标 - 雾)
    colors = ["green" if d > 0 else "red" for d in diff]
    axes[1].bar(x, np.abs(diff), color=colors, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("|目标 - 雾| 编码差")
    axes[1].set_title("各通道区分贡献 (高 = 该频率对区分雾/目标贡献大)")
    axes[1].grid(True, alpha=0.3, axis="y")

    # 标注 cosine similarity
    cos_sim = np.dot(fog_vec, target_vec) / (np.linalg.norm(fog_vec) * np.linalg.norm(target_vec) + 1e-8)
    l2_dist = np.linalg.norm(diff)
    axes[1].text(0.98, 0.92,
                 f"余弦相似度 = {cos_sim:.4f}\nL2距离 = {l2_dist:.4f}",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "06_fog_vs_target_encoding.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


# ─── 主程序 ──────────────────────────────────────────────────
def demo_without_cli() -> None:
    """Run a no-CLI demo with explicit parameters and keep plots open."""
    raw_path = r"E:\essay\硕士\研一\SPAD数据\0825\2025-08-25_16-50-11_Delay-0_Width-200.raw"
    pages_per_group = 2000
    time_threshold = 150
    plot_group_num = 20
    total_pages = 48000

    freqs = FREQ_PRESETS["C"]
    t_max = 150
    target_bin = 60
    frame_idx = 0
    save_dir = None

    data = load_raw_group_from_binary(
        raw_path=raw_path,
        pages_per_group=pages_per_group,
        time_threshold=time_threshold,
        plot_group_num=plot_group_num,
        total_pages=total_pages,
    )
    frames = data_to_frames(data)
    frame_idx = min(frame_idx, frames.shape[0] - 1)
    frame = frames[frame_idx]

    # plot_single_frame_raw(frame, frame_idx, save_dir, target_bin=target_bin)
    # plot_encoded_channels(frame, frame_idx, freqs, t_max, save_dir)
    # plot_frequency_response(freqs, t_max, save_dir)
    # plot_multi_frame_aggregation(frames, t_max, save_dir, target_bin=target_bin)
    # plot_encoding_discrimination(freqs, t_max, save_dir)
    # plot_fog_vs_target_encoding(freqs, t_max, target_bin, save_dir)
    plot_target_bin_intensity(frames, target_bin, save_dir)

    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description="SPAD 正弦编码可视化")
    parser.add_argument("--data_path", type=str,
                        default=r"D:\PYproject\SPADdata\0825\frame\2025-08-25_16-50-14_Delay-0_Width-200.raw.txt",
                        help="raw.txt 数据文件路径")
    parser.add_argument("--raw_path", type=str, default=None,
                        help="raw 原始文件路径 (提供后将忽略 data_path)")
    parser.add_argument("--pages_per_group", type=int, default=500,
                        help="每组页数 (用于 raw 读取)")
    parser.add_argument("--time_threshold", type=int, default=150,
                        help="有效 ToF 上限 (用于 raw 读取)")
    parser.add_argument("--total_pages", type=int, default=None,
                        help="raw 读取的总页数 (默认自动对齐)")
    parser.add_argument("--plot_group_num", type=int, default=1,
                        help="绘制第几组 (从 1 开始, 仅对 raw 生效)")
    parser.add_argument("--n_freq", type=int, default=8, help="频率对数 (默认 8)")
    parser.add_argument("--freqs", type=str, default=None,
                        help="频率列表或预设(A/B/C/D/E), 例如 '1,2,4,6,8,12,16,24'")
    parser.add_argument("--t_max", type=int, default=150, help="最大有效 bin (默认 150)")
    parser.add_argument("--target_bin", type=int, default=60,
                        help="目标真实距离对应的 tof bin (默认 60)")
    parser.add_argument("--frame_idx", type=int, default=0, help="用于单帧可视化的帧号")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="图片保存目录 (默认: data_path 同级 vis_encoding/)")
    args = parser.parse_args()

    if args.save_dir is None:
        base_path = args.raw_path if args.raw_path else args.data_path
        args.save_dir = os.path.join(os.path.dirname(base_path), "vis_encoding")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录: {args.save_dir}")
    freqs = parse_freqs_arg(args.freqs, args.n_freq)
    print(f"参数: t_max={args.t_max}, target_bin={args.target_bin}, freqs={freqs}\n")

    # 加载数据
    if args.raw_path:
        data = load_raw_group_from_binary(
            raw_path=args.raw_path,
            pages_per_group=args.pages_per_group,
            time_threshold=args.time_threshold,
            plot_group_num=args.plot_group_num,
            total_pages=args.total_pages,
        )
    else:
        data = load_raw_data(args.data_path)
    frames = data_to_frames(data)                               # [P, H, W]
    P = frames.shape[0]

    frame_idx = min(args.frame_idx, P - 1)
    frame = frames[frame_idx]                                   # [H, W]

    # 1. 单帧原始 tof (标注雾区 / 目标区)
    print("[1/6] 单帧原始 tof 热力图...")
    plot_single_frame_raw(frame, frame_idx, args.save_dir, target_bin=args.target_bin)

    # 2. 编码后各通道
    print("[2/6] 编码后 17 通道可视化...")
    plot_encoded_channels(frame, frame_idx, freqs, args.t_max, args.save_dir)

    # 3. 频率响应曲线
    print("[3/6] 频率响应分析...")
    plot_frequency_response(freqs, args.t_max, args.save_dir)

    # 4. 多帧聚合 + 雾/目标光子占比趋势
    print("[4/6] 多帧聚合效果 + 雾/目标光子统计...")
    plot_multi_frame_aggregation(frames, args.t_max, args.save_dir,
                                 target_bin=args.target_bin)

    # 5. 编码区分度分析
    print("[5/6] 编码区分度分析...")
    plot_encoding_discrimination(freqs, args.t_max, args.save_dir)

    # 6. 雾 bin=40 vs 目标 bin=60 的编码向量对比
    print("[6/6] 雾/目标编码向量对比...")
    plot_fog_vs_target_encoding(freqs, args.t_max, args.target_bin, args.save_dir)

    # 7. 只聚合指定深度 bin 的强度图
    print("[7/7] 指定深度 bin 的聚合强度图...")
    plot_target_bin_intensity(frames, args.target_bin, args.save_dir)

    print(f"\n完成! 共生成 7 张图, 保存在: {args.save_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        demo_without_cli()
