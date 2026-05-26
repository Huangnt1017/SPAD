"""正弦位置编码可视化工具.

功能:
1. 单帧 raw tof 热力图
2. 单帧编码后 17 通道 (valid + 8对 sin/cos) 逐通道可视化
3. 多帧累积聚合效果: 随帧数递增观察 depth / intensity 如何收敛
4. 频率响应分析: 不同 tof 值经过编码后的频谱表征

用法:
    python visualize_encoding.py --data_path <txt文件路径> [--n_freq 8] [--t_max 150]
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from SNN import encode_tof


# ─── 数据加载 ─────────────────────────────────────────────────

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

def plot_single_frame_raw(frame: torch.Tensor, frame_idx: int, save_dir: str,
                          target_bin: int = 60):
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
    path = os.path.join(save_dir, f"01_raw_frame_{frame_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_encoded_channels(frame: torch.Tensor, frame_idx: int, n_freq: int,
                          t_max: int, save_dir: str):
    """可视化单帧编码后的全部 2*n_freq+1 通道.

    Args:
        frame: [H, W] raw tof
        n_freq: 频率对数
        t_max: 最大 bin
        save_dir: 保存目录
    """
    valid = ((frame >= 1) & (frame <= t_max)).float()      # [H, W]
    tof = frame.float() * valid

    # encode_tof 期望 [*, H, W], 输出 [*, 17, H, W]
    encoded = encode_tof(tof, valid, n_freq=n_freq, t_max=t_max)  # [17, H, W]
    n_channels = encoded.shape[0]

    # 通道名称
    names = ["valid"]
    for i in range(n_freq):
        names.append(f"sin({i+1}π·t)")
        names.append(f"cos({i+1}π·t)")

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

    fig.suptitle(f"正弦编码 {n_channels} 通道 (帧 #{frame_idx}, n_freq={n_freq})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"02_encoded_channels_frame_{frame_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_frequency_response(n_freq: int, t_max: int, save_dir: str):
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
    for i in range(n_freq):
        freq = (i + 1) * math.pi
        y = torch.sin(freq * t_norm).numpy()
        axes[0].plot(tof_vals.numpy(), y, label=f"sin({i+1}π·t)", alpha=0.8)
    axes[0].set_ylabel("输出值")
    axes[0].set_title("sin 通道频率响应")
    axes[0].legend(loc="upper right", fontsize=7, ncol=4)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.1, 1.1)

    # cos 通道
    for i in range(n_freq):
        freq = (i + 1) * math.pi
        y = torch.cos(freq * t_norm).numpy()
        axes[1].plot(tof_vals.numpy(), y, label=f"cos({i+1}π·t)", alpha=0.8)
    axes[1].set_xlabel("ToF bin")
    axes[1].set_ylabel("输出值")
    axes[1].set_title("cos 通道频率响应")
    axes[1].legend(loc="upper right", fontsize=7, ncol=4)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.1, 1.1)

    plt.tight_layout()
    path = os.path.join(save_dir, "03_frequency_response.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_multi_frame_aggregation(frames: torch.Tensor, t_max: int, save_dir: str,
                                 target_bin: int = 60, fog_range: tuple = (30, 50)):
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
    path2 = os.path.join(save_dir, "04b_fog_vs_target_ratio.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  保存: {path2}")


def plot_encoding_discrimination(n_freq: int, t_max: int, save_dir: str):
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
    for i in range(n_freq):
        freq = (i + 1) * math.pi
        vectors.append(torch.sin(freq * t_norm))
        vectors.append(torch.cos(freq * t_norm))
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
    path = os.path.join(save_dir, "05_encoding_discrimination.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_fog_vs_target_encoding(n_freq: int, t_max: int, target_bin: int,
                                save_dir: str, fog_bin: int = 40):
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
    for i in range(n_freq):
        freq = (i + 1) * math.pi
        names.append(f"sin({i+1}π)")
        names.append(f"cos({i+1}π)")
        fog_vec.extend([math.sin(freq * t_fog), math.cos(freq * t_fog)])
        target_vec.extend([math.sin(freq * t_target), math.cos(freq * t_target)])

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
    path = os.path.join(save_dir, "06_fog_vs_target_encoding.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


# ─── 主程序 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPAD 正弦编码可视化")
    parser.add_argument("--data_path", type=str,
                        default=r"D:\PYproject\SPADdata\0825\frame\2025-08-25_16-50-14_Delay-0_Width-200.raw.txt",
                        help="raw.txt 数据文件路径")
    parser.add_argument("--n_freq", type=int, default=8, help="频率对数 (默认 8)")
    parser.add_argument("--t_max", type=int, default=150, help="最大有效 bin (默认 150)")
    parser.add_argument("--target_bin", type=int, default=60,
                        help="目标真实距离对应的 tof bin (默认 60)")
    parser.add_argument("--frame_idx", type=int, default=0, help="用于单帧可视化的帧号")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="图片保存目录 (默认: data_path 同级 vis_encoding/)")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.data_path), "vis_encoding")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录: {args.save_dir}")
    print(f"参数: n_freq={args.n_freq}, t_max={args.t_max}, target_bin={args.target_bin}\n")

    # 加载数据
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
    plot_encoded_channels(frame, frame_idx, args.n_freq, args.t_max, args.save_dir)

    # 3. 频率响应曲线
    print("[3/6] 频率响应分析...")
    plot_frequency_response(args.n_freq, args.t_max, args.save_dir)

    # 4. 多帧聚合 + 雾/目标光子占比趋势
    print("[4/6] 多帧聚合效果 + 雾/目标光子统计...")
    plot_multi_frame_aggregation(frames, args.t_max, args.save_dir,
                                 target_bin=args.target_bin)

    # 5. 编码区分度分析
    print("[5/6] 编码区分度分析...")
    plot_encoding_discrimination(args.n_freq, args.t_max, args.save_dir)

    # 6. 雾 bin=40 vs 目标 bin=60 的编码向量对比
    print("[6/6] 雾/目标编码向量对比...")
    plot_fog_vs_target_encoding(args.n_freq, args.t_max, args.target_bin, args.save_dir)

    print(f"\n完成! 共生成 7 张图, 保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
