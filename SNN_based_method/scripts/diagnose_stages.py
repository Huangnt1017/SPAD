"""逐环节诊断 flow 模型的中间输出, 定位 gate / 直方图 / 深度退化点。

不画图、不写产物, 只把单样本流水线每一环的关键统计打到 stdout, 用于回答:
gate 是否全开、gate 加权直方图峰值落在哪个 ToF bin、该 bin 是雾峰还是目标峰、
depth/intensity 是被雾主导还是抓到了目标。

运行 (仓库根目录):
    $env:PYTHONPATH = "D:\\PYproject\\SPAD"
    & D:\\Anaconda3\\envs\\torchnew\\python.exe SNN_based_method/scripts/diagnose_stages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.config.SNN_config import SNNConfig
from SNN_based_method.utils.data import seed_everything, spad_time_first_collate
from SNN_based_method.utils.runtime import load_checkpoint, prepare_model_input

# test1.py 把这些定义为模块级函数; 直接复用避免重复实现加载逻辑。
from SNN_based_method.scripts.test1 import build_single_sample_dataset


# ===== 可编辑参数 =====
CHECKPOINT = r"D:\PYproject\SPAD\checkpoints\SNN\train_20260611_011445\best.pth"
RAW_PATH = r"D:\PYproject\SPADdata\0917\2025-09-17_17-33-26_Delay-0_Width-2000.raw"
PAGES_PER_GROUP = 480
GROUP_INDEX = 71


def _describe(name: str, tensor: torch.Tensor) -> None:
    """打印张量的 min/mean/max/std, 统一格式。"""
    t = tensor.detach().float().reshape(-1)
    print(
        f"  {name:<22} shape={tuple(tensor.shape)!s:<18} "
        f"min={t.min():.4f} mean={t.mean():.4f} max={t.max():.4f} std={t.std():.4f}"
    )


def main() -> None:
    seed_everything(42)

    # ── 加载配置 (结构参数从 checkpoint 读, 保证权重对齐) ──
    cfg = SNNConfig()
    cfg = cfg.clone_with(
        checkpoint_path=CHECKPOINT,
        pages_per_group=PAGES_PER_GROUP,
        return_sequence=True,
    )

    device = cfg.resolved_device()
    print(f"device={device}  backend={cfg.model_backend}  C={cfg.C}  "
          f"num_blocks={cfg.num_blocks}  chunk_size={cfg.chunk_size}")

    # ── 构建单样本 ──
    dataset = build_single_sample_dataset(
        cfg, RAW_PATH, pages_per_group=PAGES_PER_GROUP, group_index=GROUP_INDEX
    )
    item = dataset[0]
    batch = spad_time_first_collate([item])
    frames = batch["frames"].to(device)             # [P, B, 1, 1, H, W] 或类似
    model_input = prepare_model_input(frames).to(device)  # [B, N, P]
    print(f"model_input shape = {tuple(model_input.shape)}")

    # ── 加载模型 ──
    model = cfg.build_model().to(device)
    load_checkpoint(CHECKPOINT, model, map_location=device)
    model.eval()

    with torch.no_grad():
        result = model(model_input, return_sequence=True)

    print("\n===== 中间环节统计 =====")
    gate = result["gate"]          # [P, B, 1, H, W]
    valid = result["valid"]        # [P, B, H, W]
    tof = result["tof"]            # [P, B, H, W]

    _describe("gate", gate)
    _describe("valid", valid)
    _describe("tof(masked)", tof)

    # gate 在有效位置的平均激活率 = 训练日志里的 sparse_rate
    gate_sq = gate.squeeze(2)
    mean_rate = (gate_sq * valid).sum() / (valid.sum() + 1e-6)
    print(f"\n  gate 有效位平均激活率 (=sparse_rate) = {mean_rate:.4f}  "
          f"(目标带 0.05~0.15)")

    # gate 的方差: 是否在不同光子间有区分
    gate_valid_vals = gate_sq[valid > 0]
    print(f"  gate 在有效光子上: mean={gate_valid_vals.mean():.4f} "
          f"std={gate_valid_vals.std():.4f} "
          f"<0.3占比={float((gate_valid_vals<0.3).float().mean()):.3f} "
          f">0.7占比={float((gate_valid_vals>0.7).float().mean()):.3f}")

    # ── coarse / refine 输出 ──
    print("\n===== 输出环节 =====")
    _describe("depth_coarse", result["depth_coarse"])
    _describe("intensity_coarse", result["intensity_coarse"])
    _describe("depth(refined)", result["depth"])
    _describe("intensity(refined)", result["intensity"])
    _describe("confidence", result["confidence"])
    _describe("selectivity", result["selectivity"])
    _describe("support", result["support"])

    # ── 关键诊断: gate 加权直方图峰 vs 原始光子直方图峰 ──
    print("\n===== 直方图对比 (核心诊断) =====")
    t_max = model.t_max
    H = W = int(model_input.shape[1] ** 0.5)
    B = model_input.shape[0]

    # 原始光子直方图 (不加 gate): 把 valid 光子按 ToF bin 计数
    tof_long = tof.long().clamp(0, t_max)               # [P, B, H, W]
    raw_hist = torch.zeros(t_max + 1, device=device)
    idx = tof_long.reshape(-1)
    w = valid.reshape(-1)
    raw_hist.scatter_add_(0, idx, w)
    raw_hist_valid = raw_hist[1:]                        # 去掉 bin0 (无效)

    # gate 加权直方图 (模型实际用来定深度的)
    gate_hist = result["gate_hist"]                     # [B, t_max+1, H, W]
    gate_hist_sum = gate_hist[0].reshape(t_max + 1, -1).sum(1)  # 全图聚合
    gate_hist_valid = gate_hist_sum[1:]

    raw_peak = int(raw_hist_valid.argmax().item()) + 1
    gate_peak = int(gate_hist_valid.argmax().item()) + 1
    print(f"  原始光子直方图峰 bin   = {raw_peak}  "
          f"(count={raw_hist_valid.max():.0f})")
    print(f"  gate加权直方图峰 bin   = {gate_peak}  "
          f"(count={gate_hist_valid.max():.1f})")
    print(f"  → 两个峰{'相同' if raw_peak == gate_peak else '不同'}; "
          f"若相同说明 gate 没有把直方图峰从雾移开")

    # 打印两个直方图前若干主峰, 看雾峰/目标峰分布
    def _top_bins(hist: torch.Tensor, k: int = 8) -> str:
        vals, idxs = hist.topk(min(k, hist.numel()))
        return ", ".join(
            f"bin{int(i)+1}:{float(v):.0f}" for v, i in zip(vals, idxs)
        )

    print(f"\n  原始直方图 top8: {_top_bins(raw_hist_valid)}")
    print(f"  gate直方图 top8: {_top_bins(gate_hist_valid)}")

    # ── 与 label 对比 (如果有) ──
    if "label" in batch and batch["label"] is not None:
        label = batch["label"]
        if isinstance(label, torch.Tensor):
            print("\n===== Label 对比 =====")
            label = label.float()
            lbl_depth = label[0, 0]
            lbl_int = label[0, 1]
            pos = lbl_int > 0
            if pos.any():
                print(f"  label depth 在目标区(intensity>0): "
                      f"min={lbl_depth[pos].min():.1f} "
                      f"mean={lbl_depth[pos].mean():.1f} "
                      f"max={lbl_depth[pos].max():.1f}  "
                      f"目标像素数={int(pos.sum())}/{pos.numel()}")
                print(f"  → 目标真实深度约在 bin {float(lbl_depth[pos].mean()):.0f} 附近; "
                      f"模型峰在 bin {gate_peak}")
            else:
                print("  label 中无正强度像素")

    print("\n诊断完成。")


if __name__ == "__main__":
    main()
