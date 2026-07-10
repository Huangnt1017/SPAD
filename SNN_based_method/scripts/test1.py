"""SPAD SNN 模型的单样本冒烟测试与推理入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.config.SNN_config import SNNConfig
from SNN_based_method.utils.data import (
    RawGroupSample,
    SpadRawGroupDataset,
    attach_precomputed_label_metadata,
    collect_raw_records,
    read_raw_group,
    seed_everything,
    spad_time_first_collate,
)
from SNN_based_method.utils.runtime import (
    add_config_arguments,
    config_from_checkpoint_and_args,
    load_checkpoint,
    make_log_run_dir,
    prepare_model_input,
)


def _to_image_2d(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """把 [1,H,W] / [H,W] 张量或数组转成 float32 图像。"""
    if isinstance(array, torch.Tensor):
        data = array.detach().cpu().numpy()
    else:
        data = np.asarray(array)
    data = np.squeeze(data)
    if data.shape != (64, 64):
        raise ValueError(f"image must have shape (64, 64), got {data.shape}")
    return data.astype(np.float32, copy=False)


def _to_prediction_npy(output: torch.Tensor) -> np.ndarray:
    """把单样本模型输出转成 ``[2, 64, 64]`` 的 float32 数组。"""
    data = output.detach().cpu().numpy()
    if data.shape != (1, 2, 64, 64):
        raise ValueError(f"output must have shape (1, 2, 64, 64), got {data.shape}")
    return data[0].astype(np.float32, copy=False)


def _valid_percentile_range(image: np.ndarray, fallback_max: float) -> tuple[float, float]:
    """为 depth 图计算稳定色阶, 只统计非零有效区域。"""
    valid = image > 0
    if not np.any(valid):
        return 0.0, float(fallback_max)
    vmin = float(np.percentile(image[valid], 2))
    vmax = float(np.percentile(image[valid], 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _positive_percentile_max(image: np.ndarray, fallback_max: float) -> float:
    """为强度图计算稳定上限, 只统计正值区域。"""
    positive = image > 0
    if not np.any(positive):
        return float(fallback_max)
    vmax = float(np.percentile(image[positive], 98))
    return max(vmax, float(fallback_max))


def build_histogram_max_label_from_group(
    data: np.ndarray,
    time_threshold: int,
    *,
    active_point: int = 1,
    normalize_intensity: bool = False,
    tof_window: tuple[int, int] | None = None,
) -> torch.Tensor:
    """基于逐像素 ToF 直方图的最大值法生成 ``[2, 64, 64]`` 图像。

    通道 0 为最大计数对应的 ToF bin，通道 1 为该 bin 的最大计数。
    ``tof_window`` 非空时只在指定闭区间内找峰值, 用于目标窗口诊断。
    当 ``normalize_intensity=True`` 时，强度会额外除以 ``pages_per_group``，
    仅用于和模型输出的 ``[0, 1]`` 强度做量纲一致的误差统计。
    """
    if data.ndim != 2 or data.shape[0] != 64 * 64:
        raise ValueError(f"data must have shape [4096, P], got {data.shape}")
    if time_threshold <= 0:
        raise ValueError("time_threshold must be a positive integer")
    if active_point <= 0:
        raise ValueError("active_point must be a positive integer")

    pages_per_group = int(data.shape[1])
    label = np.zeros((2, 64, 64), dtype=np.float32)
    if pages_per_group <= 0:
        return torch.from_numpy(label)

    group = data.astype(np.int32, copy=False)
    if tof_window is None:
        min_bin, max_bin = 1, int(time_threshold)
    else:
        min_bin = max(1, int(tof_window[0]))
        max_bin = min(int(time_threshold), int(tof_window[1]))
        if min_bin > max_bin:
            raise ValueError(
                f"tof_window must overlap valid range [1, {time_threshold}], "
                f"got {tof_window}"
            )
    intensity_scale = float(pages_per_group) if normalize_intensity else 1.0

    for pixel_index in range(group.shape[0]):
        values = group[pixel_index]
        valid_values = values[(values >= min_bin) & (values <= max_bin)]
        if valid_values.size == 0:
            continue

        counts = np.bincount(valid_values, minlength=max_bin + 1)
        counts[0] = 0
        best_count = int(counts.max())
        if best_count < int(active_point):
            continue

        best_tof = int(counts.argmax())
        y_idx = pixel_index // 64
        x_idx = pixel_index % 64
        label[0, y_idx, x_idx] = float(best_tof)
        label[1, y_idx, x_idx] = float(best_count) / intensity_scale

    return torch.from_numpy(label)


def _resolve_single_raw_metadata(
    cfg: SNNConfig,
    raw_path: Path,
    *,
    pages_per_group: int,
) -> dict[str, Any]:
    """从配置 CSV 中查找单个 raw 的 metadata, 供 ``label_prior`` 定位使用。"""
    if not cfg.data_paths:
        return {}

    records = collect_raw_records(
        cfg.data_paths,
        csv_paths=cfg.csv_paths,
        recursive=cfg.recursive,
        skip_missing_csv_raw=cfg.skip_missing_csv_raw,
    )
    if cfg.use_precomputed_labels or cfg.require_precomputed_labels:
        records = attach_precomputed_label_metadata(
            records,
            pages_per_group=pages_per_group,
            total_pages=cfg.total_pages,
            label_dir_name=cfg.precomputed_label_dir_name,
        )

    target = str(raw_path.resolve())
    for record_path, metadata in records:
        if str(Path(record_path).resolve()) == target:
            return dict(metadata)
    return {}


def _label_mode_from_metadata(cfg: SNNConfig, metadata: Mapping[str, Any]) -> str:
    """给 summary 写入当前 dataset label 的来源, 避免和 max-count baseline 混淆。"""
    if not cfg.return_label:
        return "disabled"
    if cfg.use_precomputed_labels and metadata.get("precomputed_label_dir"):
        return "precomputed_label_pool"
    if cfg.use_precomputed_labels or cfg.require_precomputed_labels:
        return "weak_label_fallback_no_metadata"
    return "weak_label_from_current_group"


def _masked_mean(array: np.ndarray, mask: np.ndarray) -> float | None:
    """安全计算 mask 区域均值; mask 为空时返回 None 便于 JSON 表达。"""
    if not np.any(mask):
        return None
    return float(array[mask].mean())


def _masked_abs_mean(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    """安全计算 mask 区域 MAE。"""
    if not np.any(mask):
        return None
    return float(np.abs(a[mask] - b[mask]).mean())


def _array_summary(prefix: str, array: np.ndarray) -> dict[str, float]:
    """生成一组稳定的全局分布统计。"""
    data = np.asarray(array, dtype=np.float32)
    return {
        f"{prefix}_min": float(data.min()),
        f"{prefix}_p05": float(np.percentile(data, 5)),
        f"{prefix}_p50": float(np.percentile(data, 50)),
        f"{prefix}_p95": float(np.percentile(data, 95)),
        f"{prefix}_max": float(data.max()),
        f"{prefix}_mean": float(data.mean()),
        f"{prefix}_std": float(data.std()),
    }


def _positive_counts(prefix: str, array: np.ndarray) -> dict[str, int]:
    """统计强度图在常用阈值下的非零/高响应像素数。"""
    data = np.asarray(array, dtype=np.float32)
    return {
        f"{prefix}_positive_pixels": int((data > 0).sum()),
        f"{prefix}_gt_0p01_pixels": int((data > 0.01).sum()),
        f"{prefix}_gt_0p05_pixels": int((data > 0.05).sum()),
        f"{prefix}_gt_0p10_pixels": int((data > 0.10).sum()),
        f"{prefix}_gt_0p50_pixels": int((data > 0.50).sum()),
    }


def _top_depth_bins(depth_image: np.ndarray, limit: int = 12) -> list[dict[str, int]]:
    """统计全局 max-count depth 图中出现最多的 ToF bin。"""
    depth = np.asarray(depth_image, dtype=np.int32)
    values, counts = np.unique(depth, return_counts=True)
    pairs = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)
    return [
        {"bin": int(bin_value), "pixels": int(pixel_count)}
        for bin_value, pixel_count in pairs[:limit]
    ]


def _save_single_image(
    image: np.ndarray,
    path: Path,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """保存单通道热力图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prediction_images(
    output: torch.Tensor,
    max_label_count: torch.Tensor,
    max_label_normalized: torch.Tensor,
    run_dir: Path,
    *,
    time_threshold: int,
) -> dict[str, str]:
    """保存模型输出 npy 与最大值法 baseline 的深度/强度对比图。"""
    model_output_npy = _to_prediction_npy(output)
    model_depth = _to_image_2d(output[0, 0])
    model_intensity = _to_image_2d(output[0, 1])
    max_depth = _to_image_2d(max_label_count[0])
    max_intensity_count = _to_image_2d(max_label_count[1])
    max_intensity_normalized = _to_image_2d(max_label_normalized[1])

    depth_vmin, depth_vmax = _valid_percentile_range(
        np.concatenate([model_depth, max_depth], axis=0),
        fallback_max=float(time_threshold),
    )
    model_intensity_vmax = _positive_percentile_max(model_intensity, 0.01)
    max_intensity_count_vmax = _positive_percentile_max(max_intensity_count, 1.0)
    comparison_intensity_vmax = max(
        _positive_percentile_max(model_intensity, 0.01),
        _positive_percentile_max(max_intensity_normalized, 0.01),
    )

    image_dir = run_dir / "images"
    saved_paths = {
        "model_output_npy": str(image_dir / "model_output.npy"),
        "model_depth_png": str(image_dir / "model_depth.png"),
        "model_intensity_png": str(image_dir / "model_intensity.png"),
        "max_depth_png": str(image_dir / "max_method_depth.png"),
        "max_intensity_png": str(image_dir / "max_method_intensity.png"),
        "comparison_png": str(image_dir / "model_vs_max_method.png"),
    }

    Path(saved_paths["model_output_npy"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(saved_paths["model_output_npy"], model_output_npy)
    _save_single_image(
        model_depth,
        Path(saved_paths["model_depth_png"]),
        title="Model Depth",
        cmap="turbo",
        vmin=depth_vmin,
        vmax=depth_vmax,
    )
    _save_single_image(
        model_intensity,
        Path(saved_paths["model_intensity_png"]),
        title="Model Intensity",
        cmap="inferno",
        vmin=0.0,
        vmax=model_intensity_vmax,
    )
    _save_single_image(
        max_depth,
        Path(saved_paths["max_depth_png"]),
        title="Max-count Depth",
        cmap="turbo",
        vmin=depth_vmin,
        vmax=depth_vmax,
    )
    _save_single_image(
        max_intensity_count,
        Path(saved_paths["max_intensity_png"]),
        title="Max-count Intensity (Counts)",
        cmap="inferno",
        vmin=0.0,
        vmax=max_intensity_count_vmax,
    )

    depth_diff = np.abs(model_depth - max_depth)
    intensity_diff = np.abs(model_intensity - max_intensity_normalized)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        (model_depth, "Model Depth", "turbo", depth_vmin, depth_vmax),
        (max_depth, "Max-count Depth", "turbo", depth_vmin, depth_vmax),
        (depth_diff, "Abs Depth Diff", "magma", 0.0, max(float(np.percentile(depth_diff, 98)), 1.0)),
        (model_intensity, "Model Intensity", "inferno", 0.0, comparison_intensity_vmax),
        (
            max_intensity_normalized,
            "Max-count Intensity (/P)",
            "inferno",
            0.0,
            comparison_intensity_vmax,
        ),
        (
            intensity_diff,
            "Abs Intensity Diff",
            "magma",
            0.0,
            max(float(np.percentile(intensity_diff, 98)), 0.01),
        ),
    ]
    for ax, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(saved_paths["comparison_png"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return saved_paths


def _pick_diagnostic_pixel(
    *,
    target_mask_np: np.ndarray | None,
    selectivity_np: np.ndarray | None,
    output_intensity_np: np.ndarray,
) -> tuple[int, int] | None:
    """为 gate 直方图诊断挑一个代表性目标像素。

    优先级: 目标窗口内 selectivity 最高 → 目标 mask 内强度最高 → 全图强度最高。
    返回 ``(y_idx, x_idx)``; 全图无正值时返回 None。
    """
    if target_mask_np is not None and np.any(target_mask_np):
        if selectivity_np is not None:
            scored = np.where(target_mask_np, selectivity_np, -1.0)
        else:
            scored = np.where(target_mask_np, output_intensity_np, -1.0)
        flat_index = int(np.argmax(scored))
        return flat_index // 64, flat_index % 64

    if np.any(output_intensity_np > 0):
        flat_index = int(np.argmax(output_intensity_np))
        return flat_index // 64, flat_index % 64

    return None


def save_diagnostic_images(
    coarse_depth_np: np.ndarray,
    refined_depth_np: np.ndarray,
    depth_residual_np: np.ndarray,
    confidence_np: np.ndarray,
    selectivity_np: np.ndarray | None,
    gate_hist: torch.Tensor | None,
    run_dir: Path,
    *,
    depth_vmin: float,
    depth_vmax: float,
    target_mask_np: np.ndarray | None,
    peak_pixel: tuple[int, int] | None,
) -> dict[str, str]:
    """保存第 0 层诊断面板: 分离 coarse vs refined, 看深度高估来自哪一级。

    第 6 格画一个代表性目标像素的 gate 加权 ToF 直方图, 直接看 gate 把权重
    压在了哪个 bin (真实回波 vs 雾)。
    """
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    diag_path = str(image_dir / "diagnostics.png")

    resid_abs = float(np.percentile(np.abs(depth_residual_np), 98))
    resid_lim = max(resid_abs, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # (0,0) coarse depth: refine 前的原始深度估计
    im = axes[0, 0].imshow(coarse_depth_np, cmap="turbo", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 0].set_title("Coarse Depth (pre-refine)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0,1) refined depth: refine 后的最终输出
    im = axes[0, 1].imshow(refined_depth_np, cmap="turbo", vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 1].set_title("Refined Depth (output)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (0,2) depth residual = refined - coarse: refine 净修正量 (正=推高)
    im = axes[0, 2].imshow(depth_residual_np, cmap="coolwarm", vmin=-resid_lim, vmax=resid_lim)
    axes[0, 2].set_title("Depth Residual (refined - coarse)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (1,0) confidence: refine 残差被它缩放
    im = axes[1, 0].imshow(confidence_np, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Confidence")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (1,1) selectivity: peak_count / weight_sum, gate 选择性
    if selectivity_np is not None:
        im = axes[1, 1].imshow(selectivity_np, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Selectivity (peak/weight_sum)")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        axes[1, 1].set_title("Selectivity (n/a)")

    for ax in axes.ravel()[:5]:
        ax.set_xticks([])
        ax.set_yticks([])

    # (1,2) 目标像素的 gate 加权 ToF 直方图
    ax_hist = axes[1, 2]
    if gate_hist is not None and peak_pixel is not None:
        y_idx, x_idx = peak_pixel
        # gate_hist [B, t_max+1, H, W] → 取 batch0 该像素 → [t_max+1]
        hist_curve = gate_hist[0, :, y_idx, x_idx].numpy()
        bins = np.arange(hist_curve.shape[0])
        # 跳过 bin0 (无效 ToF 占位)
        ax_hist.bar(bins[1:], hist_curve[1:], width=1.0, color="#d6604d")
        peak_bin = int(np.argmax(hist_curve[1:])) + 1
        ax_hist.axvline(peak_bin, color="black", ls="--", lw=1.0, label=f"peak bin={peak_bin}")
        ax_hist.set_title(f"Gate Hist @ pixel(y={y_idx},x={x_idx})")
        ax_hist.set_xlabel("ToF bin")
        ax_hist.set_ylabel("gate-weighted count")
        ax_hist.legend(fontsize=8)
    else:
        ax_hist.set_title("Gate Hist (no target pixel)")

    fig.tight_layout()
    fig.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"diagnostics_png": diag_path}


def build_single_sample_dataset(
    cfg: SNNConfig,
    raw_path: str | Path,
    *,
    pages_per_group: int,
    group_index: int,
) -> SpadRawGroupDataset:
    """为指定 raw 文件和组号构建只含一个样本的数据集。"""
    raw_path = Path(raw_path).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw file not found: {raw_path}")
    if pages_per_group <= 0:
        raise ValueError("pages_per_group must be positive")

    raw_metadata = _resolve_single_raw_metadata(
        cfg,
        raw_path,
        pages_per_group=pages_per_group,
    )
    can_use_precomputed = bool(raw_metadata.get("precomputed_label_dir"))

    # 先让 Dataset 推断完整组列表, 再从中选出目标组, 避免重复实现 page 数推断逻辑。
    full_dataset = SpadRawGroupDataset(
        raw_paths=[raw_path],
        pages_per_group=pages_per_group,
        total_pages=cfg.total_pages,
        time_threshold=cfg.time_threshold,
        return_label=cfg.return_label,
        normalize=cfg.normalize_input,
        shuffle_pages=False,
        active_point=cfg.active_point,
        cache_size=cfg.cache_size,
        raw_load_mode=cfg.raw_load_mode,
        raw_metadata=[raw_metadata],
        use_precomputed_labels=cfg.use_precomputed_labels and can_use_precomputed,
        require_precomputed_labels=cfg.require_precomputed_labels and can_use_precomputed,
        precomputed_labels_per_class=cfg.precomputed_labels_per_class,
    )
    if group_index < 0 or group_index >= len(full_dataset.samples):
        raise IndexError(
            f"group_index {group_index} out of range; "
            f"available groups: 0..{len(full_dataset.samples) - 1}"
        )

    selected = full_dataset.samples[group_index]
    sample = RawGroupSample(
        raw_path=selected.raw_path,
        group_index=selected.group_index,
        total_pages=selected.total_pages,
    )
    return SpadRawGroupDataset(
        raw_paths=[raw_path],
        pages_per_group=pages_per_group,
        total_pages=cfg.total_pages,
        time_threshold=cfg.time_threshold,
        return_label=cfg.return_label,
        normalize=cfg.normalize_input,
        shuffle_pages=False,
        active_point=cfg.active_point,
        cache_size=cfg.cache_size,
        raw_load_mode=cfg.raw_load_mode,
        raw_metadata=[raw_metadata],
        samples=[sample],
        use_precomputed_labels=cfg.use_precomputed_labels and can_use_precomputed,
        require_precomputed_labels=cfg.require_precomputed_labels and can_use_precomputed,
        precomputed_labels_per_class=cfg.precomputed_labels_per_class,
    )


@torch.no_grad()
def run_single_test(
    cfg: SNNConfig,
    raw_path: str | Path,
    *,
    pages_per_group: int | None = None,
    group_index: int = 0,
    save_prediction: bool = False,
    target_window: tuple[int, int] | None = None,
    target_min_count: int | None = None,
) -> dict[str, object]:
    """把一个 raw 分组送入当前配置的模型并返回基本统计。"""
    seed_everything(cfg.seed)
    resolved_pages_per_group = (
        int(pages_per_group)
        if pages_per_group is not None
        else int(cfg.pages_per_group)
    )
    cfg = cfg.clone_with(pages_per_group=resolved_pages_per_group)
    dataset = build_single_sample_dataset(
        cfg,
        raw_path,
        pages_per_group=resolved_pages_per_group,
        group_index=group_index,
    )
    item = dataset[0]
    batch = spad_time_first_collate([item])
    metadata = item.get("metadata", {})

    device = cfg.resolved_device()
    model = cfg.build_model().to(device)
    if cfg.checkpoint_path:
        load_checkpoint(cfg.checkpoint_path, model, map_location=device)
    model.eval()

    frames = batch["frames"].to(device)
    model_input = prepare_model_input(frames).to(device)
    result = model(model_input, return_sequence=False)

    output = result["output"].detach().cpu()
    coarse_depth = result["depth_coarse"].detach().cpu()
    coarse_intensity = result["intensity_coarse"].detach().cpu()
    confidence = result["confidence"].detach().cpu()
    support = result.get("support")
    selectivity = result.get("selectivity")
    support = support.detach().cpu() if support is not None else None
    selectivity = selectivity.detach().cpu() if selectivity is not None else None
    gate_hist = result.get("gate_hist")
    gate_hist = gate_hist.detach().cpu() if gate_hist is not None else None
    sample = dataset.samples[0]
    raw_group = read_raw_group(
        sample.raw_path,
        group_index=sample.group_index,
        pages_per_group=resolved_pages_per_group,
        total_pages=sample.total_pages,
    )
    max_label_count = build_histogram_max_label_from_group(
        raw_group,
        cfg.time_threshold,
        active_point=cfg.active_point,
    )
    max_label_normalized = build_histogram_max_label_from_group(
        raw_group,
        cfg.time_threshold,
        active_point=cfg.active_point,
        normalize_intensity=True,
    )
    target_label_count = None
    target_label_normalized = None
    target_mask_np = None
    if target_window is not None:
        target_active_point = int(target_min_count) if target_min_count is not None else cfg.active_point
        target_label_count = build_histogram_max_label_from_group(
            raw_group,
            cfg.time_threshold,
            active_point=target_active_point,
            tof_window=target_window,
        )
        target_label_normalized = build_histogram_max_label_from_group(
            raw_group,
            cfg.time_threshold,
            active_point=target_active_point,
            normalize_intensity=True,
            tof_window=target_window,
        )
        target_mask_np = _to_image_2d(target_label_count[1]) > 0

    depth_diff = torch.abs(output[:, 0:1] - max_label_count[0:1].unsqueeze(0))
    intensity_diff = torch.abs(output[:, 1:2] - max_label_normalized[1:2].unsqueeze(0))
    output_depth_np = _to_image_2d(output[0, 0])
    output_intensity_np = _to_image_2d(output[0, 1])
    coarse_depth_np = _to_image_2d(coarse_depth[0, 0])
    coarse_intensity_np = _to_image_2d(coarse_intensity[0, 0])
    confidence_np = _to_image_2d(confidence[0, 0])
    support_np = _to_image_2d(support[0, 0]) if support is not None else None
    selectivity_np = _to_image_2d(selectivity[0, 0]) if selectivity is not None else None
    max_depth_np = _to_image_2d(max_label_count[0])
    max_intensity_np = _to_image_2d(max_label_normalized[1])
    depth_residual_np = output_depth_np - coarse_depth_np
    intensity_residual_np = output_intensity_np - coarse_intensity_np
    info: dict[str, object] = {
        "raw_path": str(Path(raw_path).resolve()),
        "pages_per_group": int(resolved_pages_per_group),
        "group_index": int(group_index),
        "target_class": metadata.get("target_class"),
        "fog_level": metadata.get("fog_level"),
        "dataset_label_mode": _label_mode_from_metadata(cfg, metadata),
        "frames_shape": list(batch["frames"].shape),
        "model_input_shape": list(model_input.shape),
        "output_shape": list(output.shape),
        "depth_min": float(output[:, 0:1].min().item()),
        "depth_max": float(output[:, 0:1].max().item()),
        "depth_mean": float(output[:, 0:1].mean().item()),
        "intensity_min": float(output[:, 1:2].min().item()),
        "intensity_max": float(output[:, 1:2].max().item()),
        "intensity_mean": float(output[:, 1:2].mean().item()),
        "coarse_depth_min": float(coarse_depth.min().item()),
        "coarse_depth_max": float(coarse_depth.max().item()),
        "coarse_depth_mean": float(coarse_depth.mean().item()),
        "coarse_intensity_min": float(coarse_intensity.min().item()),
        "coarse_intensity_max": float(coarse_intensity.max().item()),
        "coarse_intensity_mean": float(coarse_intensity.mean().item()),
        "confidence_min": float(confidence.min().item()),
        "confidence_max": float(confidence.max().item()),
        "confidence_mean": float(confidence.mean().item()),
        "max_method_depth_max": float(max_label_count[0:1].max().item()),
        "max_method_intensity_max": float(max_label_count[1:2].max().item()),
        "max_method_intensity_normalized_max": float(max_label_normalized[1:2].max().item()),
        "depth_abs_diff_mean": float(depth_diff.mean().item()),
        "depth_abs_diff_max": float(depth_diff.max().item()),
        "intensity_abs_diff_mean": float(intensity_diff.mean().item()),
        "intensity_abs_diff_max": float(intensity_diff.max().item()),
        "checkpoint": cfg.checkpoint_path,
    }
    if support is not None:
        info.update(
            {
                "support_min": float(support.min().item()),
                "support_max": float(support.max().item()),
                "support_mean": float(support.mean().item()),
            }
        )
    if selectivity is not None:
        info.update(
            {
                "selectivity_min": float(selectivity.min().item()),
                "selectivity_max": float(selectivity.max().item()),
                "selectivity_mean": float(selectivity.mean().item()),
            }
        )
    info["max_method_depth_top_bins"] = _top_depth_bins(max_depth_np)
    info.update(_positive_counts("max_method_intensity", max_intensity_np))
    info.update(_positive_counts("model_intensity", output_intensity_np))
    info.update(_positive_counts("coarse_intensity", coarse_intensity_np))
    info.update(_array_summary("refine_depth_residual", depth_residual_np))
    info.update(_array_summary("refine_intensity_residual", intensity_residual_np))
    if target_window is not None and target_label_count is not None and target_label_normalized is not None:
        target_depth_np = _to_image_2d(target_label_count[0])
        target_intensity_np = _to_image_2d(target_label_normalized[1])
        assert target_mask_np is not None
        info.update(
            {
                "target_window": [int(target_window[0]), int(target_window[1])],
                "target_window_min_count": int(
                    target_min_count if target_min_count is not None else cfg.active_point
                ),
                "target_window_pixels": int(target_mask_np.sum()),
                "target_window_depth_max": float(target_label_count[0:1].max().item()),
                "target_window_intensity_max": float(target_label_count[1:2].max().item()),
                "target_window_intensity_normalized_max": float(
                    target_label_normalized[1:2].max().item()
                ),
                "target_window_depth_mean_on_target": _masked_mean(target_depth_np, target_mask_np),
                "target_window_model_depth_mean_on_target": _masked_mean(output_depth_np, target_mask_np),
                "target_window_coarse_depth_mean_on_target": _masked_mean(coarse_depth_np, target_mask_np),
                "target_window_model_intensity_mean_on_target": _masked_mean(
                    output_intensity_np,
                    target_mask_np,
                ),
                "target_window_coarse_intensity_mean_on_target": _masked_mean(
                    coarse_intensity_np,
                    target_mask_np,
                ),
                "target_window_confidence_mean_on_target": _masked_mean(confidence_np, target_mask_np),
                "target_window_support_mean_on_target": (
                    _masked_mean(support_np, target_mask_np) if support_np is not None else None
                ),
                "target_window_selectivity_mean_on_target": (
                    _masked_mean(selectivity_np, target_mask_np) if selectivity_np is not None else None
                ),
                "target_window_depth_abs_diff_mean_on_target": _masked_abs_mean(
                    output_depth_np,
                    target_depth_np,
                    target_mask_np,
                ),
                "target_window_coarse_depth_abs_diff_mean_on_target": _masked_abs_mean(
                    coarse_depth_np,
                    target_depth_np,
                    target_mask_np,
                ),
                "target_window_intensity_abs_diff_mean_on_target": _masked_abs_mean(
                    output_intensity_np,
                    target_intensity_np,
                    target_mask_np,
                ),
            }
        )

    if "label" in batch:
        label = batch["label"]
        info["label_shape"] = list(label.shape)
        info["label_depth_max"] = float(label[:, 0:1].max().item())
        info["label_intensity_max"] = float(label[:, 1:2].max().item())
        label_intensity_np = _to_image_2d(label[0, 1])
        info.update(_positive_counts("label_intensity", label_intensity_np))

    if save_prediction:
        run_dir = make_log_run_dir(cfg, "test1")
        cfg.save(run_dir / "config.json")
        saved_images = save_prediction_images(
            output,
            max_label_count,
            max_label_normalized,
            run_dir,
            time_threshold=cfg.time_threshold,
        )
        info.update(saved_images)

        # 第 0 层诊断面板: 分离 coarse vs refined 深度, 定位高估来自哪一级。
        # 深度色阶与 save_prediction_images 一致, 便于跨图对照。
        diag_depth_vmin, diag_depth_vmax = _valid_percentile_range(
            np.concatenate([output_depth_np, max_depth_np], axis=0),
            fallback_max=float(cfg.time_threshold),
        )
        # peak_pixel: 优先选目标窗口内 selectivity 最高的像素 (最代表"成功选中目标"),
        # 无窗口或无 selectivity 时退回到目标 mask 质心, 再退回全图最高强度像素。
        peak_pixel = _pick_diagnostic_pixel(
            target_mask_np=target_mask_np,
            selectivity_np=selectivity_np,
            output_intensity_np=output_intensity_np,
        )
        if peak_pixel is not None:
            info["diagnostic_pixel"] = [int(peak_pixel[0]), int(peak_pixel[1])]
        diag_images = save_diagnostic_images(
            coarse_depth_np,
            output_depth_np,
            depth_residual_np,
            confidence_np,
            selectivity_np,
            gate_hist,
            run_dir,
            depth_vmin=diag_depth_vmin,
            depth_vmax=diag_depth_vmax,
            target_mask_np=target_mask_np,
            peak_pixel=peak_pixel,
        )
        info.update(diag_images)

        with (run_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
            json.dump(info, file_obj, indent=2, ensure_ascii=False)
        info["run_dir"] = str(run_dir)

    print(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="使用 SPAD SNN 测试一个 raw 分组")
    add_config_arguments(parser)
    parser.add_argument("--raw-path", required=True, help="单个 .raw 文件路径")
    parser.add_argument("--group-index", type=int, default=0, help="从 0 开始的分组索引")
    parser.add_argument(
        "--target-window",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="可选目标 ToF 闭区间, 仅用于额外 masked 诊断; 不改变全局 max-count baseline",
    )
    parser.add_argument(
        "--target-min-count",
        type=int,
        default=None,
        help="目标窗口诊断的最小峰值计数, 含等号; 与 i>15 对齐时传 16",
    )
    parser.add_argument(
        "--save-prediction",
        action="store_true",
        help="保存模型输出 npy、最大值法 baseline、差异对比 PNG 图片和 summary.json",
    )
    return parser


def main() -> None:
    """执行单样本推理。"""
    args = build_argparser().parse_args()
    cfg = config_from_checkpoint_and_args(args)
    run_single_test(
        cfg,
        args.raw_path,
        pages_per_group=args.pages_per_group,
        group_index=args.group_index,
        save_prediction=args.save_prediction,
        target_window=tuple(args.target_window) if args.target_window else None,
        target_min_count=args.target_min_count,
    )


def main_without_cli() -> None:
    """无 CLI 直接运行入口; 在这里显式修改单样本测试参数。"""
    # ===== Editable parameters =====
    # checkpoint_path = r"D:\\PYproject\\SPAD\\checkpoints\\SNN\\train_20260604_113734\\last.pth"
    checkpoint_path = r"D:\\PYproject\\SPAD\\checkpoints\\SNN\\train_20260611_011445\\best.pth"  # 005820 154437 
    raw_path = r"D:\\PYproject\\SPADdata\\0917\\2025-09-17_17-33-26_Delay-0_Width-2000.raw"  # M
    # raw_path = r"D:\\PYproject\\SPADdata\\0826\\2025-08-26_17-26-44_Delay-0_Width-2000.raw"  # 5
    pages_per_group = 640
    group_index = 71
    target_window = None
    target_min_count = None
    save_prediction = True

    # 测试输入参数显式在这里指定, 不从 checkpoint/config 继承。
    # 其余模型结构参数仍优先从 checkpoint 读取, 以保证权重能正确加载。
    args = argparse.Namespace(
        config=None,
        checkpoint_path=checkpoint_path or None,
        data_paths=None,
        csv_paths=None,
        skip_missing_csv_raw=False,
        pages_per_group=pages_per_group,
        total_pages=None,
        time_threshold=None,
        raw_load_mode=None,
        split_ratios=None,
        batch_size=None,
        num_workers=None,
        pin_memory=None,
        persistent_workers=None,
        prefetch_factor=None,
        precompute_model_input=None,
        epochs=None,
        lr=None,
        weight_decay=None,
        grad_clip=None,
        grad_accum_steps=None,
        device=None,
        log_dir=None,
        checkpoint_dir=None,
        output_dir=None,
        run_name=None,
        model_backend=None,
        encoding_mode=None,
        embed_dim=None,
        lut_init=None,
        C=None,
        chunk_size=None,
        spike_mode=None,
        spike_backend=None,
        num_blocks=None,
        refine_mid=None,
        return_sequence=False,
        w_gt=None,
        w_ssim=None,
        w_var=None,
        w_sparse=None,
        w_smooth=None,
        w_lut_smooth=None,
        w_lut_norm=None,
        sigma_target=None,
        rho_target=None,
        beta_smooth=None,
        ssim_kernel_size=None,
        ssim_smooth_kernel_size=None,
        gt_use_mask=None,
        ssim_use_mask=None,
        num_aug=None,
        tof_shift_max=None,
        tof_shift_prob=None,
        page_dropout_prob=None,
        amp=None,
        tf32=None,
        cudnn_benchmark=None,
        cuda_prefetch=None,
        progress_interval=None,
        recursive=False,
        no_label=False,
        normalize_input=False,
        augment_train=False,
        page_dropout=False,
        shuffle_pages=False,
        page_shuffle=False,
    )
    cfg = config_from_checkpoint_and_args(args)
    run_single_test(
        cfg,
        raw_path,
        pages_per_group=pages_per_group,
        group_index=group_index,
        save_prediction=save_prediction,
        target_window=target_window,
        target_min_count=target_min_count,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        main_without_cli()
