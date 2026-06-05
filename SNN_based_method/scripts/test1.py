"""SPAD SNN 模型的单样本冒烟测试与推理入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from SNN_based_method.SNN_config import SNNConfig
from SNN_based_method.scripts.data import (
    RawGroupSample,
    SpadRawGroupDataset,
    read_raw_group,
    seed_everything,
    spad_time_first_collate,
)
from SNN_based_method.scripts.runtime import (
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
) -> torch.Tensor:
    """基于逐像素 ToF 直方图的最大值法生成 ``[2, 64, 64]`` 图像。

    通道 0 为最大计数对应的 ToF bin，通道 1 为该 bin 的最大计数。
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
    max_bin = int(time_threshold)
    intensity_scale = float(pages_per_group) if normalize_intensity else 1.0

    for pixel_index in range(group.shape[0]):
        values = group[pixel_index]
        valid_values = values[(values >= 1) & (values <= max_bin)]
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
    """保存模型输出与最大值法 baseline 的深度/强度对比图。"""
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
        "model_depth_png": str(image_dir / "model_depth.png"),
        "model_intensity_png": str(image_dir / "model_intensity.png"),
        "max_depth_png": str(image_dir / "max_method_depth.png"),
        "max_intensity_png": str(image_dir / "max_method_intensity.png"),
        "comparison_png": str(image_dir / "model_vs_max_method.png"),
    }

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
        samples=[sample],
    )


@torch.no_grad()
def run_single_test(
    cfg: SNNConfig,
    raw_path: str | Path,
    *,
    pages_per_group: int | None = None,
    group_index: int = 0,
    save_prediction: bool = False,
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
    batch = spad_time_first_collate([dataset[0]])

    device = cfg.resolved_device()
    model = cfg.build_model().to(device)
    if cfg.checkpoint_path:
        load_checkpoint(cfg.checkpoint_path, model, map_location=device)
    model.eval()

    frames = batch["frames"].to(device)
    model_input = prepare_model_input(frames).to(device)
    result = model(model_input, return_sequence=False)

    output = result["output"].detach().cpu()
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
    depth_diff = torch.abs(output[:, 0:1] - max_label_count[0:1].unsqueeze(0))
    intensity_diff = torch.abs(output[:, 1:2] - max_label_normalized[1:2].unsqueeze(0))
    info: dict[str, object] = {
        "raw_path": str(Path(raw_path).resolve()),
        "pages_per_group": int(resolved_pages_per_group),
        "group_index": int(group_index),
        "frames_shape": list(batch["frames"].shape),
        "model_input_shape": list(model_input.shape),
        "output_shape": list(output.shape),
        "depth_min": float(output[:, 0:1].min().item()),
        "depth_max": float(output[:, 0:1].max().item()),
        "intensity_min": float(output[:, 1:2].min().item()),
        "intensity_max": float(output[:, 1:2].max().item()),
        "max_method_depth_max": float(max_label_count[0:1].max().item()),
        "max_method_intensity_max": float(max_label_count[1:2].max().item()),
        "max_method_intensity_normalized_max": float(max_label_normalized[1:2].max().item()),
        "depth_abs_diff_mean": float(depth_diff.mean().item()),
        "depth_abs_diff_max": float(depth_diff.max().item()),
        "intensity_abs_diff_mean": float(intensity_diff.mean().item()),
        "intensity_abs_diff_max": float(intensity_diff.max().item()),
        "checkpoint": cfg.checkpoint_path,
    }

    if "label" in batch:
        label = batch["label"]
        info["label_shape"] = list(label.shape)
        info["label_depth_max"] = float(label[:, 0:1].max().item())
        info["label_intensity_max"] = float(label[:, 1:2].max().item())

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
        "--save-prediction",
        action="store_true",
        help="保存模型输出、最大值法 baseline 及差异对比 PNG 图片和 summary.json",
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
    )


def main_without_cli() -> None:
    """无 CLI 直接运行入口; 在这里显式修改单样本测试参数。"""
    # ===== Editable parameters =====
    checkpoint_path = r"D:\\PYproject\\SPAD\\checkpoints\\SNN\\train_20260604_113734\\last.pth"
    raw_path = r"D:\\PYproject\\SPADdata\\0826\\2025-08-26_16-59-37_Delay-0_Width-2000.raw"  # 
    pages_per_group = 2400
    group_index = 15
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
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        main_without_cli()
