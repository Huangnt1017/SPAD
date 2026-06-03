"""SPAD SNN 模型的单样本冒烟测试与推理入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    seed_everything,
    spad_time_first_collate,
)
from SNN_based_method.scripts.runtime import (
    add_config_arguments,
    config_from_checkpoint_and_args,
    load_checkpoint,
    make_log_run_dir,
    prepare_model_input,
    reset_spiking_state,
)


def build_single_sample_dataset(
    cfg: SNNConfig,
    raw_path: str | Path,
    group_index: int,
) -> SpadRawGroupDataset:
    """为指定 raw 文件和组号构建只含一个样本的数据集。"""
    raw_path = Path(raw_path).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw file not found: {raw_path}")

    # 先让 Dataset 推断完整组列表, 再从中选出目标组, 避免重复实现 page 数推断逻辑。
    full_dataset = SpadRawGroupDataset(
        raw_paths=[raw_path],
        pages_per_group=cfg.pages_per_group,
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
        pages_per_group=cfg.pages_per_group,
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
    group_index: int = 0,
    save_prediction: bool = False,
) -> dict[str, object]:
    """把一个 raw 分组送入当前配置的模型并返回基本统计。"""
    seed_everything(cfg.seed)
    dataset = build_single_sample_dataset(cfg, raw_path, group_index)
    batch = spad_time_first_collate([dataset[0]])

    device = cfg.resolved_device()
    model = cfg.build_model().to(device)
    if cfg.checkpoint_path:
        load_checkpoint(cfg.checkpoint_path, model, map_location=device)
    model.eval()

    frames = batch["frames"].to(device)
    model_input = prepare_model_input(frames).to(device)
    result = model(model_input)
    reset_spiking_state(model)

    output = result["output"].detach().cpu()
    info: dict[str, object] = {
        "raw_path": str(Path(raw_path).resolve()),
        "group_index": int(group_index),
        "frames_shape": list(batch["frames"].shape),
        "model_input_shape": list(model_input.shape),
        "output_shape": list(output.shape),
        "depth_min": float(output[:, 0:1].min().item()),
        "depth_max": float(output[:, 0:1].max().item()),
        "intensity_min": float(output[:, 1:2].min().item()),
        "intensity_max": float(output[:, 1:2].max().item()),
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
        np.save(run_dir / "output.npy", output.numpy())
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
    parser.add_argument("--save-prediction", action="store_true", help="保存 output.npy 和 summary.json")
    return parser


def main() -> None:
    """执行单样本推理。"""
    args = build_argparser().parse_args()
    cfg = config_from_checkpoint_and_args(args)
    run_single_test(
        cfg,
        args.raw_path,
        group_index=args.group_index,
        save_prediction=args.save_prediction,
    )


if __name__ == "__main__":
    main()
