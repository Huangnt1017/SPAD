"""SPAD SNN 模型的批量测试入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.SNN_config import SNNConfig
from SNN_based_method.scripts.data import seed_everything
from SNN_based_method.scripts.runtime import (
    add_config_arguments,
    config_from_checkpoint_and_args,
    divide_average,
    load_checkpoint,
    make_log_run_dir,
    prepare_model_input,
    reduce_loss_dict,
    update_average,
)

DEFAULT_TEST_DATA_PATHS = [r"D:\PYproject\SPADdata\0917"]
DEFAULT_TEST_CSV_PATHS = [r"D:\PYproject\SPADdata\0917\917group.csv"]
DEFAULT_TRAIN_DATA_PATHS = [
    r"D:\PYproject\SPADdata\0825",
    r"D:\PYproject\SPADdata\0826",
]


def _same_paths(left: list[str] | None, right: list[str]) -> bool:
    """比较路径集合, 用于识别从 checkpoint 继承的训练目录。"""
    if left is None:
        return False
    left_resolved = {str(Path(path).resolve()).lower() for path in left}
    right_resolved = {str(Path(path).resolve()).lower() for path in right}
    return left_resolved == right_resolved


def apply_default_test_paths(cfg: SNNConfig, args: argparse.Namespace) -> SNNConfig:
    """测试默认使用 0917; 显式命令行路径优先。"""
    if getattr(args, "data_paths", None) is not None:
        return cfg
    if cfg.data_paths and not _same_paths(cfg.data_paths, DEFAULT_TRAIN_DATA_PATHS):
        return cfg
    return cfg.clone_with(
        data_paths=DEFAULT_TEST_DATA_PATHS,
        csv_paths=DEFAULT_TEST_CSV_PATHS,
    )


@torch.no_grad()
def run_test(
    cfg: SNNConfig,
    *,
    save_predictions: bool = False,
) -> dict[str, object]:
    """执行批量测试并返回汇总统计。"""
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path is required for testing")

    seed_everything(cfg.seed)
    data_loader, dataset = cfg.build_dataloader(shuffle=False)
    device = cfg.resolved_device()
    model = cfg.build_model().to(device)
    criterion = cfg.build_loss().to(device)
    metrics = cfg.build_metrics()
    load_checkpoint(cfg.checkpoint_path, model, map_location=device)
    model.eval()

    run_dir = make_log_run_dir(cfg, "test")
    cfg.save(run_dir / "config.json")
    prediction_dir = run_dir / "predictions"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    loss_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    total_loss = 0.0
    num_steps = 0

    for batch_index, batch in enumerate(tqdm(data_loader, desc="test")):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch.get("label")
        labels = labels.to(device, non_blocking=True) if labels is not None else None

        model_input = prepare_model_input(frames).to(device, non_blocking=True)
        result = model(model_input)
        loss, loss_items = criterion(result, labels)

        if labels is not None:
            update_average(metric_sums, metrics.compute(result, labels))

        if save_predictions:
            output = result["output"].detach().cpu().numpy()
            np.save(prediction_dir / f"batch_{batch_index:04d}_output.npy", output)

        total_loss += float(loss.detach().cpu().item())
        update_average(loss_sums, reduce_loss_dict(loss_items))
        num_steps += 1

    summary = {
        "num_samples": len(dataset),
        "num_batches": num_steps,
        "loss": total_loss / max(num_steps, 1),
        "loss_items": divide_average(loss_sums, num_steps),
        "metrics": divide_average(metric_sums, num_steps),
        "checkpoint": cfg.checkpoint_path,
        "config": cfg.to_dict(),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print(f"run_dir={run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="测试 SPAD SNN 成像模型")
    add_config_arguments(parser)
    parser.add_argument("--save-predictions", action="store_true", help="把每个 batch 的输出图保存为 .npy")
    return parser


def main() -> None:
    """执行标准批量测试流程。"""
    args = build_argparser().parse_args()
    cfg = config_from_checkpoint_and_args(args)
    cfg = apply_default_test_paths(cfg, args)
    run_test(cfg, save_predictions=args.save_predictions)


if __name__ == "__main__":
    main()
