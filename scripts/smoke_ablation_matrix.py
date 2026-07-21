"""对消融注册表执行单批次前向/反向与配置语义检查。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\smoke_ablation_matrix.py --families structure_core --device cuda

无参数运行：在 CPU 上检查 core 的四种唯一配置。输入 ``(B,N,4)``，输出必须含
``logits (B,26)`` 和 ``box_pred (B,3)``；A2/A3 额外含 ``seg_logits (B,N)``。
结果写入 ``outputs/ABL/smoke/ablation_smoke_<timestamp>.json``，不会覆盖正式训练资产。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

import torch

from scripts.ablation_registry import PROJECT_ROOT, AblationExperiment, select_experiments
from scripts.train import build_model, set_seed
from utils.loss import PointCloudMultiTaskLoss


SMOKE_ROOT = PROJECT_ROOT / "outputs" / "ABL" / "smoke"


def parse_csv(value: str) -> List[str]:
    """解析逗号分隔列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def config_key(experiment: AblationExperiment) -> Tuple[object, ...]:
    """用于去重同构的不同 seed 实验。"""
    return (
        experiment.model,
        experiment.box_head,
        experiment.seg_loss_weight,
        experiment.gcn_aggregation,
        experiment.gcn_operator,
        experiment.gcn_exclude_self,
        experiment.gcn_feature_residual,
        experiment.gcn_use_physical_branch,
        experiment.gcn_use_se_gate,
        experiment.gcn_use_coord_residual,
    )


def unique_experiments(
    experiments: Iterable[AblationExperiment],
) -> List[AblationExperiment]:
    """保留每种有效配置的首个实验。"""
    selected: List[AblationExperiment] = []
    seen = set()
    for experiment in experiments:
        key = config_key(experiment)
        if key in seen:
            continue
        seen.add(key)
        selected.append(experiment)
    return selected


def build_args(experiment: AblationExperiment, num_points: int) -> argparse.Namespace:
    """构建模型所需参数。"""
    return argparse.Namespace(
        model=experiment.model,
        box_head=experiment.box_head,
        num_points=num_points,
        gcn_k=None,
        gcn_use_checkpoint=False,
        gcn_aggregation=experiment.gcn_aggregation,
        gcn_operator=experiment.gcn_operator,
        gcn_exclude_self=experiment.gcn_exclude_self,
        gcn_feature_residual=experiment.gcn_feature_residual,
        gcn_coord_scale_init=0.1,
        gcn_legacy_mode=False,
        gcn_use_physical_branch=experiment.gcn_use_physical_branch,
        gcn_use_se_gate=experiment.gcn_use_se_gate,
        gcn_use_coord_residual=experiment.gcn_use_coord_residual,
    )


def run_one(
    experiment: AblationExperiment,
    *,
    device: torch.device,
    batch_size: int,
    num_points: int,
) -> Dict[str, object]:
    """执行一个配置的单批次 forward/backward。"""
    set_seed(20260717)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(
        experiment.model,
        num_classes=26,
        project_root=PROJECT_ROOT,
        args=build_args(experiment, num_points),
    ).to(device)
    model.train()

    points = torch.rand(batch_size, num_points, 4, device=device)
    labels = torch.arange(batch_size, device=device) % 26
    centers = torch.tensor(
        [0.5, 0.5, 0.75],
        device=device,
        dtype=points.dtype,
    ).repeat(batch_size, 1)
    half = torch.tensor(
        [0.10, 0.10, 0.02],
        device=device,
        dtype=points.dtype,
    )
    box_targets = torch.stack(
        (
            centers[:, 0] - half[0],
            centers[:, 0] + half[0],
            centers[:, 1] - half[1],
            centers[:, 1] + half[1],
            centers[:, 2] - half[2],
            centers[:, 2] + half[2],
        ),
        dim=-1,
    )
    valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    outputs = model(points)
    effective_head = "centroid" if isinstance(outputs, dict) and "seg_logits" in outputs else "mlp"
    effective_seg = experiment.seg_loss_weight if effective_head == "centroid" else 0.0
    criterion = PointCloudMultiTaskLoss(
        cls_weight=1.0,
        box_weight=10.0,
        label_smoothing=0.1,
        auto_balance=False,
        seg_weight=effective_seg,
    ).to(device)
    losses = criterion(
        outputs,
        labels,
        box_targets,
        valid_mask,
        points=points,
    )
    losses["total_loss"].backward()

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    box_pred = outputs["box_pred"] if isinstance(outputs, dict) else outputs[1]
    if tuple(logits.shape) != (batch_size, 26):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if tuple(box_pred.shape) != (batch_size, 3):
        raise AssertionError(f"Unexpected box shape: {tuple(box_pred.shape)}")
    if effective_head != experiment.box_head:
        raise AssertionError(
            f"{experiment.experiment_id}: expected head={experiment.box_head}, got {effective_head}"
        )
    if experiment.box_head == "centroid":
        seg_logits = outputs["seg_logits"]
        if tuple(seg_logits.shape) != (batch_size, num_points):
            raise AssertionError(f"Unexpected seg shape: {tuple(seg_logits.shape)}")
    elif float(losses["seg_loss"].detach()) != 0.0:
        raise AssertionError("MLP/baseline seg_loss must be exactly zero")

    result: Dict[str, object] = {
        "experiment_id": experiment.experiment_id,
        "device": str(device),
        "input_shape": list(points.shape),
        "logits_shape": list(logits.shape),
        "box_shape": list(box_pred.shape),
        "seg_shape": (
            list(outputs["seg_logits"].shape)
            if isinstance(outputs, dict) and "seg_logits" in outputs
            else None
        ),
        "requested_box_head": experiment.box_head,
        "effective_box_head": effective_head,
        "requested_seg_loss_weight": experiment.seg_loss_weight,
        "effective_seg_loss_weight": effective_seg,
        "total_loss": float(losses["total_loss"].detach()),
        "seg_loss": float(losses["seg_loss"].detach()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        ),
    }
    del model, criterion, points, outputs, losses
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def verify_baseline_seg_equivalence(
    *,
    device: torch.device,
    batch_size: int,
    num_points: int,
) -> Dict[str, object]:
    """验证 baseline 在 seg_weight=0 与 0.5 时总损失完全一致。"""
    experiment = next(
        item for item in select_experiments(["core"]) if item.experiment_id == "A0_seed42"
    )
    set_seed(20260717)
    model = build_model(
        experiment.model,
        num_classes=26,
        project_root=PROJECT_ROOT,
        args=build_args(experiment, num_points),
    ).to(device)
    model.eval()
    points = torch.rand(batch_size, num_points, 4, device=device)
    labels = torch.arange(batch_size, device=device) % 26
    boxes = torch.tensor(
        [[0.4, 0.6, 0.4, 0.6, 0.7, 0.8]],
        device=device,
        dtype=points.dtype,
    ).repeat(batch_size, 1)
    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    outputs = model(points)
    losses = []
    for seg_weight in (0.0, 0.5):
        criterion = PointCloudMultiTaskLoss(
            cls_weight=1.0,
            box_weight=10.0,
            label_smoothing=0.1,
            auto_balance=False,
            seg_weight=seg_weight,
        ).to(device)
        losses.append(
            float(
                criterion(outputs, labels, boxes, mask, points=points)[
                    "total_loss"
                ].detach()
            )
        )
    if losses[0] != losses[1]:
        raise AssertionError(f"Baseline seg equivalence failed: {losses}")
    return {"seg_weight_0": losses[0], "seg_weight_0p5": losses[1], "equal": True}


def run_smoke(
    experiments: Iterable[AblationExperiment],
    *,
    device_name: str,
    batch_size: int,
    num_points: int,
) -> Path:
    """运行并保存 smoke 结果。"""
    if batch_size <= 0 or num_points < 2:
        raise ValueError("batch_size must be positive and num_points >= 2")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(device_name)

    selected = unique_experiments(experiments)
    results = [
        run_one(
            experiment,
            device=device,
            batch_size=batch_size,
            num_points=num_points,
        )
        for experiment in selected
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "batch_size": batch_size,
        "num_points": num_points,
        "baseline_seg_equivalence": verify_baseline_seg_equivalence(
            device=device,
            batch_size=batch_size,
            num_points=num_points,
        ),
        "results": results,
    }
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = SMOKE_ROOT / f"ablation_smoke_{timestamp}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        print(
            result["experiment_id"],
            f"input={tuple(result['input_shape'])}",
            f"box={tuple(result['box_shape'])}",
            f"seg={result['seg_shape']}",
            f"peak_vram_mb={result['peak_vram_mb']:.1f}",
        )
    print(f"output={output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI。"""
    parser = argparse.ArgumentParser(description="SPAD 消融单批次 smoke")
    parser.add_argument(
        "--families",
        type=parse_csv,
        default=["core"],
        help="逗号分隔：core,robustness,structure_core,structure_appendix,operator,lambda；structure 为兼容别名；默认 core",
    )
    parser.add_argument("--experiments", type=parse_csv, default=None, help="逗号分隔实验 ID")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=32)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    experiments = select_experiments(args.families, args.experiments)
    run_smoke(
        experiments,
        device_name=args.device,
        batch_size=args.batch_size,
        num_points=args.num_points,
    )
    return 0


def main_without_cli() -> None:
    """无参数模式：CPU core smoke。"""
    run_smoke(
        select_experiments(["core"]),
        device_name="cpu",
        batch_size=2,
        num_points=32,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
