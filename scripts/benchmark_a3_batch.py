"""A3 配置单批次显存/耗时基准，不启动正式训练。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\benchmark_a3_batch.py --execute --batch-sizes 8,16,32,48,64 --checkpoint both

默认仅打印计划。执行模式在 CUDA 上为每个 batch size 重建模型，运行少量
forward/backward/optimizer step，报告峰值 allocated VRAM 和每步耗时；遇到 OOM 后清缓存并
继续/停止。它只能用于选择安全物理 batch，不能替代正式训练或改变 A0--A3 冻结协议。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

import torch

from scripts.ablation_registry import FIXED_PYTHON, PROJECT_ROOT
from scripts.train import build_model, set_seed
from utils.loss import PointCloudMultiTaskLoss


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ABL" / "batch_benchmark"


def parse_batch_sizes(value: str) -> List[int]:
    """解析正整数 batch 列表。"""
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def model_args(*, use_checkpoint: bool, num_points: int) -> argparse.Namespace:
    """构建冻结 A3 模型参数。"""
    return argparse.Namespace(
        model="graph_residual_gcn_ablation",
        box_head="centroid",
        num_points=num_points,
        gcn_k=None,
        gcn_use_checkpoint=use_checkpoint,
        gcn_aggregation="max",
        gcn_exclude_self=True,
        gcn_feature_residual=True,
        gcn_coord_scale_init=0.1,
        gcn_legacy_mode=False,
        gcn_use_physical_branch=True,
        gcn_use_se_gate=True,
        gcn_use_coord_residual=True,
    )


def synthetic_targets(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成合法分类与归一化固定框目标。"""
    labels = torch.arange(batch_size, device=device) % 26
    boxes = torch.tensor(
        [[0.35, 0.65, 0.35, 0.65, 0.72, 0.78]],
        device=device,
        dtype=dtype,
    ).repeat(batch_size, 1)
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    return labels, boxes, valid


def benchmark_one(
    *,
    batch_size: int,
    num_points: int,
    use_checkpoint: bool,
    steps: int,
) -> Dict[str, object]:
    """运行一个物理 batch 配置。"""
    device = torch.device("cuda")
    set_seed(20260717)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(
        "graph_residual_gcn_ablation",
        num_classes=26,
        project_root=PROJECT_ROOT,
        args=model_args(use_checkpoint=use_checkpoint, num_points=num_points),
    ).to(device)
    criterion = PointCloudMultiTaskLoss(
        cls_weight=1.0,
        box_weight=10.0,
        label_smoothing=0.1,
        auto_balance=False,
        seg_weight=0.5,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    points = torch.rand(batch_size, num_points, 4, device=device)
    labels, boxes, valid = synthetic_targets(batch_size, device, points.dtype)

    elapsed: List[float] = []
    model.train()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        outputs = model(points)
        losses = criterion(outputs, labels, boxes, valid, points=points)
        losses["total_loss"].backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if step > 0:  # 首步作为 CUDA/cuDNN 暖机
            elapsed.append(time.perf_counter() - start)

    result = {
        "batch_size": batch_size,
        "num_points": num_points,
        "use_checkpoint": use_checkpoint,
        "steps": steps,
        "mean_step_seconds": sum(elapsed) / len(elapsed),
        "min_step_seconds": min(elapsed),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "status": "ok",
    }
    del model, criterion, optimizer, points, labels, boxes, valid, outputs, losses
    torch.cuda.empty_cache()
    return result


def run_benchmark(
    *,
    execute: bool,
    batch_sizes: Sequence[int],
    num_points: int,
    checkpoint_mode: str,
    steps: int,
) -> int:
    """执行或打印基准计划。"""
    if Path(sys.executable).resolve() != FIXED_PYTHON.resolve():
        raise RuntimeError(f"Expected fixed Python {FIXED_PYTHON}, got {sys.executable}")
    modes = [True, False] if checkpoint_mode == "both" else [checkpoint_mode == "on"]
    for use_checkpoint in modes:
        for batch_size in batch_sizes:
            print(
                "DRY-RUN",
                f"batch={batch_size}",
                f"points={num_points}",
                f"checkpoint={use_checkpoint}",
                f"steps={steps}",
            )
    if not execute:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the fixed environment")
    if steps <= 0 or num_points < 2:
        raise ValueError("steps must be positive and num_points >= 2")

    results: List[Dict[str, object]] = []
    for use_checkpoint in modes:
        for batch_size in batch_sizes:
            try:
                result = benchmark_one(
                    batch_size=batch_size,
                    num_points=num_points,
                    use_checkpoint=use_checkpoint,
                    steps=steps,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                result = {
                    "batch_size": batch_size,
                    "num_points": num_points,
                    "use_checkpoint": use_checkpoint,
                    "status": "oom",
                    "error": str(exc),
                }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"a3_batch_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "python": sys.executable,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output={output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI。"""
    parser = argparse.ArgumentParser(description="A3 单批次显存和速度基准")
    parser.add_argument("--execute", action="store_true", help="实际运行 CUDA 基准；省略时 dry-run")
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[8, 16, 32, 48, 64])
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--checkpoint", choices=["on", "off", "both"], default="both")
    parser.add_argument("--steps", type=int, default=3, help="计时 step 数；另有一个暖机 step")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    return run_benchmark(
        execute=args.execute,
        batch_sizes=args.batch_sizes,
        num_points=args.num_points,
        checkpoint_mode=args.checkpoint,
        steps=args.steps,
    )


def main_without_cli() -> None:
    """无参数模式：安全 dry-run。"""
    run_benchmark(
        execute=False,
        batch_sizes=[8, 16, 32, 48, 64],
        num_points=1024,
        checkpoint_mode="both",
        steps=3,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
