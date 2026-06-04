"""SPAD 5 折交叉验证训练脚本。

CLI example:
    python scripts/train_kfold.py --model graph_residual_gcn --folds 5 --epochs 100 --batch-size 32

Non-CLI example:
    python scripts/train_kfold.py
    # 默认执行 dry-run，只打印 5 折划分计划；在 main_without_cli() 中把 dry_run 改为 False 可启动训练。

Parameter notes:
    --folds 默认 5；--folds-to-run 支持 all、单折 "3" 或范围/列表 "1,3-5"。
    --train-ratio/--val-ratio/--test-ratio 在本脚本中不参与划分，验证集由当前 fold 决定。
    普通训练超参、模型名、loss 权重和 best-score 权重复用 scripts/train.py 的同名参数。

Input/output contract:
    输入数据目录沿用 utils.data 的 SPAD 多任务数据格式；每个样本输出点云 (N,4) 与单目标 bbox。
    输出包含 logs/train_<model>_<timestamp>_foldXX.log、每折 checkpoint，以及
    logs/kfold_<model>_<timestamp>_summary.json。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train as train_module
from utils.checkpoint import save_checkpoint
from utils.data import (
    SPADMultiTaskDataset,
    build_class_mapping,
    collate_fn,
    discover_spad_classification_samples,
)
from utils.loss import PointCloudMultiTaskLoss


Sample = Dict[str, Optional[str]]


def canonical_sample_path(sample: Mapping[str, Optional[str]]) -> str:
    """返回稳定排序使用的样本路径键。"""
    return Path(str(sample.get("path", ""))).as_posix().lower()


def parse_folds_to_run(text: str, total_folds: int) -> List[int]:
    """解析需要执行的折号，返回 0-based fold index。"""
    normalized = str(text).strip().lower()
    if normalized in {"", "all"}:
        return list(range(total_folds))

    selected: List[int] = []
    for part in normalized.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid fold range: {item}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(item))

    unique_sorted = sorted(set(selected))
    invalid = [fold for fold in unique_sorted if fold < 1 or fold > total_folds]
    if invalid:
        raise ValueError(f"Fold ids must be in [1, {total_folds}], got: {invalid}")
    return [fold - 1 for fold in unique_sorted]


def build_kfold_indices(
    samples: Sequence[Sample],
    folds: int,
    seed: int,
    stratified: bool = True,
) -> List[List[int]]:
    """构建 K 折样本索引。

    stratified=True 时按样本原始 label 做近似分层；每个类别内部先随机打乱，再轮转分配到各折。
    若某些折为空，会从最大折移动样本补齐，保证每折至少有一个验证样本。
    """
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    if len(samples) < folds:
        raise ValueError(f"Need at least {folds} samples for {folds}-fold CV, got {len(samples)}")

    rng = np.random.default_rng(seed)
    fold_indices: List[List[int]] = [[] for _ in range(folds)]

    if stratified:
        label_to_indices: Dict[str, List[int]] = {}
        for index, sample in enumerate(samples):
            label = str(sample.get("label") or "__unlabeled__")
            label_to_indices.setdefault(label, []).append(index)

        for label in sorted(label_to_indices):
            indices = np.array(label_to_indices[label], dtype=np.int64)
            rng.shuffle(indices)
            start_fold = int(rng.integers(0, folds))
            for offset, sample_index in enumerate(indices.tolist()):
                fold_indices[(start_fold + offset) % folds].append(int(sample_index))
    else:
        indices = np.arange(len(samples), dtype=np.int64)
        rng.shuffle(indices)
        for fold_index, split in enumerate(np.array_split(indices, folds)):
            fold_indices[fold_index].extend(int(item) for item in split.tolist())

    for fold_index, indices in enumerate(fold_indices):
        if indices:
            continue
        donor = max(range(folds), key=lambda item: len(fold_indices[item]))
        if len(fold_indices[donor]) <= 1:
            raise ValueError("Unable to rebalance K-fold split without empty validation folds.")
        moved = fold_indices[donor].pop()
        fold_indices[fold_index].append(moved)

    return [sorted(indices) for indices in fold_indices]


def load_cv_samples(data_root: Path, label_mode: str) -> Tuple[List[Sample], Dict[str, int]]:
    """扫描数据集并返回 K 折使用的样本列表与类别映射。"""
    labeled_samples, unlabeled_samples = discover_spad_classification_samples(str(data_root))
    labeled_samples = sorted(labeled_samples, key=canonical_sample_path)
    unlabeled_samples = sorted(unlabeled_samples, key=canonical_sample_path)

    if not labeled_samples:
        raise ValueError(f"No labeled samples found in: {data_root}")

    class_to_idx = build_class_mapping(labeled_samples)
    if label_mode == "raw":
        samples = labeled_samples
    elif label_mode == "generated":
        samples = labeled_samples + unlabeled_samples
    else:
        raise ValueError(f"label_mode must be 'raw' or 'generated', got: {label_mode}")

    if len(samples) < 2:
        raise ValueError(f"Need at least 2 samples for K-fold CV, got {len(samples)}")
    return samples, class_to_idx


def build_loader(
    samples: Sequence[Sample],
    class_to_idx: Dict[str, int],
    args: argparse.Namespace,
    seed: int,
    apply_augment: bool,
    num_aug: int,
    shuffle: bool,
) -> DataLoader:
    """从显式样本列表构建 DataLoader，避免复用普通训练的 split cache。"""
    dataset = SPADMultiTaskDataset(
        samples=list(samples),
        class_to_idx=class_to_idx,
        num_points=args.num_points,
        seed=seed,
        apply_augment=apply_augment,
        num_aug=num_aug,
        label_mode=args.label_mode,
    )

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    loader_extra = {"persistent_workers": True, "prefetch_factor": 2} if args.num_workers > 0 else {}

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        collate_fn=collate_fn,
        **loader_extra,
    )


def resolve_device(device_text: str) -> torch.device:
    """解析训练设备，保持与普通训练脚本一致。"""
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_text == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def close_logger(logger: Any) -> None:
    """关闭单折 logger 的文件句柄，避免连续 5 折持有过多 handler。"""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def train_one_fold(
    args: argparse.Namespace,
    fold_index: int,
    total_folds: int,
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    class_to_idx: Dict[str, int],
    cv_timestamp: str,
    project_root: Path,
) -> Dict[str, Any]:
    """训练单个 fold，并返回该 fold 的最佳验证指标与输出路径。"""
    fold_id = fold_index + 1
    fold_seed = int(args.seed) + fold_index * 1009

    train_module.set_seed(fold_seed)
    train_module.configure_torch_runtime(args.tf32)

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    log_dir = train_module.resolve_path(args.log_dir, project_root)
    save_dir = train_module.resolve_path(args.save_dir, project_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_timestamp = f"{cv_timestamp}_fold{fold_id:02d}"
    logger, log_file, run_timestamp = train_module.setup_logger(
        log_dir=log_dir,
        model_name=args.model,
        timestamp=fold_timestamp,
    )

    fold_args = argparse.Namespace(**vars(args))
    fold_args.resume = ""
    fold_args.cv_fold = fold_id
    fold_args.cv_total_folds = total_folds
    fold_args.cv_timestamp = cv_timestamp

    try:
        train_loader = build_loader(
            samples=train_samples,
            class_to_idx=class_to_idx,
            args=args,
            seed=fold_seed + 11,
            apply_augment=True,
            num_aug=args.num_aug,
            shuffle=True,
        )
        val_loader = build_loader(
            samples=val_samples,
            class_to_idx=class_to_idx,
            args=args,
            seed=fold_seed + 29,
            apply_augment=args.augment_eval,
            num_aug=1,
            shuffle=False,
        )

        num_classes = len(class_to_idx)
        model = train_module.build_model(
            args.model,
            num_classes=num_classes,
            project_root=project_root,
            args=args,
        ).to(device)
        criterion = PointCloudMultiTaskLoss(
            cls_weight=args.cls_loss_weight,
            box_weight=args.box_loss_weight,
            label_smoothing=args.label_smoothing,
            auto_balance=args.auto_balance,
        )
        optimizer = optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_ckpt = save_dir / f"{args.model}_{run_timestamp}_best.pth"
        last_ckpt = save_dir / f"{args.model}_{run_timestamp}_last.pth"
        best_val_score = float("-inf")
        best_val_top1 = 0.0
        best_epoch = 0
        best_val_metrics: Dict[str, float] = {}
        score_config = train_module.score_config_from_args(args)

        logger.info("=== K-Fold Training Configuration ===")
        logger.info("cv_timestamp=%s", cv_timestamp)
        logger.info("fold=%d/%d", fold_id, total_folds)
        logger.info("data_root=%s", train_module.resolve_path(args.data_root, project_root))
        logger.info("device=%s", device)
        logger.info("model=%s", args.model)
        logger.info("num_classes=%d", num_classes)
        logger.info("split train/val = %d / %d", len(train_loader.dataset), len(val_loader.dataset))
        logger.info("label_mode=%s", args.label_mode)
        logger.info("augment_train=%s augment_eval=%s num_aug=%d", True, args.augment_eval, args.num_aug)
        logger.info("amp=%s tf32=%s", use_amp, args.tf32)
        logger.info("loss_auto_balance=%s", args.auto_balance)
        logger.info(
            "best_score_weights cls=%.4f iou=%.4f depth=%.4f depth_scale=%.6f",
            args.best_score_cls_weight,
            args.best_score_iou_weight,
            args.best_score_depth_weight,
            args.best_score_depth_scale,
        )
        logger.info("args=%s", json.dumps(vars(fold_args), ensure_ascii=False))

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_module.run_epoch(
                loader=train_loader,
                model=model,
                criterion=criterion,
                device=device,
                epoch=epoch,
                phase=f"Fold{fold_id:02d} Train",
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
            )
            val_metrics = train_module.run_epoch(
                loader=val_loader,
                model=model,
                criterion=criterion,
                device=device,
                epoch=epoch,
                phase=f"Fold{fold_id:02d} Val",
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
            )
            scheduler.step()

            logger.info(
                "Fold [%d/%d] Epoch [%d/%d] | train_loss=%.4f train_top1=%.4f train_top3=%.4f | "
                "val_loss=%.4f val_top1=%.4f val_top3=%.4f",
                fold_id,
                total_folds,
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["top1"],
                train_metrics["top3"],
                val_metrics["loss"],
                val_metrics["top1"],
                val_metrics["top3"],
            )
            if train_metrics["box_samples"] > 0 or val_metrics["box_samples"] > 0:
                logger.info(
                    "Fold [%d/%d] Epoch [%d/%d] | train_box_iou=%.4f train_box_depth=%.4f | "
                    "val_box_iou=%.4f val_box_depth=%.4f",
                    fold_id,
                    total_folds,
                    epoch,
                    args.epochs,
                    train_metrics["box_iou"],
                    train_metrics["box_depth"],
                    val_metrics["box_iou"],
                    val_metrics["box_depth"],
                )

            val_score, score_components, score_weights = train_module.compute_composite_score(val_metrics, args)
            logger.info(
                "Fold [%d/%d] Epoch [%d/%d] | val_score=%.4f | components cls=%.4f iou=%.4f depth=%.4f | "
                "weights cls=%.3f iou=%.3f depth=%.3f",
                fold_id,
                total_folds,
                epoch,
                args.epochs,
                val_score,
                score_components.get("cls_top1", 0.0),
                score_components.get("box_iou", 0.0),
                score_components.get("box_depth", 0.0),
                score_weights.get("cls_top1", 0.0),
                score_weights.get("box_iou", 0.0),
                score_weights.get("box_depth", 0.0),
            )

            if val_score >= best_val_score:
                best_val_score = val_score
                best_val_top1 = val_metrics["top1"]
                best_epoch = epoch
                best_val_metrics = {
                    key: float(value)
                    for key, value in val_metrics.items()
                    if isinstance(value, (int, float))
                }
                best_val_metrics["score"] = best_val_score
                best_val_metrics["epoch"] = float(best_epoch)
                save_checkpoint(
                    path=best_ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_top1=best_val_top1,
                    class_to_idx=class_to_idx,
                    args=fold_args,
                    criterion=criterion,
                    best_val_score=best_val_score,
                    best_val_metrics=best_val_metrics,
                    score_config=score_config,
                )
                logger.info("Saved new best checkpoint to %s (score=%.4f)", best_ckpt, best_val_score)

            save_checkpoint(
                path=last_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_top1=best_val_top1,
                class_to_idx=class_to_idx,
                args=fold_args,
                criterion=criterion,
                best_val_score=best_val_score,
                best_val_metrics=best_val_metrics,
                score_config=score_config,
            )
            logger.info("Saved last checkpoint to %s", last_ckpt)

        logger.info(
            "Fold finished. fold=%d/%d best_epoch=%d best_val_score=%.4f best_val_top1=%.4f",
            fold_id,
            total_folds,
            best_epoch,
            best_val_score,
            best_val_top1,
        )

        return {
            "fold": fold_id,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "best_val_top1": best_val_top1,
            "best_val_metrics": best_val_metrics,
            "log_file": str(log_file),
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
        }
    finally:
        close_logger(logger)


def summarize_results(fold_results: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """计算折间均值和标准差。"""
    key_getters = {
        "best_val_score": lambda item: item.get("best_val_score"),
        "best_val_top1": lambda item: item.get("best_val_top1"),
        "best_val_top3": lambda item: item.get("best_val_metrics", {}).get("top3"),
        "best_val_loss": lambda item: item.get("best_val_metrics", {}).get("loss"),
        "best_val_box_iou": lambda item: item.get("best_val_metrics", {}).get("box_iou"),
        "best_val_box_depth": lambda item: item.get("best_val_metrics", {}).get("box_depth"),
    }
    summary: Dict[str, Dict[str, float]] = {}
    for key, getter in key_getters.items():
        values = [
            float(value)
            for item in fold_results
            for value in [getter(item)]
            if isinstance(value, (int, float)) and np.isfinite(value)
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def run_kfold_training(args: argparse.Namespace) -> Dict[str, Any]:
    """执行 K 折交叉验证训练。"""
    project_root = Path(__file__).resolve().parents[1]
    data_root = train_module.resolve_path(args.data_root, project_root)
    log_dir = train_module.resolve_path(args.log_dir, project_root)

    if args.resume:
        raise ValueError("K-fold training does not support --resume. Use --folds-to-run to rerun selected folds.")
    if not args.augment_train:
        raise ValueError("K-fold training requires --augment-train for consistency with scripts/train.py.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    samples, class_to_idx = load_cv_samples(data_root, args.label_mode)
    fold_indices = build_kfold_indices(
        samples=samples,
        folds=args.folds,
        seed=args.seed,
        stratified=args.stratified,
    )
    folds_to_run = parse_folds_to_run(args.folds_to_run, args.folds)
    cv_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    plan = []
    for fold_index, val_indices in enumerate(fold_indices):
        val_set = set(val_indices)
        train_count = len(samples) - len(val_indices)
        plan.append({
            "fold": fold_index + 1,
            "train_samples": train_count,
            "val_samples": len(val_indices),
        })

    if args.dry_run:
        print("K-fold dry run:")
        print(f"  data_root: {data_root}")
        print(f"  model: {args.model}")
        print(f"  folds: {args.folds}")
        print(f"  folds_to_run: {[fold + 1 for fold in folds_to_run]}")
        print(f"  num_samples: {len(samples)}")
        print(f"  num_classes: {len(class_to_idx)}")
        print(f"  stratified: {args.stratified}")
        for item in plan:
            print(f"  fold {item['fold']}: train={item['train_samples']} val={item['val_samples']}")
        return {
            "dry_run": True,
            "plan": plan,
            "class_to_idx": class_to_idx,
        }

    fold_results: List[Dict[str, Any]] = []
    for fold_index in folds_to_run:
        val_indices = fold_indices[fold_index]
        val_set = set(val_indices)
        train_indices = [index for index in range(len(samples)) if index not in val_set]
        train_samples = [samples[index] for index in train_indices]
        val_samples = [samples[index] for index in val_indices]

        fold_result = train_one_fold(
            args=args,
            fold_index=fold_index,
            total_folds=args.folds,
            train_samples=train_samples,
            val_samples=val_samples,
            class_to_idx=class_to_idx,
            cv_timestamp=cv_timestamp,
            project_root=project_root,
        )
        fold_results.append(fold_result)

    summary = {
        "cv_timestamp": cv_timestamp,
        "model": args.model,
        "folds": args.folds,
        "folds_to_run": [fold + 1 for fold in folds_to_run],
        "num_samples": len(samples),
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "args": vars(args),
        "fold_plan": plan,
        "fold_results": fold_results,
        "summary": summarize_results(fold_results),
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / f"kfold_{args.model}_{cv_timestamp}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"K-fold summary saved to: {summary_path}")
    return {**summary, "summary_path": str(summary_path)}


def build_parser() -> argparse.ArgumentParser:
    """构建 K 折训练命令行参数。"""
    parser = train_module.build_parser()
    parser.description = "SPAD 3D point cloud K-fold cross-validation training"
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--folds-to-run", type=str, default="all", help="Fold ids to run: all, 3, or 1,3-5")
    parser.add_argument("--stratified", dest="stratified", action="store_true", help="Use approximate label-stratified folds")
    parser.add_argument("--no-stratified", dest="stratified", action="store_false", help="Use plain shuffled K-fold split")
    parser.add_argument("--dry-run", action="store_true", help="Print fold plan and exit without training")
    parser.set_defaults(stratified=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_kfold_training(args)
    return 0


def main_without_cli() -> None:
    """非 CLI 调试入口；默认只打印 5 折计划。"""
    # ===== Editable parameters =====
    data_root = Path(r"D:\PYproject\SPADdata\2025-04-30-dpc")
    model = "graph_residual_gcn"
    folds = 5
    epochs = 100
    batch_size = 32
    dry_run = True

    # ===== Intermediate variables =====
    argv = [
        "--data-root", str(data_root),
        "--model", model,
        "--folds", str(folds),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    if dry_run:
        argv.append("--dry-run")

    parser = build_parser()
    args = parser.parse_args(argv)
    run_kfold_training(args)


if __name__ == "__main__":
    # Usage examples:
    #   python scripts/train_kfold.py
    #       执行 main_without_cli()，默认 dry-run，只打印 5 折划分计划。
    #   python scripts/train_kfold.py --model graph_residual_gcn --folds 5 --epochs 100 --batch-size 32
    #       启动完整 5 折交叉验证训练。
    #   python scripts/train_kfold.py --model graph_residual_gcn --folds 5 --folds-to-run 3 --epochs 100
    #       只训练第 3 折，适合中断后手动补跑某一折。
    #
    # Common parameters:
    #   --folds 5                 K 折数量，默认 5。
    #   --folds-to-run all        可选 all、单折或范围/列表，例如 1,3-5。
    #   --stratified              默认按原始 label 近似分层划分。
    #   --dry-run                 只打印折划分，不启动训练。
    #   --box-loss-weight 10.0    默认复用 train.py 的 box/depth 加权配置。
    #
    # Outputs:
    #   logs/train_<model>_<timestamp>_foldXX.log
    #   checkpoints/<model>_<timestamp>_foldXX_best.pth
    #   checkpoints/<model>_<timestamp>_foldXX_last.pth
    #   logs/kfold_<model>_<timestamp>_summary.json
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
