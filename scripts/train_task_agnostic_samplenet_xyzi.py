"""训练、恢复和导出强度感知任务无关 SampleNet-XYZI。

CLI 示例：
    python scripts/train_task_agnostic_samplenet_xyzi.py --mode sanity --device cpu
    python scripts/train_task_agnostic_samplenet_xyzi.py --mode train --epochs 100 --device cuda
    python scripts/train_task_agnostic_samplenet_xyzi.py --mode train --resume outputs/samplenet_xyzi_task_agnostic/<run>/checkpoints/last.pth --epochs 150
    python scripts/train_task_agnostic_samplenet_xyzi.py --mode export --checkpoint outputs/samplenet_xyzi_task_agnostic/<run>/checkpoints/best.pth --export-dir D:/exports/samplenet_xyzi

无参运行：
    python scripts/train_task_agnostic_samplenet_xyzi.py
    只对 1 个正式文件执行前向、反向、梯度、唯一索引、checkpoint 恢复和
    TXT 导出检查，不启动全量长期训练。

输入输出契约：
    输入为严格正式文件名的 XYZI 文本。每个文件无放回选择 8192 个候选行，
    模型输入为 ``(B, 8192, 4)``，输出 1024 点。训练只用可微的
    ``generated_points``、``projected_points`` 和现有辅助损失；导出只用
    ``output.points``，结果仍为原文件中的真实 XYZI 行。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downsampling import SampleNetXYZI, assert_unique_indices
from downsampling.common import gather_points
from downsampling.task_agnostic_xyzi import (
    FormalXYZICandidateDataset,
    scan_formal_point_files,
    split_formal_files,
    task_agnostic_samplenet_loss,
)


DEFAULT_DATA_ROOT = Path(
    r"D:\PYproject\SPADdata\20250430\2025-04-30-pc"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "samplenet_xyzi_task_agnostic"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TaskAgnosticConfig:
    """训练、检查和导出的统一配置。"""

    mode: str = "train"
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_dir: Optional[Path] = None
    export_dir: Optional[Path] = None
    checkpoint: Optional[Path] = None
    resume: Optional[Path] = None
    candidate_points: int = 8192
    num_samples: int = 1024
    projection_neighbors: int = 8
    batch_size: int = 8
    epochs: int = 100
    feature_dim: int = 256
    hidden_dim: int = 512
    initial_temperature: float = 1.0
    min_temperature: float = 0.01
    coverage_weight: float = 1.0
    distance_chunk_size: int = 256
    intensity_chunk_size: int = 512
    geometry_weight: float = 1.0
    intensity_weight: float = 1.0
    projection_weight: float = 1.0
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    max_files: int = 0
    overwrite: bool = False
    skip_existing: bool = False


def set_seed(seed: int) -> None:
    """固定训练涉及的随机源。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """解析 cpu/cuda/auto，并在显式 CUDA 不可用时失败。"""

    normalized = device_name.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"device 必须为 auto/cpu/cuda，实际为 {device_name!r}"
        )
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    return torch.device(normalized)


def validate_config(config: TaskAgnosticConfig) -> None:
    """在读取数据或启动训练前校验交叉参数。"""

    if config.mode not in {"train", "export", "sanity"}:
        raise ValueError(f"不支持 mode={config.mode!r}")
    if config.candidate_points <= 0 or config.num_samples <= 0:
        raise ValueError("candidate_points 和 num_samples 必须为正整数")
    if config.num_samples > config.candidate_points:
        raise ValueError("num_samples 不能大于 candidate_points")
    if config.projection_neighbors <= 0:
        raise ValueError("projection_neighbors 必须为正整数")
    if config.projection_neighbors > config.candidate_points:
        raise ValueError("projection_neighbors 不能大于 candidate_points")
    if config.batch_size <= 0 or config.epochs <= 0:
        raise ValueError("batch_size 和 epochs 必须为正整数")
    if config.feature_dim <= 0 or config.hidden_dim <= 0:
        raise ValueError("feature_dim 和 hidden_dim 必须为正整数")
    if config.distance_chunk_size <= 0 or config.intensity_chunk_size <= 0:
        raise ValueError("两个距离分块大小必须为正整数")
    if config.learning_rate <= 0 or config.min_learning_rate < 0:
        raise ValueError("学习率范围不合法")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate 不能大于 learning_rate")
    if config.weight_decay < 0:
        raise ValueError("weight_decay 不能为负数")
    if config.max_files < 0 or config.num_workers < 0:
        raise ValueError("max_files 和 num_workers 不能为负数")
    if config.overwrite and config.skip_existing:
        raise ValueError("overwrite 与 skip_existing 不能同时启用")
    if config.mode == "export" and config.checkpoint is None:
        raise ValueError("export 模式必须提供 --checkpoint")
    if config.mode != "export" and config.checkpoint is not None:
        raise ValueError("--checkpoint 只用于 export；恢复训练请使用 --resume")
    if config.mode == "export" and config.resume is not None:
        raise ValueError("export 模式不能使用 --resume")
    if config.mode == "train" and not 0.0 < config.val_ratio < 1.0:
        raise ValueError("train 模式的 val_ratio 必须位于 (0, 1)")
    for name, weight in (
        ("geometry_weight", config.geometry_weight),
        ("intensity_weight", config.intensity_weight),
        ("projection_weight", config.projection_weight),
    ):
        if weight < 0:
            raise ValueError(f"{name} 不能为负数")


def model_config_from_run_config(config: TaskAgnosticConfig) -> Dict[str, Any]:
    """提取构造 SampleNetXYZI 所需参数。"""

    return {
        "num_samples": config.num_samples,
        "projection_neighbors": config.projection_neighbors,
        "feature_dim": config.feature_dim,
        "hidden_dim": config.hidden_dim,
        "initial_temperature": config.initial_temperature,
        "min_temperature": config.min_temperature,
        "coverage_weight": config.coverage_weight,
        "distance_chunk_size": config.distance_chunk_size,
    }


def build_model(config: TaskAgnosticConfig) -> SampleNetXYZI:
    """复用现有 SampleNetXYZI 构建任务无关采样器。"""

    return SampleNetXYZI(**model_config_from_run_config(config))


def _jsonable_config(config: TaskAgnosticConfig) -> Dict[str, Any]:
    """把含 Path 的 dataclass 转为可写入 JSON 的字典。"""

    result: Dict[str, Any] = {}
    for key, value in asdict(config).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(config: TaskAgnosticConfig) -> Path:
    """解析新训练目录，或恢复 checkpoint 原属目录。"""

    if config.run_dir is not None:
        return config.run_dir.expanduser().resolve()
    if config.resume is not None:
        resume_path = config.resume.expanduser().resolve()
        if resume_path.parent.name == "checkpoints":
            return resume_path.parent.parent
    return (config.output_root.expanduser().resolve() / _timestamp()).resolve()


def create_logger(log_path: Path) -> logging.Logger:
    """创建同时输出到终端和文件的独立日志器。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"task_agnostic_samplenet.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_config(config: TaskAgnosticConfig, path: Path) -> None:
    """保存完整运行配置。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable_config(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_training_checkpoint(
    path: Path,
    model: SampleNetXYZI,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    best_val_loss: float,
    config: TaskAgnosticConfig,
) -> None:
    """原子保存模型、优化器、scheduler、epoch 和完整配置。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "task": "task_agnostic_samplenet_xyzi",
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model_config_from_run_config(config),
        "candidate_points": int(config.candidate_points),
        "loss_config": {
            "geometry_weight": config.geometry_weight,
            "intensity_weight": config.intensity_weight,
            "projection_weight": config.projection_weight,
            "intensity_chunk_size": config.intensity_chunk_size,
        },
        "config": _jsonable_config(config),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint_file(
    checkpoint_path: Path,
    map_location: torch.device | str,
) -> Dict[str, Any]:
    """读取并校验任务无关 SampleNet checkpoint。"""

    resolved_path = checkpoint_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {resolved_path}")
    try:
        loaded = torch.load(
            resolved_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - 兼容旧 PyTorch。
        loaded = torch.load(resolved_path, map_location=map_location)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint 顶层必须为字典: {resolved_path}")
    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "model_config",
        "candidate_points",
    }
    missing = sorted(required_keys.difference(loaded))
    if missing:
        raise ValueError(f"checkpoint 缺少字段 {missing}: {resolved_path}")
    if loaded.get("task") != "task_agnostic_samplenet_xyzi":
        raise ValueError(f"checkpoint task 不匹配: {loaded.get('task')!r}")
    return loaded


def restore_training_checkpoint(
    checkpoint_path: Path,
    model: SampleNetXYZI,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: TaskAgnosticConfig,
    device: torch.device,
) -> Tuple[int, float]:
    """恢复全部训练状态，返回下一 epoch 与历史最佳验证损失。"""

    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    expected_model_config = model_config_from_run_config(config)
    if checkpoint["model_config"] != expected_model_config:
        raise ValueError(
            "恢复参数与 checkpoint 的 model_config 不一致；"
            f"current={expected_model_config}, saved={checkpoint['model_config']}"
        )
    if int(checkpoint["candidate_points"]) != config.candidate_points:
        raise ValueError(
            "candidate_points 与 checkpoint 不一致："
            f"current={config.candidate_points}, "
            f"saved={checkpoint['candidate_points']}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    next_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return next_epoch, best_val_loss


def create_data_loader(
    data_root: Path,
    files: Sequence[Path],
    config: TaskAgnosticConfig,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader[Dict[str, object]]:
    """创建无标签候选点 DataLoader。"""

    dataset = FormalXYZICandidateDataset(
        data_root=data_root,
        files=files,
        candidate_points=config.candidate_points,
        seed=config.seed + seed_offset,
    )
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    loader_options: Dict[str, Any] = {}
    if config.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        **loader_options,
    )


def run_epoch(
    model: SampleNetXYZI,
    loader: DataLoader[Dict[str, object]],
    device: torch.device,
    config: TaskAgnosticConfig,
    optimizer: Optional[Optimizer],
) -> Dict[str, float]:
    """运行一个无标签训练或验证 epoch。"""

    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "total": 0.0,
        "geometry": 0.0,
        "intensity_coverage": 0.0,
        "projection_temperature": 0.0,
    }
    sample_count = 0
    context = nullcontext() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            points_value = batch["points"]
            if not isinstance(points_value, torch.Tensor):
                raise TypeError("DataLoader batch['points'] 必须为 Tensor")
            points = points_value.to(device=device, dtype=torch.float32, non_blocking=True)
            if tuple(points.shape[1:]) != (config.candidate_points, 4):
                raise ValueError(
                    "模型输入必须为 (B, candidate_points, 4)，"
                    f"实际为 {tuple(points.shape)}"
                )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(points)
            losses = task_agnostic_samplenet_loss(
                input_points=points,
                output=output,
                geometry_weight=config.geometry_weight,
                intensity_weight=config.intensity_weight,
                projection_weight=config.projection_weight,
                intensity_chunk_size=config.intensity_chunk_size,
            )
            if optimizer is not None:
                losses["total"].backward()
                optimizer.step()

            batch_size = int(points.shape[0])
            sample_count += batch_size
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch_size

    if sample_count == 0:
        raise RuntimeError("DataLoader 未产生任何 batch")
    return {key: value / sample_count for key, value in totals.items()}


def _checkpoint_model_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """校验并返回 checkpoint 内模型构造参数。"""

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config 必须为字典")
    return dict(model_config)


@torch.inference_mode()
def export_frozen_model(
    model: SampleNetXYZI,
    data_root: Path,
    files: Sequence[Path],
    export_dir: Path,
    candidate_points: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int = 0,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """冻结模型并按输入相对目录批量导出真实 XYZI 硬点。"""

    if overwrite and skip_existing:
        raise ValueError("overwrite 与 skip_existing 不能同时启用")
    root = data_root.expanduser().resolve()
    output_root = export_dir.expanduser().resolve()
    selected_files: List[Path] = []
    skipped_count = 0
    for file_path in files:
        relative_path = file_path.resolve().relative_to(root)
        destination = output_root / relative_path
        if destination.exists():
            if skip_existing:
                skipped_count += 1
                continue
            if not overwrite:
                raise FileExistsError(
                    f"导出文件已存在；使用 --overwrite 或 --skip-existing: {destination}"
                )
        selected_files.append(file_path)

    if not selected_files:
        return {
            "export_dir": str(output_root),
            "exported_files": 0,
            "skipped_files": skipped_count,
        }

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)

    dataset = FormalXYZICandidateDataset(
        data_root=root,
        files=selected_files,
        candidate_points=candidate_points,
        seed=seed,
    )
    loader_options: Dict[str, Any] = {}
    if num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_options,
    )

    exported_count = 0
    for batch in loader:
        points_value = batch["points"]
        source_indices_value = batch["source_indices"]
        relative_paths_value = batch["relative_path"]
        if not isinstance(points_value, torch.Tensor):
            raise TypeError("导出 batch['points'] 必须为 Tensor")
        if not isinstance(source_indices_value, torch.Tensor):
            raise TypeError("导出 batch['source_indices'] 必须为 Tensor")
        if not isinstance(relative_paths_value, (list, tuple)):
            raise TypeError("导出 batch['relative_path'] 必须为路径列表")

        points = points_value.to(device=device, dtype=torch.float32, non_blocking=True)
        source_indices = source_indices_value.to(device=device, dtype=torch.long)
        output = model(points)
        if output.points.shape != (points.shape[0], model.num_samples, 4):
            raise RuntimeError(f"硬输出形状错误: {tuple(output.points.shape)}")
        assert_unique_indices(output.indices)
        expected_points = gather_points(points, output.indices)
        if not torch.equal(output.points, expected_points):
            raise RuntimeError("output.points 不是按 output.indices 提取的原始候选行")
        raw_indices = torch.gather(source_indices, dim=1, index=output.indices)
        assert_unique_indices(raw_indices)

        sampled_batch = output.points.cpu().numpy()
        for batch_index, relative_path_text in enumerate(relative_paths_value):
            relative_path = Path(str(relative_path_text))
            destination = output_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                destination,
                sampled_batch[batch_index],
                delimiter=",",
                fmt="%.10g",
            )
            exported_count += 1

    return {
        "export_dir": str(output_root),
        "exported_files": exported_count,
        "skipped_files": skipped_count,
    }


def run_training(config: TaskAgnosticConfig) -> Dict[str, Any]:
    """执行无标签任务无关训练；只有显式 train 模式会调用。"""

    validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    data_root = config.data_root.expanduser().resolve()
    files = scan_formal_point_files(data_root, max_files=config.max_files)
    train_files, val_files = split_formal_files(
        files=files,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    run_dir = resolve_run_dir(config)
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "train.log"
    config_path = run_dir / (
        f"resume_config_{_timestamp()}.json" if config.resume else "config.json"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, config_path)
    logger = create_logger(log_path)
    logger.info("mode=train device=%s", device)
    if device.type == "cuda":
        logger.info("gpu=%s", torch.cuda.get_device_name(device))
    logger.info(
        "files total=%d train=%d val=%d candidate_points=%d num_samples=%d",
        len(files),
        len(train_files),
        len(val_files),
        config.candidate_points,
        config.num_samples,
    )
    logger.info(
        "projection_neighbors=%d batch_size=%d loss_weights=(%.4f, %.4f, %.4f)",
        config.projection_neighbors,
        config.batch_size,
        config.geometry_weight,
        config.intensity_weight,
        config.projection_weight,
    )

    train_loader = create_data_loader(
        data_root, train_files, config, shuffle=True, seed_offset=101
    )
    val_loader = create_data_loader(
        data_root, val_files, config, shuffle=False, seed_offset=202
    )
    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
        eta_min=config.min_learning_rate,
    )
    start_epoch = 1
    best_val_loss = float("inf")
    if config.resume is not None:
        start_epoch, best_val_loss = restore_training_checkpoint(
            checkpoint_path=config.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
        )
        logger.info(
            "resumed checkpoint=%s next_epoch=%d best_val_loss=%.8f",
            config.resume.expanduser().resolve(),
            start_epoch,
            best_val_loss,
        )

    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            config=config,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            config=config,
            optimizer=None,
        )
        scheduler.step()
        logger.info(
            "epoch=%d/%d train total=%.8f geometry=%.8f intensity=%.8f temp=%.8f",
            epoch,
            config.epochs,
            train_metrics["total"],
            train_metrics["geometry"],
            train_metrics["intensity_coverage"],
            train_metrics["projection_temperature"],
        )
        logger.info(
            "epoch=%d/%d val total=%.8f geometry=%.8f intensity=%.8f temp=%.8f",
            epoch,
            config.epochs,
            val_metrics["total"],
            val_metrics["geometry"],
            val_metrics["intensity_coverage"],
            val_metrics["projection_temperature"],
        )

        if val_metrics["total"] <= best_val_loss:
            best_val_loss = val_metrics["total"]
            save_training_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )
            logger.info("saved best checkpoint=%s", best_path)
        save_training_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config,
        )
        logger.info("saved last checkpoint=%s", last_path)

    if start_epoch > config.epochs:
        logger.info(
            "checkpoint next_epoch=%d 已超过目标 epochs=%d；未执行训练",
            start_epoch,
            config.epochs,
        )
    return {
        "run_dir": str(run_dir),
        "log_file": str(log_path),
        "config_file": str(config_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "start_epoch": start_epoch,
        "target_epochs": config.epochs,
    }


def run_export(config: TaskAgnosticConfig) -> Dict[str, Any]:
    """从结构化 checkpoint 冻结模型并批量导出 TXT。"""

    validate_config(config)
    if config.checkpoint is None:
        raise ValueError("export 模式缺少 checkpoint")
    set_seed(config.seed)
    device = resolve_device(config.device)
    checkpoint_path = config.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    model = SampleNetXYZI(**_checkpoint_model_config(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    candidate_points = int(checkpoint["candidate_points"])
    data_root = config.data_root.expanduser().resolve()
    files = scan_formal_point_files(data_root, max_files=config.max_files)
    if config.export_dir is not None:
        export_dir = config.export_dir
    else:
        run_name = checkpoint_path.parent.parent.name
        export_dir = config.output_root / f"export_{run_name}_{checkpoint_path.stem}"
    return export_frozen_model(
        model=model,
        data_root=data_root,
        files=files,
        export_dir=export_dir,
        candidate_points=candidate_points,
        batch_size=config.batch_size,
        seed=config.seed,
        device=device,
        num_workers=config.num_workers,
        overwrite=config.overwrite,
        skip_existing=config.skip_existing,
    )


def _assert_required_gradients(model: SampleNetXYZI) -> Dict[str, float]:
    """检查编码器、解码器和温度参数均收到有限非零梯度。"""

    parameters = {
        "encoder": model.encoder[0].weight,
        "decoder": model.decoder[-1].weight,
        "temperature": model._temperature_unconstrained,
    }
    norms: Dict[str, float] = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"sanity 失败：{name} 参数无梯度")
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"sanity 失败：{name} 梯度包含 NaN 或 Inf")
        norm = float(gradient.norm())
        if norm <= 0.0:
            raise RuntimeError(f"sanity 失败：{name} 梯度范数为 0")
        norms[name] = norm
    return norms


def run_sanity(config: TaskAgnosticConfig) -> Dict[str, Any]:
    """用一个正式文件完成端到端短检查，不运行完整 epoch。"""

    validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    data_root = config.data_root.expanduser().resolve()
    files = scan_formal_point_files(
        data_root,
        max_files=config.max_files if config.max_files > 0 else 1,
    )
    sanity_file = files[0]
    dataset = FormalXYZICandidateDataset(
        data_root=data_root,
        files=[sanity_file],
        candidate_points=config.candidate_points,
        seed=config.seed,
    )
    batch = dataset[0]
    points_value = batch["points"]
    if not isinstance(points_value, torch.Tensor):
        raise TypeError("sanity points 必须为 Tensor")
    points = points_value.unsqueeze(0).to(device=device, dtype=torch.float32)

    model = build_model(config).to(device).train()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
        eta_min=config.min_learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    output = model(points)
    if output.generated_points is None or not output.generated_points.requires_grad:
        raise RuntimeError("generated_points 未保持梯度")
    if output.projected_points is None or not output.projected_points.requires_grad:
        raise RuntimeError("projected_points 未保持梯度")
    losses = task_agnostic_samplenet_loss(
        input_points=points,
        output=output,
        geometry_weight=config.geometry_weight,
        intensity_weight=config.intensity_weight,
        projection_weight=config.projection_weight,
        intensity_chunk_size=config.intensity_chunk_size,
    )
    losses["total"].backward()
    gradient_norms = _assert_required_gradients(model)
    assert_unique_indices(output.indices)
    if output.points.shape != (1, config.num_samples, 4):
        raise RuntimeError(f"sanity 硬输出形状错误: {tuple(output.points.shape)}")
    if not torch.equal(output.points, gather_points(points, output.indices)):
        raise RuntimeError("sanity 硬输出不是原始候选行")
    optimizer.step()
    scheduler.step()

    run_dir = resolve_run_dir(config)
    checkpoint_path = run_dir / "checkpoints" / "sanity_last.pth"
    config_path = run_dir / "sanity_config.json"
    save_config(config, config_path)
    save_training_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        best_val_loss=float(losses["total"].detach()),
        config=config,
    )

    restored_model = build_model(config).to(device)
    restored_optimizer = AdamW(
        restored_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    restored_scheduler = CosineAnnealingLR(
        restored_optimizer,
        T_max=max(1, config.epochs),
        eta_min=config.min_learning_rate,
    )
    next_epoch, restored_best_loss = restore_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        config=config,
        device=device,
    )
    if next_epoch != 2:
        raise RuntimeError(f"checkpoint 恢复 epoch 错误: {next_epoch}")

    export_summary = export_frozen_model(
        model=restored_model,
        data_root=data_root,
        files=[sanity_file],
        export_dir=run_dir / "export",
        candidate_points=config.candidate_points,
        batch_size=1,
        seed=config.seed,
        device=device,
        num_workers=0,
        overwrite=True,
    )
    return {
        "mode": "sanity",
        "device": str(device),
        "input_file": str(sanity_file),
        "input_shape": tuple(points.shape),
        "output_shape": tuple(output.points.shape),
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "gradient_norms": gradient_norms,
        "unique_indices": True,
        "checkpoint": str(checkpoint_path),
        "restored_next_epoch": next_epoch,
        "restored_best_loss": restored_best_loss,
        "export": export_summary,
    }


def run_with_config(config: TaskAgnosticConfig) -> Dict[str, Any]:
    """按配置分派训练、导出或短检查。"""

    if config.mode == "train":
        return run_training(config)
    if config.mode == "export":
        return run_export(config)
    if config.mode == "sanity":
        return run_sanity(config)
    raise ValueError(f"不支持 mode={config.mode!r}")


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(
        description="强度感知任务无关 SampleNet-XYZI 训练、恢复与 TXT 导出"
    )
    parser.add_argument(
        "--mode",
        choices=("train", "export", "sanity"),
        default="train",
        help="train=完整训练，export=冻结 checkpoint 批量导出，sanity=单文件短检查",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--candidate-points", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--projection-neighbors", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.01)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    parser.add_argument("--intensity-chunk-size", type=int, default=512)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--intensity-weight", type=float, default=1.0)
    parser.add_argument("--projection-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="大于 0 时只使用严格扫描结果中的前若干文件；短测试推荐 1 或 2",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def config_from_namespace(args: argparse.Namespace) -> TaskAgnosticConfig:
    """把 argparse 结果转换为显式类型配置。"""

    return TaskAgnosticConfig(
        mode=args.mode,
        data_root=args.data_root,
        output_root=args.output_root,
        run_dir=args.run_dir,
        export_dir=args.export_dir,
        checkpoint=args.checkpoint,
        resume=args.resume,
        candidate_points=args.candidate_points,
        num_samples=args.num_samples,
        projection_neighbors=args.projection_neighbors,
        batch_size=args.batch_size,
        epochs=args.epochs,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        initial_temperature=args.initial_temperature,
        min_temperature=args.min_temperature,
        coverage_weight=args.coverage_weight,
        distance_chunk_size=args.distance_chunk_size,
        intensity_chunk_size=args.intensity_chunk_size,
        geometry_weight=args.geometry_weight,
        intensity_weight=args.intensity_weight,
        projection_weight=args.projection_weight,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        max_files=args.max_files,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""

    args = build_parser().parse_args(argv)
    summary = run_with_config(config_from_namespace(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def main_without_cli() -> None:
    """IDE 无参数调试入口；固定执行单文件正式数据 sanity。"""

    # ===== 可直接在 IDE 中修改的参数 =====
    data_root = DEFAULT_DATA_ROOT
    output_root = DEFAULT_OUTPUT_ROOT
    device_name = "auto"

    # ===== 中间变量与安全短流程配置 =====
    resolved_data_root = Path(data_root)
    resolved_output_root = Path(output_root)
    config = TaskAgnosticConfig(
        mode="sanity",
        data_root=resolved_data_root,
        output_root=resolved_output_root,
        run_dir=resolved_output_root / f"sanity_{_timestamp()}",
        device=device_name,
        candidate_points=8192,
        num_samples=1024,
        projection_neighbors=8,
        batch_size=1,
        max_files=1,
        epochs=2,
    )
    summary = run_with_config(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    # 有 CLI 参数时按显式 mode 执行；无参数始终走单文件 sanity，不启动全量训练。
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
