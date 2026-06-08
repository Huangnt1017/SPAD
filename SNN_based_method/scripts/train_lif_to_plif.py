"""LIF 到 PLIF 模型的二阶段训练入口。

CLI example:
    python SNN_based_method/scripts/train_lif_to_plif.py --lif-epochs 15 --plif-epochs 5

Non-CLI example:
    python SNN_based_method/scripts/train_lif_to_plif.py

参数说明:
    普通网络结构、数据、loss 和运行时参数默认由 SNN_config.py 中的 SNNConfig 决定。
    本脚本第一阶段强制构建 spike_mode=lif, 默认训练 15 个 epoch。
    第二阶段加载第一阶段 LIF last.pth, 强制构建 spike_mode=plif, 默认微调 5 个 epoch。
    可传 --lif-checkpoint 跳过第一阶段, 直接从已有 LIF checkpoint 进入 PLIF 微调。
    PLIF 阶段只迁移模型权重, 不恢复 LIF optimizer/scheduler。

输入/输出:
    输入模型数据仍为 [B, 4096, P] raw ToF, 由现有 dataloader 和 train.py 训练循环准备。
    输出 checkpoint 保存到 checkpoint_dir/lif_to_plif_<timestamp>/ 下,
    其中 lif_stage/ 保存 LIF checkpoint, plif_stage/ 保存 PLIF checkpoint 和迁移报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.config.SNN_config import SNNConfig
from SNN_based_method.scripts.train import (
    _format_values,
    apply_default_train_paths,
    configure_torch_runtime,
    ensure_train_precomputed_labels,
    setup_logger,
    train_one_epoch,
    validate_one_epoch,
)
from SNN_based_method.utils.data import seed_everything
from SNN_based_method.utils.runtime import (
    adapt_state_dict_for_model,
    add_config_arguments,
    build_run_name,
    config_from_checkpoint_and_args,
    resolve_output_root,
    save_checkpoint,
)


@dataclass
class ScriptConfig:
    """CLI 与非 CLI 共享的 LIF->PLIF 二阶段训练配置。"""

    lif_checkpoint: Path | None
    args: argparse.Namespace
    lif_epochs: int = 15
    plif_epochs: int = 5
    dry_run: bool = False
    trace_steps: int = 0
    freeze_plif_epochs: int = 0
    allow_partial_model_load: bool = False


@dataclass
class MigrationReport:
    """记录 LIF checkpoint 迁移到 PLIF 模型的键匹配情况。"""

    source_checkpoint: str
    source_epoch: int | None
    source_spike_mode: str | None
    target_spike_mode: str
    copied_keys: list[str]
    initialized_keys: list[str]
    skipped_keys: list[str]
    unexpected_keys: list[str]

    def summary(self) -> dict[str, Any]:
        """返回适合日志和 checkpoint metrics 保存的摘要。"""
        return {
            "source_checkpoint": self.source_checkpoint,
            "source_epoch": self.source_epoch,
            "source_spike_mode": self.source_spike_mode,
            "target_spike_mode": self.target_spike_mode,
            "copied_count": len(self.copied_keys),
            "initialized_count": len(self.initialized_keys),
            "skipped_count": len(self.skipped_keys),
            "unexpected_count": len(self.unexpected_keys),
            "initialized_keys": self.initialized_keys,
            "skipped_keys": self.skipped_keys,
            "unexpected_keys": self.unexpected_keys,
        }


@dataclass
class StageResult:
    """记录单个训练阶段的主要输出路径和指标。"""

    checkpoint_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path
    log_file: Path
    best_val_loss: float


def build_parser() -> argparse.ArgumentParser:
    """构建 LIF->PLIF 二阶段训练命令行参数。"""
    parser = argparse.ArgumentParser(
        description="先训练 LIF SPAD SNN, 再迁移权重并微调 PLIF SPAD SNN",
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--lif-checkpoint",
        default=None,
        help="已有 LIF checkpoint 路径; 传入时跳过第一阶段 LIF 训练",
    )
    parser.add_argument(
        "--lif-epochs",
        type=int,
        default=15,
        help="第一阶段 LIF 训练 epoch 数, 默认 15",
    )
    parser.add_argument(
        "--plif-epochs",
        type=int,
        default=5,
        help="第二阶段 PLIF 微调 epoch 数, 默认 5",
    )
    parser.add_argument("--save-every", type=int, default=None, help="每 N 个 epoch 额外保存一次 checkpoint")
    parser.add_argument("--trace-steps", type=int, default=0, help="打印前 N 个训练 batch 的阶段耗时")
    parser.add_argument(
        "--freeze-plif-epochs",
        type=int,
        default=0,
        help="前 N 个 fine-tune epoch 冻结 PLIF tau 参数 *.w",
    )
    parser.add_argument(
        "--allow-partial-model-load",
        action="store_true",
        help="允许除 PLIF tau 外的缺失权重保持目标模型初始化; 默认严格报错",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只构建目标配置和模型; 如果传入 --lif-checkpoint 则额外检查 PLIF 迁移情况, 不启动训练",
    )
    return parser


def _resolve_checkpoint(path: str | Path) -> Path:
    """解析并校验 checkpoint 文件路径。"""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_output_root(checkpoint_path)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LIF checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _checkpoint_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """从不同格式的 checkpoint 中取出模型 state_dict。"""
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint does not contain a valid model state_dict")
    return state_dict


def _checkpoint_source_mode(checkpoint: dict[str, Any]) -> str | None:
    """读取源 checkpoint 中记录的 spike_mode。"""
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return None
    mode = config.get("spike_mode")
    return str(mode).lower() if mode is not None else None


def _validate_script_config(script_cfg: ScriptConfig) -> None:
    """校验二阶段脚本专属参数, 避免和通用训练参数含义冲突。"""
    args = script_cfg.args
    if args.checkpoint_path:
        raise ValueError("本脚本中 --checkpoint 保留给普通训练入口使用; 请使用 --lif-checkpoint")
    if args.epochs is not None:
        raise ValueError("本脚本用 --lif-epochs 和 --plif-epochs 控制二阶段 epoch, 请不要传 --epochs")
    if args.spike_mode is not None:
        raise ValueError("本脚本会自动控制 LIF/PLIF 阶段的 spike_mode, 请不要传 --spike-mode")
    if script_cfg.lif_checkpoint is None and script_cfg.lif_epochs <= 0:
        raise ValueError("--lif-epochs must be a positive integer")
    if script_cfg.plif_epochs <= 0:
        raise ValueError("--plif-epochs must be a positive integer")


def _base_config_from_args(args: argparse.Namespace) -> SNNConfig:
    """从 SNNConfig 和命令行通用覆盖项构建基础配置。"""
    config_args = argparse.Namespace(**vars(args))
    config_args.checkpoint_path = None
    cfg = config_from_checkpoint_and_args(config_args)
    cfg = apply_default_train_paths(cfg)
    if args.save_every is not None:
        cfg = cfg.clone_with(save_every=args.save_every)
    return cfg


def _build_lif_config(base_cfg: SNNConfig, *, lif_epochs: int, run_name: str) -> SNNConfig:
    """基于基础配置构建第一阶段 LIF 训练配置。"""
    return base_cfg.clone_with(
        model_backend="new",
        spike_mode="lif",
        epochs=lif_epochs,
        checkpoint_path=None,
        run_name=run_name,
    )


def _plif_tau_from_config(cfg: SNNConfig, args: argparse.Namespace) -> float:
    """解析 PLIF 阶段的初始 tau, 避免从 LIF 配置继承无效值。"""
    plif_tau = float(cfg.spike_tau)
    if args.spike_tau is None and plif_tau <= 1.0:
        plif_tau = 2.0
    if plif_tau <= 1.0:
        raise ValueError("PLIF fine-tune requires spike_tau > 1.0")
    return plif_tau


def _build_plif_config_from_base(
    base_cfg: SNNConfig,
    args: argparse.Namespace,
    *,
    plif_epochs: int,
    run_name: str,
    lif_checkpoint: Path | None = None,
) -> SNNConfig:
    """基于基础配置构建第二阶段 PLIF 配置。"""
    return base_cfg.clone_with(
        model_backend="new",
        spike_mode="plif",
        spike_tau=_plif_tau_from_config(base_cfg, args),
        epochs=plif_epochs,
        checkpoint_path=str(lif_checkpoint) if lif_checkpoint is not None else None,
        run_name=run_name,
    )


def _build_plif_config(
    args: argparse.Namespace,
    lif_checkpoint: Path,
    *,
    plif_epochs: int,
    run_name: str,
) -> SNNConfig:
    """从 LIF checkpoint/config 继承配置, 再强制切换到 PLIF 微调。"""

    config_args = argparse.Namespace(**vars(args))
    config_args.checkpoint_path = str(lif_checkpoint)
    cfg = config_from_checkpoint_and_args(config_args)
    cfg = apply_default_train_paths(cfg)

    cfg = cfg.clone_with(
        model_backend="new",
        spike_mode="plif",
        spike_tau=_plif_tau_from_config(cfg, args),
        epochs=plif_epochs,
        checkpoint_path=str(lif_checkpoint),
        run_name=run_name,
    )
    if args.save_every is not None:
        cfg = cfg.clone_with(save_every=args.save_every)
    return cfg


def _stage_checkpoint_dir(cfg: SNNConfig, overall_run_name: str, stage_dir_name: str) -> Path:
    """返回二阶段 run 下的单个阶段 checkpoint 目录。"""
    checkpoint_dir = resolve_output_root(cfg.checkpoint_dir) / overall_run_name / stage_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def load_lif_weights_into_plif(
    checkpoint_path: Path,
    model: torch.nn.Module,
    *,
    map_location: torch.device | str,
    allow_partial_model_load: bool = False,
) -> tuple[dict[str, Any], MigrationReport]:
    """只迁移模型权重, PLIF 新增 tau 参数保持目标模型初始化。

    LIF 和 PLIF 的同结构主干参数同名同形状, 可以直接复制。PLIF 的 ``*.w``
    参数在 LIF checkpoint 中不存在, 由目标模型根据 ``spike_tau`` 初始化。
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    source_state = _checkpoint_model_state(checkpoint)
    source_state = adapt_state_dict_for_model(source_state, model)
    target_state = model.state_dict()

    load_state: dict[str, torch.Tensor] = {}
    copied_keys: list[str] = []
    initialized_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if isinstance(source_value, torch.Tensor) and source_value.shape == target_value.shape:
            load_state[key] = source_value
            copied_keys.append(key)
            continue

        load_state[key] = target_value
        if key.endswith(".w"):
            initialized_keys.append(key)
        else:
            skipped_keys.append(key)

    unexpected_keys = sorted(key for key in source_state if key not in target_state)
    if skipped_keys and not allow_partial_model_load:
        preview = ", ".join(skipped_keys[:20])
        raise ValueError(
            "LIF checkpoint 与目标 PLIF 模型结构不一致, 除 tau 参数外仍有缺失键: "
            f"{preview}"
        )

    model.load_state_dict(load_state, strict=True)
    report = MigrationReport(
        source_checkpoint=str(checkpoint_path),
        source_epoch=checkpoint.get("epoch") if isinstance(checkpoint.get("epoch"), int) else None,
        source_spike_mode=_checkpoint_source_mode(checkpoint),
        target_spike_mode="plif",
        copied_keys=sorted(copied_keys),
        initialized_keys=sorted(initialized_keys),
        skipped_keys=sorted(skipped_keys),
        unexpected_keys=unexpected_keys,
    )
    return checkpoint, report


def set_plif_tau_trainable(model: torch.nn.Module, trainable: bool) -> list[str]:
    """设置 PLIF tau 参数 ``*.w`` 是否参与训练。"""
    affected: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".w"):
            parameter.requires_grad = trainable
            affected.append(name)
    return affected


def _print_dry_run(
    *,
    lif_cfg: SNNConfig | None,
    plif_cfg: SNNConfig,
    report: MigrationReport | None,
    device: torch.device,
    overall_run_name: str,
) -> None:
    """输出 dry-run 检查结果。"""
    print("=== LIF -> PLIF two-stage dry run ===")
    print(f"device={device}")
    print(f"run_name={overall_run_name}")
    if lif_cfg is not None:
        print(f"lif_epochs={lif_cfg.epochs}")
        print("=== LIF configuration ===")
        print(lif_cfg.summary())
    else:
        print("lif_stage=skipped because --lif-checkpoint was provided")
    print(f"plif_epochs={plif_cfg.epochs}")
    print("=== PLIF configuration ===")
    print(plif_cfg.summary())
    if report is not None:
        print("=== LIF -> PLIF migration ===")
        print(f"source_checkpoint={report.source_checkpoint}")
        print(f"source_epoch={report.source_epoch}")
        print(f"source_spike_mode={report.source_spike_mode}")
        print(f"target_spike_mode={plif_cfg.spike_mode}")
        print(f"target_spike_tau={plif_cfg.spike_tau}")
        print(f"copied_count={len(report.copied_keys)}")
        print(f"initialized_tau_keys={report.initialized_keys}")
        print(f"skipped_keys={report.skipped_keys}")
        print(f"unexpected_keys={report.unexpected_keys}")


def _log_stage_configuration(
    *,
    logger,
    title: str,
    cfg: SNNConfig,
    log_file: Path,
    checkpoint_run_dir: Path,
    label_stats,
    train_loader,
    val_loader,
    test_loader,
    dataset,
    extra: dict[str, Any] | None = None,
) -> None:
    """把阶段配置、数据划分和可选迁移信息写入日志。"""
    logger.info("=== %s ===", title)
    logger.info("run_name=%s", cfg.run_name)
    logger.info("log_file=%s", log_file)
    logger.info("checkpoint_run_dir=%s", checkpoint_run_dir)
    logger.info("data_paths=%s", cfg.data_paths)
    logger.info("csv_paths=%s", cfg.csv_paths)
    if extra:
        for key, value in extra.items():
            logger.info("%s=%s", key, value)
    if label_stats is not None:
        logger.info(
            "precomputed_labels pages_per_group=%d planned=%d generated=%d "
            "skipped_existing=%d roots=%s",
            cfg.pages_per_group,
            label_stats.planned,
            label_stats.generated,
            label_stats.skipped_existing,
            sorted(label_stats.label_roots or set()),
        )
    logger.info(
        "split train/val/test=%d/%d/%d dataset_size=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        len(dataset),
    )
    for line in cfg.summary().splitlines():
        logger.info(line)
    logger.info("config_json=%s", json.dumps(cfg.to_dict(), ensure_ascii=False))


def _train_stage(
    *,
    stage_name: str,
    cfg: SNNConfig,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion: torch.nn.Module,
    metrics,
    device: torch.device,
    checkpoint_run_dir: Path,
    logger,
    trace_steps: int,
    freeze_plif_epochs: int = 0,
    migration_report: MigrationReport | None = None,
) -> StageResult:
    """执行单个 LIF 或 PLIF 训练阶段并保存 checkpoint。"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
    )

    best_val_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tau_trainable = True
        if cfg.spike_mode == "plif":
            tau_trainable = epoch > freeze_plif_epochs
            affected_tau = set_plif_tau_trainable(model, tau_trainable)
            if epoch == 1 or epoch == freeze_plif_epochs + 1:
                logger.info(
                    "PLIF tau trainable=%s affected_count=%d",
                    tau_trainable,
                    len(affected_tau),
                )

        train_loss, train_items = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            epoch,
            trace_steps=max(0, trace_steps),
        )
        val_loss, val_items, val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            metrics,
            device,
            cfg,
            epoch,
        )
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_record = {
            "event": f"{stage_name}_epoch",
            "stage": stage_name,
            "spike_mode": cfg.spike_mode,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_items": train_items,
            "val_items": val_items,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        if cfg.spike_mode == "plif":
            epoch_record["freeze_plif_epochs"] = freeze_plif_epochs
            epoch_record["tau_trainable"] = tau_trainable
        if migration_report is not None:
            epoch_record["source_epoch"] = migration_report.source_epoch
            epoch_record["source_checkpoint"] = migration_report.source_checkpoint
            epoch_record["migration"] = migration_report.summary()

        logger.info(
            "%s Epoch [%d/%d] | train_loss=%.6f val_loss=%.6f best_val_loss=%.6f lr=%.8f",
            stage_name,
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            best_val_loss,
            scheduler.get_last_lr()[0],
        )
        logger.info("%s Epoch [%d/%d] | train_items: %s", stage_name, epoch, cfg.epochs, _format_values(train_items))
        logger.info("%s Epoch [%d/%d] | val_items: %s", stage_name, epoch, cfg.epochs, _format_values(val_items))
        logger.info("%s Epoch [%d/%d] | val_metrics: %s", stage_name, epoch, cfg.epochs, _format_values(val_metrics))

        last_checkpoint = checkpoint_run_dir / "last.pth"
        best_checkpoint = checkpoint_run_dir / "best.pth"
        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            cfg=cfg,
            metrics=epoch_record,
        )
        if is_best:
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
            logger.info(
                "Saved new best checkpoint to %s (val_loss=%.6f)",
                best_checkpoint,
                best_val_loss,
            )
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            epoch_checkpoint = checkpoint_run_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                epoch_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
            logger.info("Saved epoch checkpoint to %s", epoch_checkpoint)
        logger.info("Updated last checkpoint to %s", last_checkpoint)

    logger.info("%s training finished. Best val loss=%.6f", stage_name, best_val_loss)
    return StageResult(
        checkpoint_dir=checkpoint_run_dir,
        last_checkpoint=checkpoint_run_dir / "last.pth",
        best_checkpoint=checkpoint_run_dir / "best.pth",
        log_file=Path(logger.handlers[0].baseFilename) if logger.handlers else Path(),
        best_val_loss=best_val_loss,
    )


def run_with_config(script_cfg: ScriptConfig) -> None:
    """执行 LIF 预训练和 PLIF 微调。"""
    _validate_script_config(script_cfg)
    args = script_cfg.args
    base_cfg = _base_config_from_args(args)
    overall_run_name = build_run_name(base_cfg, "lif_to_plif")
    lif_checkpoint = (
        _resolve_checkpoint(script_cfg.lif_checkpoint)
        if script_cfg.lif_checkpoint is not None
        else None
    )

    lif_cfg = None
    if lif_checkpoint is None:
        lif_cfg = _build_lif_config(
            base_cfg,
            lif_epochs=script_cfg.lif_epochs,
            run_name=overall_run_name,
        )
        plif_cfg = _build_plif_config_from_base(
            base_cfg,
            args,
            plif_epochs=script_cfg.plif_epochs,
            run_name=overall_run_name,
        )
    else:
        plif_cfg = _build_plif_config(
            args,
            lif_checkpoint,
            plif_epochs=script_cfg.plif_epochs,
            run_name=overall_run_name,
        )

    runtime_cfg = lif_cfg if lif_cfg is not None else plif_cfg
    seed_everything(runtime_cfg.seed)
    configure_torch_runtime(runtime_cfg)
    device = runtime_cfg.resolved_device()

    if script_cfg.dry_run:
        migration_report = None
        if lif_checkpoint is not None:
            plif_model = plif_cfg.build_model().to(device)
            _, migration_report = load_lif_weights_into_plif(
                lif_checkpoint,
                plif_model,
                map_location=device,
                allow_partial_model_load=script_cfg.allow_partial_model_load,
            )
        elif lif_cfg is not None:
            lif_cfg.build_model().to(device)
            plif_cfg.build_model().to(device)
        _print_dry_run(
            lif_cfg=lif_cfg,
            plif_cfg=plif_cfg,
            report=migration_report,
            device=device,
            overall_run_name=overall_run_name,
        )
        return

    logger, log_file = setup_logger(runtime_cfg.log_dir, overall_run_name)
    label_stats = ensure_train_precomputed_labels(runtime_cfg)
    train_loader, val_loader, test_loader, dataset = runtime_cfg.build_dataloaders()
    criterion = runtime_cfg.build_loss().to(device)
    metrics = runtime_cfg.build_metrics()

    lif_result = None
    if lif_cfg is not None:
        lif_checkpoint_run_dir = _stage_checkpoint_dir(lif_cfg, overall_run_name, "lif_stage")
        lif_cfg.save(lif_checkpoint_run_dir / "config.json")
        _log_stage_configuration(
            logger=logger,
            title="Stage 1/2 LIF Training Configuration",
            cfg=lif_cfg,
            log_file=log_file,
            checkpoint_run_dir=lif_checkpoint_run_dir,
            label_stats=label_stats,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset=dataset,
            extra={"stage_epochs": lif_cfg.epochs},
        )
        lif_model = lif_cfg.build_model().to(device)
        lif_result = _train_stage(
            stage_name="lif",
            cfg=lif_cfg,
            model=lif_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            metrics=metrics,
            device=device,
            checkpoint_run_dir=lif_checkpoint_run_dir,
            logger=logger,
            trace_steps=script_cfg.trace_steps,
        )
        lif_checkpoint = lif_result.last_checkpoint
        plif_cfg = plif_cfg.clone_with(checkpoint_path=str(lif_checkpoint))
        del lif_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.info("Stage 1/2 LIF training skipped; using lif_checkpoint=%s", lif_checkpoint)

    if lif_checkpoint is None:
        raise RuntimeError("internal error: LIF checkpoint was not produced")

    plif_model = plif_cfg.build_model().to(device)
    _, migration_report = load_lif_weights_into_plif(
        lif_checkpoint,
        plif_model,
        map_location=device,
        allow_partial_model_load=script_cfg.allow_partial_model_load,
    )

    plif_checkpoint_run_dir = _stage_checkpoint_dir(plif_cfg, overall_run_name, "plif_stage")
    plif_cfg.save(plif_checkpoint_run_dir / "config.json")
    with (plif_checkpoint_run_dir / "lif_to_plif_migration.json").open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(migration_report), file_obj, indent=2, ensure_ascii=False)
    _log_stage_configuration(
        logger=logger,
        title="Stage 2/2 PLIF Fine-tune Configuration",
        cfg=plif_cfg,
        log_file=log_file,
        checkpoint_run_dir=plif_checkpoint_run_dir,
        label_stats=label_stats,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=dataset,
        extra={
            "stage_epochs": plif_cfg.epochs,
            "lif_checkpoint": lif_checkpoint,
            "source_epoch": migration_report.source_epoch,
            "source_spike_mode": migration_report.source_spike_mode,
            "migration_summary": json.dumps(migration_report.summary(), ensure_ascii=False),
        },
    )
    plif_result = _train_stage(
        stage_name="plif",
        cfg=plif_cfg,
        model=plif_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        metrics=metrics,
        device=device,
        checkpoint_run_dir=plif_checkpoint_run_dir,
        logger=logger,
        trace_steps=script_cfg.trace_steps,
        freeze_plif_epochs=script_cfg.freeze_plif_epochs,
        migration_report=migration_report,
    )

    run_summary = {
        "run_name": overall_run_name,
        "lif_checkpoint": str(lif_checkpoint),
        "lif_stage_dir": str(lif_result.checkpoint_dir) if lif_result is not None else None,
        "plif_stage_dir": str(plif_result.checkpoint_dir),
        "plif_last_checkpoint": str(plif_result.last_checkpoint),
        "plif_best_checkpoint": str(plif_result.best_checkpoint),
        "lif_best_val_loss": lif_result.best_val_loss if lif_result is not None else None,
        "plif_best_val_loss": plif_result.best_val_loss,
    }
    summary_path = resolve_output_root(runtime_cfg.checkpoint_dir) / overall_run_name / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(run_summary, file_obj, indent=2, ensure_ascii=False)
    logger.info("LIF -> PLIF two-stage training finished.")
    logger.info("run_summary=%s", summary_path)
    logger.info("log_file=%s", log_file)
    logger.info("lif_checkpoint=%s", lif_checkpoint)
    logger.info("plif_checkpoint_run_dir=%s", plif_result.checkpoint_dir)


def _script_config_from_args(args: argparse.Namespace) -> ScriptConfig:
    """把 argparse 结果整理成脚本配置。"""
    return ScriptConfig(
        lif_checkpoint=Path(args.lif_checkpoint) if args.lif_checkpoint else None,
        args=args,
        lif_epochs=max(0, int(args.lif_epochs)),
        plif_epochs=max(0, int(args.plif_epochs)),
        dry_run=bool(args.dry_run),
        trace_steps=max(0, int(args.trace_steps)),
        freeze_plif_epochs=max(0, int(args.freeze_plif_epochs)),
        allow_partial_model_load=bool(args.allow_partial_model_load),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    script_cfg = _script_config_from_args(args)
    run_with_config(script_cfg)
    return 0


def main_without_cli() -> None:
    """无命令行参数运行时的可编辑入口。"""
    # ===== Editable parameters =====
    lif_epochs = 15
    plif_epochs = 5
    dry_run = False

    # ===== Intermediate variables =====
    argv = [
        "--lif-epochs",
        str(lif_epochs),
        "--plif-epochs",
        str(plif_epochs),
    ]
    if dry_run:
        argv.append("--dry-run")
    main(argv)


if __name__ == "__main__":
    # Usage examples:
    #   python SNN_based_method/scripts/train_lif_to_plif.py
    #       Run main_without_cli(), using editable parameters above: LIF 15 epochs + PLIF 5 epochs.
    #
    #   PowerShell:
    #       & D:/Anaconda3/envs/torchnew/python.exe D:/PYproject/SPAD/SNN_based_method/scripts/train_lif_to_plif.py --lif-epochs 15 --plif-epochs 5
    #
    #   PowerShell, skip LIF stage and fine-tune from an existing LIF checkpoint:
    #       & D:/Anaconda3/envs/torchnew/python.exe `
    #         D:/PYproject/SPAD/SNN_based_method/scripts/train_lif_to_plif.py `
    #         --lif-checkpoint D:/PYproject/SPAD/checkpoints/SNN/lif_to_plif_xxx/lif_stage/last.pth `
    #         --plif-epochs 5
    #
    # Common parameters:
    #   --lif-epochs <N>            First-stage LIF epochs, default 15.
    #   --plif-epochs <N>           Second-stage PLIF fine-tune epochs, default 5.
    #   --lif-checkpoint <path>     Optional existing LIF source checkpoint; skips LIF training.
    #   --dry-run                   Check target configuration without training.
    #   --freeze-plif-epochs <N>    Freeze PLIF tau *.w for the first N fine-tune epochs.
    #   --spike-tau <value>         Optional PLIF initial tau override; must be > 1.0.
    #
    # Outputs:
    #   checkpoints/SNN/lif_to_plif_<timestamp>/lif_stage/{last.pth,best.pth,config.json}
    #   checkpoints/SNN/lif_to_plif_<timestamp>/plif_stage/{last.pth,best.pth,config.json,lif_to_plif_migration.json}
    #   checkpoints/SNN/lif_to_plif_<timestamp>/run_summary.json
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
