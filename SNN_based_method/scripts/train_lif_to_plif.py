"""LIF checkpoint 到 PLIF 模型的二阶段微调训练入口。

CLI example:
    python SNN_based_method/scripts/train_lif_to_plif.py --lif-checkpoint checkpoints/SNN/train_xxx/last.pth --epochs 10 --lr 2e-4 --spike-tau 2.0

Non-CLI example:
    python SNN_based_method/scripts/train_lif_to_plif.py

参数说明:
    --lif-checkpoint 指向第一阶段 LIF 训练得到的 last.pth 或 best.pth。
    普通网络结构、数据、loss 和运行时参数继续使用 train.py 的通用参数。
    本脚本会强制构建 spike_mode=plif 的目标模型, 不恢复 LIF optimizer/scheduler。

输入/输出:
    输入模型数据仍为 [B, 4096, P] raw ToF, 由现有 dataloader 和 train.py 训练循环准备。
    输出 checkpoint 保存到 checkpoint_dir/lif_to_plif_<timestamp>/ 下, 包含 best.pth、last.pth 和迁移报告。
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
    make_checkpoint_run_dir,
    resolve_output_root,
    save_checkpoint,
)


@dataclass
class ScriptConfig:
    """CLI 与非 CLI 共享的 LIF->PLIF 微调配置。"""

    lif_checkpoint: Path
    args: argparse.Namespace
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


def build_parser() -> argparse.ArgumentParser:
    """构建 LIF->PLIF 微调命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从 LIF checkpoint 迁移权重并微调 PLIF SPAD SNN 模型",
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--lif-checkpoint",
        required=True,
        help="第一阶段 LIF checkpoint 路径, 可为 last.pth 或 best.pth",
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
        help="只构建目标 PLIF 模型并检查 checkpoint 迁移情况, 不启动训练",
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


def _build_plif_config(args: argparse.Namespace, lif_checkpoint: Path) -> SNNConfig:
    """从 LIF checkpoint/config 继承配置, 再强制切换到 PLIF 微调。"""
    if args.checkpoint_path:
        raise ValueError("本脚本中 --checkpoint 保留给普通训练入口使用; 请使用 --lif-checkpoint")
    if args.spike_mode not in (None, "plif"):
        raise ValueError("LIF->PLIF 微调目标固定为 --spike-mode plif, 请不要传其他 spike_mode")

    config_args = argparse.Namespace(**vars(args))
    config_args.checkpoint_path = str(lif_checkpoint)
    cfg = config_from_checkpoint_and_args(config_args)
    cfg = apply_default_train_paths(cfg)

    plif_tau = float(cfg.spike_tau)
    if args.spike_tau is None and plif_tau <= 1.0:
        plif_tau = 2.0
    if plif_tau <= 1.0:
        raise ValueError("PLIF fine-tune requires spike_tau > 1.0")

    cfg = cfg.clone_with(
        model_backend="new",
        spike_mode="plif",
        spike_tau=plif_tau,
        checkpoint_path=str(lif_checkpoint),
    )
    if args.save_every is not None:
        cfg = cfg.clone_with(save_every=args.save_every)

    run_name = build_run_name(cfg, "lif_to_plif")
    return cfg.clone_with(run_name=run_name)


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


def _print_dry_run(cfg: SNNConfig, report: MigrationReport, device: torch.device) -> None:
    """输出 dry-run 检查结果。"""
    print("=== LIF -> PLIF fine-tune dry run ===")
    print(f"device={device}")
    print(f"source_checkpoint={report.source_checkpoint}")
    print(f"source_epoch={report.source_epoch}")
    print(f"source_spike_mode={report.source_spike_mode}")
    print(f"target_spike_mode={cfg.spike_mode}")
    print(f"target_spike_tau={cfg.spike_tau}")
    print(f"run_name={cfg.run_name}")
    print(f"copied_count={len(report.copied_keys)}")
    print(f"initialized_tau_keys={report.initialized_keys}")
    print(f"skipped_keys={report.skipped_keys}")
    print(f"unexpected_keys={report.unexpected_keys}")
    print("=== Target configuration ===")
    print(cfg.summary())


def run_with_config(script_cfg: ScriptConfig) -> None:
    """执行 LIF->PLIF 权重迁移和 fine-tune 训练。"""
    lif_checkpoint = _resolve_checkpoint(script_cfg.lif_checkpoint)
    args = script_cfg.args
    cfg = _build_plif_config(args, lif_checkpoint)

    seed_everything(cfg.seed)
    configure_torch_runtime(cfg)
    device = cfg.resolved_device()
    model = cfg.build_model().to(device)
    source_checkpoint, migration_report = load_lif_weights_into_plif(
        lif_checkpoint,
        model,
        map_location=device,
        allow_partial_model_load=script_cfg.allow_partial_model_load,
    )

    if script_cfg.dry_run:
        _print_dry_run(cfg, migration_report, device)
        return

    checkpoint_run_dir = make_checkpoint_run_dir(cfg, "lif_to_plif", run_name=cfg.run_name)
    logger, log_file = setup_logger(cfg.log_dir, cfg.run_name or "lif_to_plif")
    cfg.save(checkpoint_run_dir / "config.json")
    with (checkpoint_run_dir / "lif_to_plif_migration.json").open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(migration_report), file_obj, indent=2, ensure_ascii=False)

    label_stats = ensure_train_precomputed_labels(cfg)
    train_loader, val_loader, test_loader, dataset = cfg.build_dataloaders()
    criterion = cfg.build_loss().to(device)
    metrics = cfg.build_metrics()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
    )

    logger.info("=== LIF -> PLIF Fine-tune Configuration ===")
    logger.info("run_name=%s", cfg.run_name)
    logger.info("log_file=%s", log_file)
    logger.info("checkpoint_run_dir=%s", checkpoint_run_dir)
    logger.info("lif_checkpoint=%s", lif_checkpoint)
    logger.info("source_epoch=%s", migration_report.source_epoch)
    logger.info("source_spike_mode=%s", migration_report.source_spike_mode)
    logger.info("migration_summary=%s", json.dumps(migration_report.summary(), ensure_ascii=False))
    logger.info("data_paths=%s", cfg.data_paths)
    logger.info("csv_paths=%s", cfg.csv_paths)
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

    best_val_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tau_trainable = epoch > script_cfg.freeze_plif_epochs
        affected_tau = set_plif_tau_trainable(model, tau_trainable)
        if epoch == 1 or epoch == script_cfg.freeze_plif_epochs + 1:
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
            trace_steps=max(0, script_cfg.trace_steps),
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
            "event": "lif_to_plif_epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_items": train_items,
            "val_items": val_items,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "lr": scheduler.get_last_lr()[0],
            "source_epoch": migration_report.source_epoch,
            "source_checkpoint": str(lif_checkpoint),
            "migration": migration_report.summary(),
            "freeze_plif_epochs": script_cfg.freeze_plif_epochs,
            "tau_trainable": tau_trainable,
        }

        logger.info(
            "Epoch [%d/%d] | train_loss=%.6f val_loss=%.6f best_val_loss=%.6f lr=%.8f",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            best_val_loss,
            scheduler.get_last_lr()[0],
        )
        logger.info("Epoch [%d/%d] | train_items: %s", epoch, cfg.epochs, _format_values(train_items))
        logger.info("Epoch [%d/%d] | val_items: %s", epoch, cfg.epochs, _format_values(val_items))
        logger.info("Epoch [%d/%d] | val_metrics: %s", epoch, cfg.epochs, _format_values(val_metrics))

        save_checkpoint(
            checkpoint_run_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            cfg=cfg,
            metrics=epoch_record,
        )
        if is_best:
            save_checkpoint(
                checkpoint_run_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
            logger.info(
                "Saved new best checkpoint to %s (val_loss=%.6f)",
                checkpoint_run_dir / "best.pth",
                best_val_loss,
            )
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                checkpoint_run_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
            logger.info("Saved epoch checkpoint to %s", checkpoint_run_dir / f"epoch_{epoch:03d}.pth")
        logger.info("Updated last checkpoint to %s", checkpoint_run_dir / "last.pth")

    logger.info("LIF -> PLIF fine-tune finished. Best val loss=%.6f", best_val_loss)
    logger.info("log_file=%s", log_file)
    logger.info("checkpoint_run_dir=%s", checkpoint_run_dir)


def _script_config_from_args(args: argparse.Namespace) -> ScriptConfig:
    """把 argparse 结果整理成脚本配置。"""
    return ScriptConfig(
        lif_checkpoint=Path(args.lif_checkpoint),
        args=args,
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
    lif_checkpoint = Path(r"D:/PYproject/SPAD/checkpoints/SNN/train_20260604_113734/last.pth")
    epochs = 10
    learning_rate = 2.0e-4
    spike_tau = 2.0
    dry_run = True

    # ===== Intermediate variables =====
    argv = [
        "--lif-checkpoint",
        str(lif_checkpoint),
        "--epochs",
        str(epochs),
        "--lr",
        str(learning_rate),
        "--spike-tau",
        str(spike_tau),
    ]
    if dry_run:
        argv.append("--dry-run")
    main(argv)


if __name__ == "__main__":
    # Usage examples:
    #   python SNN_based_method/scripts/train_lif_to_plif.py
    #       Run main_without_cli(), using editable parameters above.
    #   python SNN_based_method/scripts/train_lif_to_plif.py --lif-checkpoint D:/PYproject/SPAD/checkpoints/SNN/train_xxx/last.pth --epochs 10 --lr 2e-4 --spike-tau 2.0
    #       Load a LIF checkpoint, initialize PLIF tau parameters, and fine-tune from epoch 1.
    #
    # Common parameters:
    #   --lif-checkpoint <path>     Required LIF source checkpoint.
    #   --dry-run                   Check migration without training.
    #   --freeze-plif-epochs <N>    Freeze PLIF tau *.w for the first N epochs.
    #   --spike-tau <value>         PLIF initial tau; must be > 1.0.
    #
    # Outputs:
    #   checkpoints/SNN/lif_to_plif_<timestamp>/{last.pth,best.pth,config.json,lif_to_plif_migration.json}
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
