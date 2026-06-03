"""SPAD SNN 模型的标准训练入口。"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
    build_run_name,
    config_from_args,
    divide_average,
    load_checkpoint,
    make_checkpoint_run_dir,
    prepare_model_input,
    reduce_loss_dict,
    resolve_output_root,
    save_checkpoint,
    update_average,
)

DEFAULT_TRAIN_DATA_PATHS = [
    r"D:\PYproject\SPADdata\0825",
    r"D:\PYproject\SPADdata\0826",
]
DEFAULT_TRAIN_CSV_PATHS = [
    r"D:\PYproject\SPADdata\0825\0825-group.csv",
    r"D:\PYproject\SPADdata\0826\0826-group.csv",
]


def setup_logger(log_dir: str | Path, run_name: str) -> tuple[logging.Logger, Path]:
    """创建单文件训练日志, 同时输出到控制台。"""
    log_root = resolve_output_root(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"{run_name}.log"

    logger_name = f"spad_snn_train_{run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    mode = "a" if log_file.exists() else "w"

    file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, log_file


def _format_values(values: dict[str, float]) -> str:
    """把指标字典格式化为稳定的一行日志。"""
    if not values:
        return "none"
    return " ".join(f"{key}={value:.6f}" for key, value in sorted(values.items()))


def _sync_if_cuda(device: torch.device) -> None:
    """同步 CUDA, 让阶段耗时更接近真实 GPU 执行时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stamp(stage_times: dict[str, float], name: str, device: torch.device) -> None:
    """记录一个训练阶段的时间戳。"""
    _sync_if_cuda(device)
    stage_times[name] = time.perf_counter()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: SNNConfig,
    epoch: int,
    trace_steps: int = 0,
) -> tuple[float, dict[str, float]]:
    """训练一个 epoch, 返回平均总 loss 与各子 loss。"""
    model.train()
    total_loss = 0.0
    loss_sums: dict[str, float] = {}
    num_steps = 0
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))

    progress = tqdm(data_loader, desc=f"train {epoch:03d}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(progress):
        stage_times: dict[str, float] | None = {} if batch_index < trace_steps else None
        if stage_times is not None:
            _stamp(stage_times, "batch_start", device)

        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch.get("label")
        labels = labels.to(device, non_blocking=True) if labels is not None else None
        if stage_times is not None:
            _stamp(stage_times, "to_device", device)

        model_input = prepare_model_input(frames).to(device, non_blocking=True)
        if stage_times is not None:
            _stamp(stage_times, "prepare_input", device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            result = model(model_input)
            if stage_times is not None:
                _stamp(stage_times, "forward", device)
            loss, loss_items = criterion(result, labels)
            loss_for_backward = loss / grad_accum_steps
            if stage_times is not None:
                _stamp(stage_times, "loss", device)

        should_step = (
            (batch_index + 1) % grad_accum_steps == 0
            or (batch_index + 1) == len(data_loader)
        )
        if use_amp:
            scaler.scale(loss_for_backward).backward()
            if stage_times is not None:
                _stamp(stage_times, "backward", device)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if stage_times is not None:
                    _stamp(stage_times, "clip_grad", device)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if stage_times is not None:
                    _stamp(stage_times, "optimizer_step", device)
        else:
            loss_for_backward.backward()
            if stage_times is not None:
                _stamp(stage_times, "backward", device)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if stage_times is not None:
                    _stamp(stage_times, "clip_grad", device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if stage_times is not None:
                    _stamp(stage_times, "optimizer_step", device)

        loss_value = float(loss.detach().cpu().item())
        reduced_loss_items = reduce_loss_dict(loss_items)
        if stage_times is not None:
            _stamp(stage_times, "batch_done", device)
        total_loss += loss_value
        update_average(loss_sums, reduced_loss_items)
        num_steps += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

        if stage_times is not None:
            names = list(stage_times)
            durations = {
                f"{names[i - 1]}->{names[i]}": stage_times[names[i]] - stage_times[names[i - 1]]
                for i in range(1, len(names))
            }
            tqdm.write(
                f"[trace] epoch={epoch:03d} batch={batch_index:04d} "
                + " ".join(f"{key}={value:.3f}s" for key, value in durations.items())
            )

    return total_loss / max(num_steps, 1), divide_average(loss_sums, num_steps)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    metrics,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """验证一个 epoch, 返回 loss、子 loss 和图像指标。"""
    model.eval()
    total_loss = 0.0
    loss_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    num_steps = 0

    progress = tqdm(data_loader, desc=f"val {epoch:03d}", leave=False)
    for batch in progress:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch.get("label")
        labels = labels.to(device, non_blocking=True) if labels is not None else None

        model_input = prepare_model_input(frames).to(device, non_blocking=True)
        result = model(model_input)
        loss, loss_items = criterion(result, labels)

        if labels is not None:
            update_average(metric_sums, metrics.compute(result, labels))

        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        update_average(loss_sums, reduce_loss_dict(loss_items))
        num_steps += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return (
        total_loss / max(num_steps, 1),
        divide_average(loss_sums, num_steps),
        divide_average(metric_sums, num_steps),
    )


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="训练 SPAD SNN 成像模型")
    add_config_arguments(parser)
    parser.add_argument("--save-every", type=int, default=None, help="每 N 个 epoch 额外保存一次 checkpoint")
    parser.add_argument("--trace-steps", type=int, default=0, help="打印前 N 个训练 batch 的阶段耗时")
    return parser


def _same_paths(left: list[str] | None, right: list[str]) -> bool:
    """比较路径集合, 用于识别默认训练目录。"""
    if left is None:
        return False
    left_resolved = {str(Path(path).resolve()).lower() for path in left}
    right_resolved = {str(Path(path).resolve()).lower() for path in right}
    return left_resolved == right_resolved


def apply_default_train_paths(cfg: SNNConfig) -> SNNConfig:
    """无参数运行时默认使用 0825/0826 训练数据及其 CSV 清单。"""
    updates: dict[str, object] = {}
    if not cfg.data_paths:
        updates["data_paths"] = DEFAULT_TRAIN_DATA_PATHS
        updates["csv_paths"] = DEFAULT_TRAIN_CSV_PATHS
    elif not cfg.csv_paths and _same_paths(cfg.data_paths, DEFAULT_TRAIN_DATA_PATHS):
        updates["csv_paths"] = DEFAULT_TRAIN_CSV_PATHS

    if not updates:
        return cfg
    return cfg.clone_with(**updates)


def main() -> None:
    """执行标准训练流程。"""
    args = build_argparser().parse_args()
    cfg = config_from_args(args)
    cfg = apply_default_train_paths(cfg)
    if args.save_every is not None:
        cfg = cfg.clone_with(save_every=args.save_every)
    seed_everything(cfg.seed)

    train_loader, val_loader, test_loader, dataset = cfg.build_dataloaders()
    device = cfg.resolved_device()
    model = cfg.build_model().to(device)
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

    start_epoch = 1
    best_val_loss = float("inf")
    if cfg.checkpoint_path:
        checkpoint = load_checkpoint(
            cfg.checkpoint_path,
            model,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("metrics", {}).get("best_val_loss", best_val_loss))

    run_name = build_run_name(cfg, "train")
    cfg = cfg.clone_with(run_name=run_name)
    checkpoint_run_dir = make_checkpoint_run_dir(cfg, "train", run_name=run_name)
    logger, log_file = setup_logger(cfg.log_dir, run_name)
    cfg.save(checkpoint_run_dir / "config.json")

    logger.info("=== Training Configuration ===")
    logger.info("run_name=%s", run_name)
    logger.info("log_file=%s", log_file)
    logger.info("checkpoint_run_dir=%s", checkpoint_run_dir)
    logger.info("data_paths=%s", cfg.data_paths)
    logger.info("csv_paths=%s", cfg.csv_paths)
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
    if cfg.checkpoint_path:
        logger.info("resumed_from=%s start_epoch=%d", cfg.checkpoint_path, start_epoch)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_items = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            epoch,
            trace_steps=max(0, args.trace_steps),
        )
        val_loss, val_items, val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            metrics,
            device,
            epoch,
        )
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_record = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_items": train_items,
            "val_items": val_items,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "lr": scheduler.get_last_lr()[0],
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
            epoch=epoch,
            cfg=cfg,
            metrics=epoch_record,
        )
        if is_best:
            save_checkpoint(
                checkpoint_run_dir / "best.pth",
                model=model,
                optimizer=optimizer,
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
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
            logger.info("Saved epoch checkpoint to %s", checkpoint_run_dir / f"epoch_{epoch:03d}.pth")

        logger.info("Saved last checkpoint to %s", checkpoint_run_dir / "last.pth")

    logger.info("Training finished. Best val loss=%.6f", best_val_loss)
    logger.info("log_file=%s", log_file)
    logger.info("checkpoint_run_dir=%s", checkpoint_run_dir)


if __name__ == "__main__":
    main()
