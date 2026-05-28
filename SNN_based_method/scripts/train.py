"""Standard training entrypoint for the SPAD SNN model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from SNN.SNN_config import SNNConfig
    from SNN.data import seed_everything
    from SNN.runtime import (
        add_config_arguments,
        append_jsonl,
        config_from_args,
        divide_average,
        load_checkpoint,
        make_run_dir,
        prepare_model_input,
        reduce_loss_dict,
        reset_spiking_state,
        save_checkpoint,
        update_average,
    )
except ModuleNotFoundError:
    from SNN_config import SNNConfig
    from data import seed_everything
    from runtime import (
        add_config_arguments,
        append_jsonl,
        config_from_args,
        divide_average,
        load_checkpoint,
        make_run_dir,
        prepare_model_input,
        reduce_loss_dict,
        reset_spiking_state,
        save_checkpoint,
        update_average,
    )


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: SNNConfig,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_sums: dict[str, float] = {}
    num_steps = 0
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    progress = tqdm(data_loader, desc=f"train {epoch:03d}", leave=False)
    for batch in progress:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch.get("label")
        labels = labels.to(device, non_blocking=True) if labels is not None else None

        model_input = prepare_model_input(frames).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            result = model(model_input)
            loss, loss_items = criterion(result, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        reset_spiking_state(model)
        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        update_average(loss_sums, reduce_loss_dict(loss_items))
        num_steps += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

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
    """Run validation for one epoch."""
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

        reset_spiking_state(model)
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
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train SPAD SNN model")
    add_config_arguments(parser)
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N epochs")
    return parser


def main() -> None:
    """Run standard training."""
    args = build_argparser().parse_args()
    cfg = config_from_args(args)
    if args.save_every is not None:
        cfg = cfg.clone_with(save_every=args.save_every)
    seed_everything(cfg.seed)

    train_loader, val_loader, _, dataset = cfg.build_dataloaders()
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

    run_dir = make_run_dir(cfg, "train")
    log_path = run_dir / "train.jsonl"
    cfg.save(run_dir / "config.json")
    append_jsonl(
        log_path,
        {
            "event": "start",
            "dataset_size": len(dataset),
            "config": cfg.to_dict(),
        },
    )
    print(cfg.summary())
    print(f"run_dir={run_dir}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_items = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            epoch,
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
        append_jsonl(log_path, epoch_record)

        save_checkpoint(
            run_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            metrics=epoch_record,
        )
        if is_best:
            save_checkpoint(
                run_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best={best_val_loss:.6f}"
        )


if __name__ == "__main__":
    main()

