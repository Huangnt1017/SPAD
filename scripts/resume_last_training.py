#!/usr/bin/env python3
"""恢复最近一次中断的 SPAD 训练。

这个脚本只负责定位可恢复的日志/checkpoint 组合，并复用 ``scripts.train``
里的训练逻辑继续运行。恢复参数优先来自 checkpoint 中保存的 ``args``，
避免从日志文本里猜训练状态。

CLI 调用示例（PowerShell）:

    # 只查看将要恢复哪个训练，不真正启动训练。
    python scripts/resume_last_training.py --dry-run

    # 恢复最近一次未完成训练。
    python scripts/resume_last_training.py

    # 只恢复指定模型，并把目标总 epoch 改成 150。
    python scripts/resume_last_training.py --model pointtransformer --epochs 150

    # 允许选择日志里已经显示完成的训练，常用于继续追加训练轮数。
    python scripts/resume_last_training.py --include-finished --epochs 200

非 CLI 调用示例:

    from scripts.resume_last_training import resume_latest_training

    result = resume_latest_training(model="pointtransformer", epochs=150, dry_run=True)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train as train_module


LOG_NAME_RE = re.compile(r"^train_(?P<model>.+)_(?P<timestamp>\d{8}_\d{6}_\d{6})\.log$")
EPOCH_RE = re.compile(r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\]")


def resolve_project_path(path_text: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """把相对路径统一解析到项目根目录下，方便从任意工作目录启动脚本。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_checkpoint_meta(checkpoint_path: Path) -> Dict[str, Any]:
    """读取恢复所需的 checkpoint 元数据，不加载到训练流程里。"""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    if not isinstance(args, dict):
        args = {}

    return {
        "epoch": int(payload.get("epoch", 0)),
        "best_val_top1": float(payload.get("best_val_top1", 0.0)),
        "args": args,
    }


def read_log_args(log_path: Path) -> Dict[str, Any]:
    """读取日志中记录的原始参数；保留给排查旧日志或扩展逻辑使用。"""
    if not log_path.exists():
        return {}

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            marker = "args="
            if marker not in line:
                continue
            raw = line.split(marker, 1)[1].strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}
    return {}


def read_log_progress(log_path: Path) -> Dict[str, Optional[int]]:
    """扫描日志中最后一次出现的 epoch 进度，用于判断训练是否已经完成。"""
    progress: Dict[str, Optional[int]] = {"epoch": None, "total": None}
    if not log_path.exists():
        return progress

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = EPOCH_RE.search(line)
            if match is None:
                continue
            progress["epoch"] = int(match.group("epoch"))
            progress["total"] = int(match.group("total"))
    return progress


def iter_log_runs(log_dir: Path, model: str = "") -> Iterable[Dict[str, Any]]:
    """按日志文件枚举训练运行记录，模型名为空时不过滤。"""
    for log_path in log_dir.glob("train_*.log"):
        match = LOG_NAME_RE.match(log_path.name)
        if match is None:
            continue

        run_model = match.group("model")
        if model and run_model.lower() != model.lower():
            continue

        progress = read_log_progress(log_path)
        yield {
            "model": run_model,
            "timestamp": match.group("timestamp"),
            "log_path": log_path,
            "mtime": log_path.stat().st_mtime,
            "log_epoch": progress["epoch"],
            "log_total": progress["total"],
        }


def checkpoint_candidates(save_dir: Path, model: str, timestamp: str) -> list[Path]:
    """返回同一轮训练对应的 checkpoint 候选，优先使用可精确续训的 last。"""
    last_path = save_dir / f"{model}_{timestamp}_last.pth"
    best_path = save_dir / f"{model}_{timestamp}_best.pth"
    candidates = []
    if last_path.exists():
        candidates.append(last_path)
    if best_path.exists():
        candidates.append(best_path)
    return candidates


def checkpoint_kind(checkpoint_path: Path) -> str:
    """根据文件名识别 checkpoint 类型，便于打印恢复提示。"""
    if checkpoint_path.name.endswith("_last.pth"):
        return "last"
    if checkpoint_path.name.endswith("_best.pth"):
        return "best"
    return "checkpoint"


def find_resume_target(
    log_dir: Path,
    save_dir: Path,
    model: str = "",
    include_finished: bool = False,
) -> Dict[str, Any]:
    """寻找最近的可恢复训练，返回日志、checkpoint 与保存参数的组合信息。"""
    runs = sorted(iter_log_runs(log_dir, model=model), key=lambda item: item["mtime"], reverse=True)

    for run in runs:
        log_epoch = run["log_epoch"]
        log_total = run["log_total"]
        # 默认跳过已经达到目标 epoch 的日志，避免误把已完成训练当作中断任务。
        if not include_finished and log_epoch is not None and log_total is not None and log_epoch >= log_total:
            continue

        for checkpoint_path in checkpoint_candidates(save_dir, run["model"], run["timestamp"]):
            # 恢复点以 checkpoint 元数据为准；日志只用于选择候选运行和提示进度。
            meta = load_checkpoint_meta(checkpoint_path)
            target = dict(run)
            target.update(
                {
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_epoch": meta["epoch"],
                    "best_val_top1": meta["best_val_top1"],
                    "checkpoint_args": meta["args"],
                    "checkpoint_kind": checkpoint_kind(checkpoint_path),
                }
            )
            return target

    raise FileNotFoundError(
        f"No resumable run found in {log_dir} with checkpoints in {save_dir}."
    )


def namespace_from_checkpoint(
    checkpoint_args: Dict[str, Any],
    checkpoint_path: Path,
    log_dir: Path,
    save_dir: Path,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> argparse.Namespace:
    """把 checkpoint 中保存的训练参数还原为 ``train.run_training`` 可用的 Namespace。"""
    parser = train_module.build_parser()
    defaults = vars(parser.parse_args([]))
    merged = {**defaults, **checkpoint_args}

    if epochs is not None:
        merged["epochs"] = epochs
    if batch_size is not None:
        merged["batch_size"] = batch_size

    merged["resume"] = str(checkpoint_path.resolve())
    merged["log_dir"] = str(log_dir.resolve())
    merged["save_dir"] = str(save_dir.resolve())

    return argparse.Namespace(**merged)


def print_target_summary(target: Dict[str, Any], args: argparse.Namespace) -> None:
    """启动训练前打印恢复摘要，dry-run 模式下这就是最终输出。"""
    print("Selected resumable run:")
    print(f"  model: {target['model']}")
    print(f"  log: {target['log_path']}")
    print(f"  checkpoint: {target['checkpoint_path']} ({target['checkpoint_kind']})")
    print(f"  log epoch: {target['log_epoch']}/{target['log_total']}")
    print(f"  checkpoint epoch: {target['checkpoint_epoch']}")
    print(f"  resume start epoch: {target['checkpoint_epoch'] + 1}")
    print(f"  target epochs: {args.epochs}")
    print(f"  batch size: {args.batch_size}")
    print(f"  data root: {args.data_root}")
    if target["checkpoint_kind"] != "last" and target["log_epoch"] != target["checkpoint_epoch"]:
        print(
            "  note: no matching _last checkpoint was found, so resume uses the saved best checkpoint."
        )


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器，供命令行入口和测试复用。"""
    parser = argparse.ArgumentParser(description="Resume the latest interrupted SPAD training run.")
    parser.add_argument("--model", type=str, default="", help="Only resume this model, e.g. pointtransformer.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Training log directory.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override target total epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--include-finished",
        action="store_true",
        help="Allow selecting a run whose log already reached its target epoch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected checkpoint and arguments without starting training.",
    )
    return parser


def resume_latest_training(
    model: str = "",
    log_dir: str | Path = "logs",
    save_dir: str | Path = "checkpoints",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    include_finished: bool = False,
    dry_run: bool = False,
) -> Optional[Dict[str, str]]:
    """非 CLI 入口：从 Python 代码中恢复最近一次训练。

    Args:
        model: 只恢复指定模型；为空时选择最近的可恢复训练。
        log_dir: 训练日志目录，支持相对项目根目录的路径。
        save_dir: checkpoint 目录，支持相对项目根目录的路径。
        epochs: 覆盖目标总 epoch；None 表示沿用 checkpoint 中保存的值。
        batch_size: 覆盖 batch size；None 表示沿用 checkpoint 中保存的值。
        include_finished: 是否允许选择日志已达到目标 epoch 的训练。
        dry_run: 只打印恢复目标，不真正启动训练。

    Returns:
        真实恢复训练时返回 ``train.run_training`` 的结果；dry-run 时返回 None。
    """
    resolved_log_dir = resolve_project_path(log_dir)
    resolved_save_dir = resolve_project_path(save_dir)

    target = find_resume_target(
        log_dir=resolved_log_dir,
        save_dir=resolved_save_dir,
        model=model,
        include_finished=include_finished,
    )
    training_args = namespace_from_checkpoint(
        checkpoint_args=target["checkpoint_args"],
        checkpoint_path=target["checkpoint_path"],
        log_dir=resolved_log_dir,
        save_dir=resolved_save_dir,
        epochs=epochs,
        batch_size=batch_size,
    )

    print_target_summary(target, training_args)

    if dry_run:
        return None

    return train_module.run_training(training_args)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI 入口：解析命令行参数后调用可复用的非 CLI 入口。"""
    parser = build_parser()
    cli_args = parser.parse_args(argv)
    resume_latest_training(
        model=cli_args.model,
        log_dir=cli_args.log_dir,
        save_dir=cli_args.save_dir,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        include_finished=cli_args.include_finished,
        dry_run=cli_args.dry_run,
    )


if __name__ == "__main__":
    # 常用命令:
    #   python scripts/resume_last_training.py --dry-run
    #   python scripts/resume_last_training.py --model pointtransformer
    #   python scripts/resume_last_training.py --include-finished --epochs 200
    main()
