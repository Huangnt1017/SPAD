#!/usr/bin/env python3
"""恢复最近一次可恢复的 SPAD 训练。

本脚本只负责定位可恢复的日志/checkpoint 组合，并复用 ``scripts.train``
中的训练入口继续运行。恢复参数优先来自 checkpoint 中保存的 ``args``，
日志只用于筛选候选运行和展示进度，避免从日志文本中推断可恢复状态。

CLI 运行示例（PowerShell）:

    # 只检查将要恢复的日志与 checkpoint，不真正启动训练。
    python scripts/resume_last_training.py --dry-run

    # 恢复 SPT 最近一次未完成训练。
    python scripts/resume_last_training.py --model spt

    # 恢复 SPT，并把目标总 epoch 覆盖为 150。
    python scripts/resume_last_training.py --model spt --epochs 150

    # 允许选择日志里已经达到目标 epoch 的训练，常用于追加训练轮数。
    python scripts/resume_last_training.py --model spt --include-finished --epochs 200

非 CLI 运行示例:

    # 修改 main_without_cli() 中的可编辑参数后，直接无参数运行脚本。
    python scripts/resume_last_training.py

    from scripts.resume_last_training import main_without_cli

    main_without_cli()

参数说明:
    --model:
        可选。限制只恢复某个模型；默认恢复 ``spt``，如需恢复其它模型再显式指定。
    --log-dir:
        训练日志目录。支持绝对路径或相对项目根目录的路径，默认 ``logs``。
    --save-dir:
        checkpoint 目录。支持绝对路径或相对项目根目录的路径，默认 ``checkpoints``。
    --epochs:
        可选。覆盖恢复后的目标总 epoch；不提供时使用 checkpoint 中保存的训练参数。
        真正启动训练时，目标 epoch 必须大于 checkpoint epoch。
    --batch-size:
        可选。覆盖 batch size；不提供时使用 checkpoint 中保存的训练参数。
    --include-finished:
        默认跳过日志显示已完成的训练；开启后允许选择已完成训练，通常需要配合
        更大的 ``--epochs`` 做追加训练。
    --dry-run:
        只打印将恢复的日志、checkpoint、checkpoint epoch、目标 epoch、batch size
        和数据目录，不启动训练。

输入输出约定:
    输入日志形如 ``logs/train_<model>_<timestamp>.log``。
    输入 checkpoint 形如 ``checkpoints/<model>_<timestamp>_last.pth``。
    接续训练只使用 ``_last.pth``，不从 ``_best.pth`` 兜底恢复。
    输出由 ``scripts.train.run_training`` 生成；恢复时追加写入原日志，并沿用原
    timestamp 覆盖/刷新同一组 ``_best.pth`` / ``_last.pth`` checkpoint。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train as train_module


LOG_NAME_RE = re.compile(r"^train_(?P<model>.+)_(?P<timestamp>\d{8}_\d{6}_\d{6})\.log$")
EPOCH_RE = re.compile(r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\]")
RESUME_CHECKPOINT_SUFFIX = "last"
CHECKPOINT_KEYS = (
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "best_val_top1",
    "class_to_idx",
    "args",
)


@dataclass(frozen=True)
class ScriptConfig:
    """CLI 与非 CLI 共享的恢复配置。"""

    model: Optional[str] = "spt"
    log_dir: Path = Path("logs")
    save_dir: Path = Path("checkpoints")
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    include_finished: bool = False
    dry_run: bool = False


def resolve_project_path(path_text: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """把相对路径统一解析到项目根目录下，方便从任意工作目录启动脚本。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def supported_model_choices() -> tuple[str, ...]:
    """从训练脚本的 parser 中读取可选模型名称，避免恢复脚本与训练脚本漂移。"""
    parser = train_module.build_parser()
    for action in parser._actions:
        if action.dest == "model" and action.choices is not None:
            return tuple(str(choice) for choice in action.choices)
    return ()


def load_checkpoint_meta(checkpoint_path: Path) -> dict[str, Any]:
    """读取恢复所需的 checkpoint 元数据，不把权重加载到训练流程中。"""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    if not isinstance(args, dict):
        args = {}

    missing_keys = [key for key in CHECKPOINT_KEYS if key not in payload]

    return {
        "epoch": int(payload.get("epoch", 0)),
        "best_val_top1": float(payload.get("best_val_top1", 0.0)),
        "args": args,
        "missing_keys": missing_keys,
    }


def read_log_progress(log_path: Path) -> dict[str, Optional[int]]:
    """扫描日志中最后一次出现的 epoch 进度，用于判断训练是否已经完成。"""
    progress: dict[str, Optional[int]] = {"epoch": None, "total": None}
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


def iter_log_runs(log_dir: Path, model: str = "") -> Iterable[dict[str, Any]]:
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
    """返回同一轮训练对应的 last checkpoint 候选。"""
    last_path = save_dir / f"{model}_{timestamp}_{RESUME_CHECKPOINT_SUFFIX}.pth"
    return [last_path] if last_path.exists() else []


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
) -> dict[str, Any]:
    """寻找最近的可恢复训练，返回日志、checkpoint 与保存参数的组合信息。"""
    runs = sorted(
        iter_log_runs(log_dir, model=model),
        key=lambda item: item["mtime"],
        reverse=True,
    )

    for run in runs:
        log_epoch = run["log_epoch"]
        log_total = run["log_total"]
        # 默认跳过已经达到目标 epoch 的日志，避免误把已完成训练当作中断任务。
        if (
            not include_finished
            and log_epoch is not None
            and log_total is not None
            and log_epoch >= log_total
        ):
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
                    "missing_checkpoint_keys": meta["missing_keys"],
                }
            )
            return target

    model_hint = f" for model '{model}'" if model else ""
    raise FileNotFoundError(
        f"No resumable run with a *_{RESUME_CHECKPOINT_SUFFIX}.pth checkpoint found"
        f"{model_hint} in {log_dir} with checkpoints in {save_dir}."
    )


def namespace_from_checkpoint(
    checkpoint_args: dict[str, Any],
    checkpoint_path: Path,
    resume_log_path: Path,
    resume_run_timestamp: str,
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
    merged["resume_log_file"] = str(resume_log_path.resolve())
    merged["resume_run_timestamp"] = resume_run_timestamp

    return argparse.Namespace(**merged)


def print_target_summary(target: dict[str, Any], args: argparse.Namespace) -> None:
    """启动训练前打印恢复摘要；dry-run 模式下这就是最终输出。"""
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
    print(f"  best val top1: {target['best_val_top1']:.4f}")

    missing_keys = target.get("missing_checkpoint_keys") or []
    if missing_keys:
        print(f"  warning: checkpoint is missing keys: {', '.join(missing_keys)}")

def validate_config(config: ScriptConfig) -> None:
    """校验脚本级参数，避免长训练启动后才暴露明显配置错误。"""
    if config.epochs is not None and config.epochs <= 0:
        raise ValueError("epochs must be a positive integer when provided.")
    if config.batch_size is not None and config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer when provided.")


def validate_resume_start(target: dict[str, Any], args: argparse.Namespace) -> None:
    """校验恢复点和目标 epoch，避免实际训练变成空跑。"""
    missing_keys = set(target.get("missing_checkpoint_keys") or [])
    if "model_state_dict" in missing_keys:
        raise KeyError(
            f"Checkpoint cannot restore model weights because model_state_dict is missing: "
            f"{target['checkpoint_path']}"
        )

    target_epochs = int(args.epochs)
    checkpoint_epoch = int(target["checkpoint_epoch"])
    if target_epochs <= checkpoint_epoch:
        raise ValueError(
            "Target epochs must be greater than checkpoint epoch before starting training. "
            f"Got target epochs={target_epochs}, checkpoint epoch={checkpoint_epoch}. "
            "Use --epochs with a larger value, especially when --include-finished is enabled."
        )


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器，供命令行入口和测试复用。"""
    model_choices = supported_model_choices()
    parser = argparse.ArgumentParser(
        description="Resume the latest resumable SPAD training run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/resume_last_training.py --dry-run\n"
            "  python scripts/resume_last_training.py --model spt\n"
            "  python scripts/resume_last_training.py --model spt --epochs 150\n"
            "  python scripts/resume_last_training.py --model spt --include-finished --epochs 200\n\n"
            "Outputs:\n"
            "  dry-run: prints selected log, checkpoint, checkpoint epoch, target epochs, batch size, data root.\n"
            "  training: delegates to scripts.train.run_training, appends the original log, and reuses checkpoint names."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="spt",
        choices=model_choices or None,
        help="Only resume this model. Default: spt.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Training log directory, absolute or relative to the project root. Default: logs.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory, absolute or relative to the project root. Default: checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override target total epochs. Must exceed checkpoint epoch when training starts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size. Omit to reuse the checkpoint's saved value.",
    )
    parser.add_argument(
        "--include-finished",
        action="store_true",
        help="Allow selecting a run whose log already reached its target epoch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected checkpoint and resolved arguments without starting training.",
    )
    return parser


def run_with_config(config: ScriptConfig) -> Optional[dict[str, str]]:
    """执行恢复流程；CLI 和非 CLI 入口都走这一条路径。"""
    validate_config(config)

    resolved_log_dir = resolve_project_path(config.log_dir)
    resolved_save_dir = resolve_project_path(config.save_dir)
    if not resolved_log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {resolved_log_dir}")
    if not resolved_save_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {resolved_save_dir}")

    target = find_resume_target(
        log_dir=resolved_log_dir,
        save_dir=resolved_save_dir,
        model=config.model or "",
        include_finished=config.include_finished,
    )
    training_args = namespace_from_checkpoint(
        checkpoint_args=target["checkpoint_args"],
        checkpoint_path=target["checkpoint_path"],
        resume_log_path=target["log_path"],
        resume_run_timestamp=target["timestamp"],
        log_dir=resolved_log_dir,
        save_dir=resolved_save_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
    )

    print_target_summary(target, training_args)

    if config.dry_run:
        return None

    validate_resume_start(target, training_args)
    return train_module.run_training(training_args)


def resume_latest_training(
    model: str = "spt",
    log_dir: str | Path = "logs",
    save_dir: str | Path = "checkpoints",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    include_finished: bool = False,
    dry_run: bool = False,
) -> Optional[dict[str, str]]:
    """非 CLI 兼容入口：从 Python 代码中恢复最近一次训练。

    Args:
        model: 只恢复指定模型；默认恢复 SPT。传入空字符串时选择最近的可恢复训练。
        log_dir: 训练日志目录，支持相对项目根目录的路径。
        save_dir: checkpoint 目录，支持相对项目根目录的路径。
        epochs: 覆盖目标总 epoch；None 表示沿用 checkpoint 中保存的值。
        batch_size: 覆盖 batch size；None 表示沿用 checkpoint 中保存的值。
        include_finished: 是否允许选择日志已达到目标 epoch 的训练。
        dry_run: 只打印恢复目标，不真正启动训练。

    Returns:
        真正恢复训练时返回 ``train.run_training`` 的结果；dry-run 时返回 None。
    """
    config = ScriptConfig(
        model=model or None,
        log_dir=Path(log_dir),
        save_dir=Path(save_dir),
        epochs=epochs,
        batch_size=batch_size,
        include_finished=include_finished,
        dry_run=dry_run,
    )
    return run_with_config(config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主程序：直接构建 parser，解析命令行参数并启动恢复流程。"""
    parser = build_parser()
    cli_args = parser.parse_args(argv)

    model_name = cli_args.model
    log_dir = cli_args.log_dir
    save_dir = cli_args.save_dir
    target_epochs = cli_args.epochs
    batch_size = cli_args.batch_size
    include_finished = cli_args.include_finished
    dry_run = cli_args.dry_run

    config = ScriptConfig(
        model=model_name,
        log_dir=log_dir,
        save_dir=save_dir,
        epochs=target_epochs,
        batch_size=batch_size,
        include_finished=include_finished,
        dry_run=dry_run,
    )
    run_with_config(config)
    return 0


def main_without_cli() -> None:
    """非 CLI 主程序：在函数内直接编辑参数，适合 IDE 调试和临时实验。"""
    # ===== 可编辑参数区 =====
    # model_name="spt" 表示恢复 SPT；改成 None 可扫描所有模型。
    model_name: Optional[str] = "spt"
    log_dir = Path("logs")
    save_dir = Path("checkpoints")
    target_epochs: Optional[int] = 100
    batch_size: Optional[int] = None
    include_finished = True
    dry_run = False

    # ===== 中间变量区 =====
    selected_model = model_name.strip() if isinstance(model_name, str) else None
    if selected_model == "":
        selected_model = None

    resolved_log_dir = Path(log_dir)
    resolved_save_dir = Path(save_dir)

    config = ScriptConfig(
        model=selected_model,
        log_dir=resolved_log_dir,
        save_dir=resolved_save_dir,
        epochs=target_epochs,
        batch_size=batch_size,
        include_finished=include_finished,
        dry_run=dry_run,
    )
    run_with_config(config)


if __name__ == "__main__":
    # 用法示例 (PowerShell):
    #   python scripts/resume_last_training.py
    #       无参数运行，进入 main_without_cli()，默认 dry-run。
    #   python scripts/resume_last_training.py --dry-run
    #       CLI dry-run，只打印恢复目标，不启动训练。
    #   python scripts/resume_last_training.py --model spt --epochs 150
    #       恢复 SPT，并覆盖目标总 epoch。
    #   python scripts/resume_last_training.py --model spt --include-finished --epochs 200
    #       允许选择已完成 SPT 日志，常用于追加训练轮数。
    # 接续cli：& "D:\Anaconda3\envs\torchnew\python.exe" "D:\PYproject\SPAD\scripts\resume_last_training.py" --model spt
    # 常用参数:
    #   --model <name>          只恢复指定模型；默认 spt。
    #   --log-dir logs          训练日志目录，支持相对项目根目录或绝对路径。
    #   --save-dir checkpoints  checkpoint 目录，只查找 *_last.pth。
    #   --epochs 150            覆盖目标总 epoch，必须大于 checkpoint epoch。
    #   --batch-size 16         覆盖 batch size，不填则沿用 checkpoint 中保存的值。
    #   --include-finished      允许选择日志已达到目标 epoch 的训练。
    #   --dry-run               打印恢复摘要，不调用 train.run_training。
    #
    # 输出:
    #   dry-run:
    #       打印选中的日志、checkpoint、checkpoint epoch、恢复起始 epoch、
    #       目标 epoch、batch size、data root 和 best_val_top1。
    #   真正训练:
    #       复用 scripts.train.run_training，追加写入原
    #       logs/train_<model>_<timestamp>.log，并刷新同 timestamp 的
    #       checkpoints/<model>_<timestamp>_best.pth 和
    #       checkpoints/<model>_<timestamp>_last.pth。
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
