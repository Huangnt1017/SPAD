#!/usr/bin/env python3
"""按模型队列自动串行训练 SPAD 点云模型。

CLI example:
    python scripts/auto_train_monitor.py --models pointnet2 graph_residual spt --poll-seconds 30

Non-CLI example:
    python scripts/auto_train_monitor.py

说明:
    无参运行会使用 main_without_cli() 中显式定义的 AUTO_TRAIN_MODELS。
    脚本每次只启动一个训练进程；检测到当前模型日志出现 "Training finished" 后,
    自动启动队列中的下一个模型。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
STATE_FILE = PROJECT_ROOT / ".auto_train_monitor_state.json"
MONITOR_LOG = PROJECT_ROOT / "auto_train_next.log"




@dataclass
class AutoTrainConfig:
    """自动训练运行配置。"""

    models: list[str]
    epochs: int = 100
    batch_size: int = 32
    poll_seconds: int = 30
    python_exe: Path = Path(sys.executable)
    data_root: Optional[Path] = None
    log_dir: Path = PROJECT_ROOT / "logs" / "CLS"
    save_dir: Path = PROJECT_ROOT / "checkpoints" / "CLS"
    state_file: Path = STATE_FILE
    monitor_log: Path = MONITOR_LOG
    dry_run: bool = False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_monitor_log(config: AutoTrainConfig, message: str) -> None:
    """写入监控日志并同步输出到控制台。"""
    line = f"[{_now()}] {message}"
    print(line)
    if config.dry_run:
        return
    config.monitor_log.parent.mkdir(parents=True, exist_ok=True)
    with open(config.monitor_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state(state_file: Path) -> dict:
    """读取自动训练状态；不存在时返回空状态。"""
    if not state_file.exists():
        return {}
    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_file: Path, state: dict) -> None:
    """保存当前训练队列状态。"""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def latest_log_for_model(log_dir: Path, model: str) -> Optional[Path]:
    """返回指定模型最近一次训练日志。"""
    pattern = f"train_{model}_*.log"
    candidates = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def log_is_finished(log_path: Path) -> bool:
    """检查训练日志是否已完整结束。"""
    if not log_path.exists():
        return False
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        tail = f.readlines()[-20:]
    text = "".join(tail)
    return "Training finished" in text


def process_is_running(process: subprocess.Popen) -> bool:
    """判断 Popen 进程是否仍在运行。"""
    return process.poll() is None


def build_train_command(config: AutoTrainConfig, model: str) -> list[str]:
    """构造单个模型的训练命令。"""
    cmd = [
        str(config.python_exe),
        str(TRAIN_SCRIPT),
        "--model",
        model,
        "--batch-size",
        str(config.batch_size),
        "--epochs",
        str(config.epochs),
        "--log-dir",
        str(config.log_dir),
        "--save-dir",
        str(config.save_dir),
    ]
    if config.data_root is not None:
        cmd.extend(["--data-root", str(config.data_root)])
    return cmd


def launch_training(config: AutoTrainConfig, model: str) -> subprocess.Popen:
    """启动一个模型训练进程。"""
    cmd = build_train_command(config, model)
    append_monitor_log(config, "启动训练: " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)


def run_auto_train(config: AutoTrainConfig) -> None:
    """按队列串行启动训练，完成一个再启动下一个。"""
    if not config.models:
        raise ValueError("models must not be empty.")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(config.state_file)
    if state.get("models") != config.models:
        state = {"models": config.models, "next_index": 0, "runs": []}

    next_index = int(state.get("next_index", 0))
    append_monitor_log(config, f"自动训练队列: {', '.join(config.models)}")

    while next_index < len(config.models):
        model = config.models[next_index]
        if config.dry_run:
            append_monitor_log(config, "dry-run: " + " ".join(build_train_command(config, model)))
            next_index += 1
            continue

        process = launch_training(config, model)
        state["current_model"] = model
        state["current_pid"] = process.pid
        state["next_index"] = next_index
        save_state(config.state_file, state)

        while process_is_running(process):
            time.sleep(config.poll_seconds)

        exit_code = process.poll()
        log_path = latest_log_for_model(config.log_dir, model)
        finished = log_path is not None and log_is_finished(log_path)
        run_record = {
            "model": model,
            "exit_code": exit_code,
            "log_path": str(log_path) if log_path else "",
            "finished": finished,
            "finished_at": _now(),
        }
        state.setdefault("runs", []).append(run_record)

        if exit_code != 0 or not finished:
            state["failed_model"] = model
            save_state(config.state_file, state)
            raise RuntimeError(
                f"Training for {model} did not finish cleanly: exit_code={exit_code}, log={log_path}"
            )

        append_monitor_log(config, f"{model} 训练完成: {log_path}")
        next_index += 1
        state["next_index"] = next_index
        state.pop("current_model", None)
        state.pop("current_pid", None)
        save_state(config.state_file, state)

    state["completed"] = True
    state["completed_at"] = _now()
    if not config.dry_run:
        save_state(config.state_file, state)
    append_monitor_log(config, "自动训练队列全部完成")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="按队列顺序执行 SPAD 训练任务")
    parser.add_argument("--models", nargs="+", default=AUTO_TRAIN_MODELS, help="按顺序训练的模型队列")
    parser.add_argument("--epochs", type=int, default=100, help="每个模型的训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="每个模型的批大小")
    parser.add_argument("--poll-seconds", type=int, default=30, help="进程检查间隔秒数")
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable), help="训练使用的 Python 可执行文件路径")
    parser.add_argument("--data-root", type=Path, default=None, help="可选的数据根目录覆盖")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs" / "CLS", help="训练日志目录")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "CLS", help="Checkpoint 保存目录")
    parser.add_argument("--state-file", type=Path, default=STATE_FILE, help="监控状态 JSON 文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印训练队列, 不启动训练")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AutoTrainConfig(
        models=list(args.models),
        epochs=args.epochs,
        batch_size=args.batch_size,
        poll_seconds=args.poll_seconds,
        python_exe=args.python_exe,
        data_root=args.data_root,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        state_file=args.state_file,
        dry_run=args.dry_run,
    )
    run_auto_train(config)
    return 0


# 无参运行时使用的自动训练队列。需要调整顺序时只改这里。
AUTO_TRAIN_MODELS: list[str] = [
    "pointnet",
    "pointnet2",
    "pointnet2msg",
    "pointbert",
    "pointmae",
    "pointtransformer",
    "pointtransv2",
    "pointtransv3",
    "pointmlp",
    "pointmlpelite",
    "upp",
]

def main_without_cli() -> None:
    """无参运行入口：显式使用 AUTO_TRAIN_MODELS 队列。"""
    config = AutoTrainConfig(
        models=AUTO_TRAIN_MODELS,
        epochs=100,
        batch_size=32,
        poll_seconds=30,
        python_exe=Path(sys.executable),
        data_root=None,
        log_dir=PROJECT_ROOT / "logs" / "CLS",
        save_dir=PROJECT_ROOT / "checkpoints" / "CLS",
        state_file=STATE_FILE,
        monitor_log=MONITOR_LOG,
        dry_run=False,
    )
    run_auto_train(config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
