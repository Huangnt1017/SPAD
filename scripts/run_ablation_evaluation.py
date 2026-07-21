"""统一执行消融 checkpoint 的无增强测试。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\run_ablation_evaluation.py --execute --families core

无参数运行：打印 core A0--A3 × seed42/43 的 dry-run，不启动 GPU 测试。
所有正式测试固定 ``eval_seed=42``、``num_points=1024``、``box_space=normalized``，
每个实验写入独立日志/输出目录，并生成可恢复审计的 evaluation_manifest.json。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

import torch

from scripts.ablation_registry import (
    DATA_ROOT,
    EVAL_SEED,
    FIXED_PYTHON,
    PROJECT_ROOT,
    AblationExperiment,
    expected_checkpoint_dir,
    expected_log_dir,
    expected_output_dir,
    select_experiments,
)


MANIFEST_PATH = PROJECT_ROOT / "outputs" / "ABL" / "evaluation_manifest.json"


def parse_csv(value: str) -> List[str]:
    """解析逗号分隔 CLI 列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_interpreter() -> None:
    """拒绝使用 PATH 中的错误 Python 环境。"""
    actual = Path(sys.executable).resolve()
    expected = FIXED_PYTHON.resolve()
    if actual != expected:
        raise RuntimeError(f"Expected fixed Python {expected}, got {actual}")


def process_exists(pid: int) -> bool:
    """检查 PID 是否仍存在。"""
    if pid <= 0:
        return False
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f'"{pid}"' in completed.stdout


def wait_for_pid(pid: int) -> None:
    """等待训练队列退出，避免测试与训练争用单卡。"""
    if pid <= 0:
        return
    print(f"waiting_for_pid={pid}", flush=True)
    while process_exists(pid):
        time.sleep(30)
    print(f"wait_complete_pid={pid}", flush=True)


def checkpoint_epoch(path: Path) -> int:
    """读取 checkpoint epoch。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", 0))


def find_best_checkpoint(experiment: AblationExperiment) -> Optional[Path]:
    """优先使用注册的历史资产，否则查找实验目录最新 best checkpoint。"""
    if experiment.reuse_best_checkpoint is not None:
        if experiment.reuse_best_checkpoint.is_file():
            return experiment.reuse_best_checkpoint
        return None
    candidates = sorted(
        expected_checkpoint_dir(experiment).glob("*_best.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def latest_metrics(output_dir: Path, checkpoint: Path) -> Optional[Path]:
    """返回与指定 checkpoint 匹配的最新指标 JSON。"""
    for metrics_path in sorted(
        output_dir.glob("metrics_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recorded = Path(str(payload.get("checkpoint", "")))
        try:
            if recorded.resolve() == checkpoint.resolve():
                return metrics_path
        except OSError:
            continue
    return None


def build_test_command(
    experiment: AblationExperiment,
    checkpoint: Path,
    batch_size: int,
    device: str,
    eval_seed: int,
) -> List[str]:
    """构建冻结协议的无增强测试命令。"""
    output_dir = expected_output_dir(experiment)
    if eval_seed != EVAL_SEED:
        output_dir = output_dir.parent / f"test_noaug_eval_seed{eval_seed}"
    log_dir = expected_log_dir(experiment) / f"test_noaug_eval_seed{eval_seed}"
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "test.py"),
        "--data-root",
        str(DATA_ROOT),
        "--checkpoint",
        str(checkpoint),
        "--model",
        experiment.model,
        "--batch-size",
        str(batch_size),
        "--num-points",
        "1024",
        "--train-ratio",
        "0.6",
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.2",
        "--num-workers",
        "0",
        "--seed",
        str(eval_seed),
        "--label-mode",
        "raw",
        "--no-augment-eval",
        "--device",
        device,
        "--box-space",
        "normalized",
        "--log-dir",
        str(log_dir),
        "--output-dir",
        str(output_dir),
    ]


def write_manifest(payload: Dict[str, object]) -> None:
    """原子更新统一测试清单。"""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(MANIFEST_PATH)


def run_evaluation_queue(
    experiments: Iterable[AblationExperiment],
    *,
    execute: bool,
    force: bool,
    batch_size: int,
    device: str,
    eval_seed: int,
    wait_pid: int,
) -> int:
    """串行测试所选实验；默认只 dry-run。"""
    validate_interpreter()
    selected = list(experiments)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    records: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "data_root": str(DATA_ROOT),
        "eval_seed": eval_seed,
        "batch_size": batch_size,
        "device": device,
        "execute": execute,
        "wait_pid": wait_pid,
        "selected": [item.experiment_id for item in selected],
        "runs": records,
        "status": "dry_run" if not execute else "running",
    }

    if execute:
        wait_for_pid(wait_pid)

    for experiment in selected:
        checkpoint = find_best_checkpoint(experiment)
        if checkpoint is None:
            record = {
                "experiment_id": experiment.experiment_id,
                "status": "missing_checkpoint",
            }
            records.append(record)
            print(f"MISSING {experiment.experiment_id}: best checkpoint not found")
            continue

        output_dir = expected_output_dir(experiment)
        if eval_seed != EVAL_SEED:
            output_dir = output_dir.parent / f"test_noaug_eval_seed{eval_seed}"
        existing_metrics = latest_metrics(output_dir, checkpoint)
        command = build_test_command(
            experiment,
            checkpoint,
            batch_size=batch_size,
            device=device,
            eval_seed=eval_seed,
        )
        print("DRY-RUN", subprocess.list2cmdline(command))

        record: Dict[str, object] = {
            "experiment_id": experiment.experiment_id,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": checkpoint_epoch(checkpoint),
            "command": command,
            "output_dir": str(output_dir),
        }
        records.append(record)

        if existing_metrics is not None and not force:
            record.update(
                status="skipped_existing",
                metrics_json=str(existing_metrics),
            )
            continue
        if not execute:
            record["status"] = "planned"
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        console_log = output_dir / "evaluation_console.log"
        record.update(
            status="running",
            started_at=datetime.now().isoformat(timespec="seconds"),
            console_log=str(console_log),
        )
        write_manifest(manifest)

        with console_log.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write("\n=== " + str(record["started_at"]) + " ===\n")
            stream.write(subprocess.list2cmdline(command) + "\n")
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )

        metrics_path = latest_metrics(output_dir, checkpoint)
        record.update(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            return_code=completed.returncode,
            metrics_json=str(metrics_path) if metrics_path else None,
            status="completed" if completed.returncode == 0 and metrics_path else "failed",
        )
        write_manifest(manifest)
        if record["status"] != "completed":
            manifest["status"] = "stopped_on_failure"
            write_manifest(manifest)
            return int(completed.returncode or 1)

    missing = [item for item in records if item.get("status") == "missing_checkpoint"]
    if execute and missing:
        manifest["status"] = "incomplete_missing_checkpoints"
        write_manifest(manifest)
        return 2
    manifest["status"] = "completed" if execute else "dry_run"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_manifest(manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="统一执行 SPAD 消融无增强测试")
    parser.add_argument("--execute", action="store_true", help="实际执行；省略时仅 dry-run")
    parser.add_argument(
        "--families",
        type=parse_csv,
        default=["core"],
        help="逗号分隔：core,robustness,structure_core,structure_appendix,operator,lambda；structure 为兼容别名；默认 core",
    )
    parser.add_argument(
        "--experiments",
        type=parse_csv,
        default=None,
        help="可选，逗号分隔实验 ID；只运行指定项",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="测试 batch，默认 32")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED, help="点采样评估种子，正式值 42")
    parser.add_argument("--force", action="store_true", help="已有匹配指标 JSON 时仍重跑")
    parser.add_argument("--wait-for-pid", type=int, default=0, help="测试前等待指定训练 PID 退出")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    experiments = select_experiments(args.families, args.experiments)
    return run_evaluation_queue(
        experiments,
        execute=args.execute,
        force=args.force,
        batch_size=args.batch_size,
        device=args.device,
        eval_seed=args.eval_seed,
        wait_pid=args.wait_for_pid,
    )


def main_without_cli() -> None:
    """无参数模式：安全打印 core 测试 dry-run。"""
    run_evaluation_queue(
        select_experiments(["core"]),
        execute=False,
        force=False,
        batch_size=32,
        device="cuda",
        eval_seed=EVAL_SEED,
        wait_pid=0,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
