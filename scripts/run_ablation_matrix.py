"""执行注册表驱动的 SPAD 消融训练队列。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\run_ablation_matrix.py --families core --run-tag dry_core_2seed

无参数运行：打印 core 矩阵 dry-run。正式训练固定 100 epoch、1024 点、batch 32、
FP32/TF32 off、训练/验证增强；默认跳过 epoch=100 的历史或新 checkpoint，部分运行从
``*_last.pth`` 恢复。每次执行写入独立 ``outputs/ABL/training_queues/<run-tag>``，
不会覆盖 2026-07-17 的 19 小时队列清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

import torch

from scripts.ablation_registry import (
    DATA_ROOT,
    FIXED_PYTHON,
    PROJECT_ROOT,
    SPLIT_CACHE,
    SPLIT_CACHE_SHA256,
    AblationExperiment,
    expected_checkpoint_dir,
    expected_log_dir,
    select_experiments,
)


def parse_csv(value: str) -> List[str]:
    """解析逗号分隔 CLI 列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    """返回文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_environment() -> None:
    """校验固定解释器、数据目录与不可变 split cache。"""
    actual_python = Path(sys.executable).resolve()
    if actual_python != FIXED_PYTHON.resolve():
        raise RuntimeError(f"Expected Python {FIXED_PYTHON}, got {actual_python}")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")
    if not SPLIT_CACHE.is_file():
        raise FileNotFoundError(f"Split cache not found: {SPLIT_CACHE}")
    actual_hash = sha256_file(SPLIT_CACHE)
    if actual_hash != SPLIT_CACHE_SHA256:
        raise RuntimeError(
            f"Split cache hash mismatch: expected={SPLIT_CACHE_SHA256}, actual={actual_hash}"
        )


def load_epoch(path: Path) -> int:
    """读取 checkpoint 完整 epoch。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", 0))


def checkpoint_state(experiment: AblationExperiment) -> Tuple[Optional[Path], int, bool]:
    """返回可恢复 last、epoch、是否为注册复用资产。"""
    if experiment.reuse_last_checkpoint is not None:
        path = experiment.reuse_last_checkpoint
        if path.is_file():
            return path, load_epoch(path), True
        return None, 0, True
    candidates = sorted(
        expected_checkpoint_dir(experiment).glob("*_last.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None, 0, False
    path = candidates[-1]
    return path, load_epoch(path), False


def bool_flag(value: bool, enabled: str, disabled: str) -> str:
    """把布尔配置映射成互斥 CLI 开关。"""
    return enabled if value else disabled


def build_train_command(
    experiment: AblationExperiment,
    resume: Optional[Path],
    *,
    batch_size: int,
    grad_accum_steps: int,
) -> List[str]:
    """构建冻结训练协议命令。"""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--data-root",
        str(DATA_ROOT),
        "--model",
        experiment.model,
        "--epochs",
        "100",
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum_steps),
        "--num-aug",
        "3",
        "--num-points",
        "1024",
        "--lr",
        "0.001",
        "--min-lr",
        "0.00001",
        "--weight-decay",
        "0.0001",
        "--train-ratio",
        "0.6",
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.2",
        "--num-workers",
        "0",
        "--seed",
        str(experiment.seed),
        "--device",
        "cuda",
        "--label-mode",
        "raw",
        "--cls-loss-weight",
        "1.0",
        "--box-loss-weight",
        "10.0",
        "--no-auto-balance",
        "--label-smoothing",
        "0.1",
        "--box-head",
        experiment.box_head,
        "--seg-loss-weight",
        str(experiment.seg_loss_weight),
        "--ema-decay",
        "0.0",
        "--no-amp",
        "--no-tf32",
        "--augment-train",
        "--augment-eval",
        "--log-dir",
        str(expected_log_dir(experiment)),
        "--save-dir",
        str(expected_checkpoint_dir(experiment)),
    ]
    if experiment.is_gcn:
        command.extend(
            [
                "--gcn-operator",
                experiment.gcn_operator,
                "--gcn-aggregation",
                experiment.gcn_aggregation,
                bool_flag(
                    experiment.gcn_exclude_self,
                    "--gcn-exclude-self",
                    "--gcn-include-self",
                ),
                bool_flag(
                    experiment.gcn_feature_residual,
                    "--gcn-feature-residual",
                    "--gcn-no-feature-residual",
                ),
                "--gcn-coord-scale-init",
                "0.1",
                "--gcn-use-checkpoint",
                "--gcn-no-legacy-mode",
                bool_flag(
                    experiment.gcn_use_physical_branch,
                    "--gcn-use-physical-branch",
                    "--gcn-no-physical-branch",
                ),
                bool_flag(
                    experiment.gcn_use_se_gate,
                    "--gcn-use-se-gate",
                    "--gcn-no-se-gate",
                ),
                bool_flag(
                    experiment.gcn_use_coord_residual,
                    "--gcn-use-coord-residual",
                    "--gcn-no-coord-residual",
                ),
            ]
        )
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def process_exists(pid: int) -> bool:
    """跨平台检查 PID 是否仍存在。"""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            return f'"{pid}"' in completed.stdout
        import os
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def wait_for_pid(pid: int) -> None:
    """等待已有训练队列退出，避免单卡并发。"""
    if pid <= 0:
        return
    print(f"waiting_for_pid={pid}", flush=True)
    while process_exists(pid):
        time.sleep(30)
    print(f"wait_complete_pid={pid}", flush=True)


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    """原子写入训练队列清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def run_queue(
    experiments: Iterable[AblationExperiment],
    *,
    execute: bool,
    run_tag: str,
    wait_pid: int,
    max_hours: float,
    reserve_hours: float,
    batch_size: int,
    grad_accum_steps: int,
) -> int:
    """串行执行注册表实验。"""
    validate_environment()
    selected = list(experiments)
    if max_hours < 0 or reserve_hours < 0:
        raise ValueError("max_hours and reserve_hours must be non-negative")
    if batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch_size and grad_accum_steps must be positive")
    if max_hours > 0 and reserve_hours >= max_hours:
        raise ValueError("reserve_hours must be smaller than max_hours")

    queue_root = PROJECT_ROOT / "outputs" / "ABL" / "training_queues" / run_tag
    manifest_path = queue_root / "run_manifest.json"
    records: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {
        "queue_started_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "python": sys.executable,
        "data_root": str(DATA_ROOT),
        "split_cache": str(SPLIT_CACHE),
        "split_cache_sha256": SPLIT_CACHE_SHA256,
        "selected": [asdict(item) for item in selected],
        "max_hours": max_hours,
        "reserve_hours": reserve_hours,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "wait_pid": wait_pid,
        "runs": records,
        "status": "dry_run" if not execute else "waiting",
    }

    for experiment in selected:
        resume, epoch, reused = checkpoint_state(experiment)
        command_resume = resume if 0 < epoch < 100 and not reused else None
        command = build_train_command(experiment, command_resume, batch_size=batch_size, grad_accum_steps=grad_accum_steps)
        status = "skipped_complete" if epoch >= 100 else "planned"
        records.append(
            {
                "experiment_id": experiment.experiment_id,
                "checkpoint": str(resume) if resume else None,
                "checkpoint_epoch": epoch,
                "reused_asset": reused,
                "status": status,
                "command": command,
            }
        )
        print(status.upper(), experiment.experiment_id, subprocess.list2cmdline(command))
    if not execute:
        return 0

    queue_root.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, manifest)
    wait_for_pid(wait_pid)
    started = time.time()
    deadline = started + max_hours * 3600.0 if max_hours > 0 else None
    manifest["status"] = "running"
    manifest["execution_started_at"] = datetime.now().isoformat(timespec="seconds")
    write_manifest(manifest_path, manifest)

    for experiment, record in zip(selected, records):
        validate_environment()
        resume, epoch, reused = checkpoint_state(experiment)
        if epoch >= 100:
            record.update(
                status="skipped_complete",
                checkpoint=str(resume) if resume else None,
                checkpoint_epoch=epoch,
                reused_asset=reused,
            )
            write_manifest(manifest_path, manifest)
            continue
        if deadline is not None and time.time() >= deadline - reserve_hours * 3600.0:
            record["status"] = "not_started_budget"
            manifest["status"] = "stopped_on_budget"
            write_manifest(manifest_path, manifest)
            return 124

        command = build_train_command(experiment, resume if epoch > 0 else None, batch_size=batch_size, grad_accum_steps=grad_accum_steps)
        expected_checkpoint_dir(experiment).mkdir(parents=True, exist_ok=True)
        expected_log_dir(experiment).mkdir(parents=True, exist_ok=True)
        console_log = queue_root / f"{experiment.experiment_id}_console.log"
        record.update(
            status="running",
            started_at=datetime.now().isoformat(timespec="seconds"),
            resume=str(resume) if resume else None,
            resume_epoch=epoch,
            command=command,
            console_log=str(console_log),
        )
        write_manifest(manifest_path, manifest)

        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with console_log.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write("\n=== " + str(record["started_at"]) + " ===\n")
            stream.write(subprocess.list2cmdline(command) + "\n")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            record["pid"] = process.pid
            write_manifest(manifest_path, manifest)
            timed_out = False
            while process.poll() is None:
                if deadline is not None and time.time() >= deadline - reserve_hours * 3600.0:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
                    break
                time.sleep(30)

        final_checkpoint, final_epoch, _ = checkpoint_state(experiment)
        record.update(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            return_code=process.returncode,
            final_checkpoint=str(final_checkpoint) if final_checkpoint else None,
            final_epoch=final_epoch,
            status=(
                "timed_out"
                if timed_out
                else ("completed" if process.returncode == 0 and final_epoch >= 100 else "failed")
            ),
        )
        write_manifest(manifest_path, manifest)
        if record["status"] != "completed":
            manifest["status"] = "stopped_on_failure"
            write_manifest(manifest_path, manifest)
            return 124 if timed_out else int(process.returncode or 1)

    manifest["status"] = "completed"
    manifest["queue_finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_hours"] = (time.time() - started) / 3600.0
    write_manifest(manifest_path, manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI。"""
    parser = argparse.ArgumentParser(description="注册表驱动的 SPAD 消融训练队列")
    parser.add_argument("--execute", action="store_true", help="实际训练；省略时 dry-run")
    parser.add_argument(
        "--families",
        type=parse_csv,
        default=["core"],
        help="逗号分隔：core,robustness,structure_core,structure_appendix,operator,lambda；structure 为两类结构实验的兼容别名；默认 core",
    )
    parser.add_argument("--experiments", type=parse_csv, default=None, help="逗号分隔实验 ID")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=datetime.now().strftime("queue_%Y%m%d_%H%M%S"),
        help="独立队列输出目录名",
    )
    parser.add_argument("--wait-for-pid", type=int, default=0, help="启动前等待指定 PID 退出")
    parser.add_argument("--max-hours", type=float, default=0.0, help="0 表示不限墙钟")
    parser.add_argument("--reserve-hours", type=float, default=0.25, help="限时模式停止余量")
    parser.add_argument("--batch-size", type=int, default=32, help="物理 batch，正式协议默认 32")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="梯度累计；effective batch=batch×accum")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    experiments = select_experiments(args.families, args.experiments)
    return run_queue(
        experiments,
        execute=args.execute,
        run_tag=args.run_tag,
        wait_pid=args.wait_for_pid,
        max_hours=args.max_hours,
        reserve_hours=args.reserve_hours,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )


def main_without_cli() -> None:
    """无参数模式：安全 dry-run core 矩阵。"""
    run_queue(
        select_experiments(["core"]),
        execute=False,
        run_tag="dry_run_core",
        wait_pid=0,
        max_hours=0.0,
        reserve_hours=0.25,
        batch_size=32,
        grad_accum_steps=1,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
