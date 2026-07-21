"""2026-07-16 历史 A0--A2/A3-seed43 队列；新实验请使用注册表驱动脚本。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\run_ablation_training_19h.py --execute --max-hours 18.75

无参数运行：仅打印历史队列 dry-run，不启动训练。A3-seed44 已从该自动队列移除；
核心协议固定为主计划中的 100 epoch、
1024 点、batch 32、无 AMP/TF32、训练/验证增强以及 seed42 固定划分缓存。
队列每个实验使用独立日志/checkpoint 目录，并在开始前、结束后校验 split cache SHA256。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch


PROJECT_ROOT = Path(r"D:\PYproject\SPAD")
DATA_ROOT = Path(r"D:\PYproject\SPADdata\2025-04-30-dpc")
SPLIT_CACHE = DATA_ROOT / ".split_cache.json"
EXPECTED_SPLIT_SHA256 = "AB94E67744AC3C73FC45A2D3E3E389773661E3EEBA85A6F8EF2C3025220A9F22"
QUEUE_ROOT = PROJECT_ROOT / "outputs" / "ABL" / "head" / "queue_20260716_19h"


@dataclass(frozen=True)
class Experiment:
    """一个可恢复的正式消融训练项。"""

    experiment_id: str
    seed: int
    model: str
    box_head: str
    seg_loss_weight: float
    estimated_hours: float


EXPERIMENTS: List[Experiment] = [
    Experiment("A2_seed42", 42, "graph_residual_gcn_ablation", "centroid", 0.0, 2.0),
    Experiment("A0_seed43", 43, "dgcnn", "mlp", 0.0, 2.833),
    Experiment("A1_seed43", 43, "graph_residual_gcn_ablation", "mlp", 0.0, 2.0),
    Experiment("A2_seed43", 43, "graph_residual_gcn_ablation", "centroid", 0.0, 2.0),
    Experiment("A3_seed43", 43, "graph_residual_gcn_ablation", "centroid", 0.5, 2.0),
    Experiment("A0_seed44", 44, "dgcnn", "mlp", 0.0, 2.833),
    Experiment("A1_seed44", 44, "graph_residual_gcn_ablation", "mlp", 0.0, 2.0),
    Experiment("A2_seed44", 44, "graph_residual_gcn_ablation", "centroid", 0.0, 2.0),
]


def sha256_file(path: Path) -> str:
    """返回文件 SHA256 大写十六进制摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_split_cache() -> str:
    """拒绝在划分缓存缺失或发生变化时继续训练。"""
    if not SPLIT_CACHE.is_file():
        raise FileNotFoundError(f"Split cache not found: {SPLIT_CACHE}")
    actual_hash = sha256_file(SPLIT_CACHE)
    if actual_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            "Split cache hash mismatch: "
            f"expected={EXPECTED_SPLIT_SHA256}, actual={actual_hash}, path={SPLIT_CACHE}"
        )
    return actual_hash


def select_within_budget(max_hours: float, reserve_hours: float) -> List[Experiment]:
    """按母版顺序选择可放入预算的完整实验，不拆分实验。"""
    usable_hours = max_hours - reserve_hours
    if usable_hours <= 0:
        raise ValueError("max_hours must be greater than reserve_hours")
    selected: List[Experiment] = []
    planned = 0.0
    for experiment in EXPERIMENTS:
        if planned + experiment.estimated_hours > usable_hours:
            continue
        selected.append(experiment)
        planned += experiment.estimated_hours
    return selected


def checkpoint_state(checkpoint_dir: Path) -> tuple[Optional[Path], int]:
    """查找最新 last checkpoint 并读取其完整 epoch。"""
    candidates = sorted(checkpoint_dir.glob("*_last.pth"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None, 0
    checkpoint_path = candidates[-1]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint_path, int(payload.get("epoch", 0))


def build_train_command(experiment: Experiment, resume: Optional[Path]) -> List[str]:
    """构建冻结协议训练命令；子进程复用当前固定解释器。"""
    log_dir = PROJECT_ROOT / "logs" / "ABL" / "head" / experiment.experiment_id
    save_dir = PROJECT_ROOT / "checkpoints" / "ABL" / "head" / experiment.experiment_id
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--data-root", str(DATA_ROOT),
        "--model", experiment.model,
        "--epochs", "100",
        "--batch-size", "32",
        "--grad-accum-steps", "1",
        "--num-aug", "3",
        "--num-points", "1024",
        "--lr", "0.001",
        "--min-lr", "0.00001",
        "--weight-decay", "0.0001",
        "--train-ratio", "0.6",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
        "--num-workers", "0",
        "--seed", str(experiment.seed),
        "--device", "cuda",
        "--label-mode", "raw",
        "--cls-loss-weight", "1.0",
        "--box-loss-weight", "10.0",
        "--no-auto-balance",
        "--label-smoothing", "0.1",
        "--box-head", experiment.box_head,
        "--seg-loss-weight", str(experiment.seg_loss_weight),
        "--ema-decay", "0.0",
        "--no-amp",
        "--no-tf32",
        "--augment-train",
        "--augment-eval",
        "--log-dir", str(log_dir),
        "--save-dir", str(save_dir),
    ]
    if experiment.model.startswith("graph_residual_gcn"):
        command.extend([
            "--gcn-aggregation", "max",
            "--gcn-exclude-self",
            "--gcn-feature-residual",
            "--gcn-coord-scale-init", "0.1",
            "--gcn-use-checkpoint",
            "--gcn-no-legacy-mode",
            "--gcn-use-physical-branch",
            "--gcn-use-se-gate",
            "--gcn-use-coord-residual",
        ])
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    """原子写入队列清单，便于中断后审计。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def run_queue(max_hours: float, reserve_hours: float, execute: bool) -> int:
    """打印或执行预算约束队列。"""
    split_hash = validate_split_cache()
    selected = select_within_budget(max_hours=max_hours, reserve_hours=reserve_hours)
    planned_hours = sum(item.estimated_hours for item in selected)
    omitted = [item.experiment_id for item in EXPERIMENTS if item not in selected]

    print(f"python={sys.executable}")
    print(f"split_cache_sha256={split_hash}")
    print(f"max_hours={max_hours:.3f} reserve_hours={reserve_hours:.3f}")
    print(f"planned_hours={planned_hours:.3f}")
    print("selected=" + ",".join(item.experiment_id for item in selected))
    print("omitted=" + ",".join(omitted))
    for item in selected:
        print("DRY-RUN", subprocess.list2cmdline(build_train_command(item, resume=None)))
    if not execute:
        return 0

    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = QUEUE_ROOT / "run_manifest.json"
    started_at = time.time()
    deadline = started_at + max_hours * 3600.0
    manifest: Dict[str, object] = {
        "queue_started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "data_root": str(DATA_ROOT),
        "split_cache": str(SPLIT_CACHE),
        "split_cache_sha256": split_hash,
        "max_hours": max_hours,
        "reserve_hours": reserve_hours,
        "planned_hours": planned_hours,
        "selected": [asdict(item) for item in selected],
        "omitted": omitted,
        "runs": [],
        "status": "running",
    }
    write_manifest(manifest_path, manifest)

    runs: List[Dict[str, object]] = manifest["runs"]  # type: ignore[assignment]
    for experiment in selected:
        validate_split_cache()
        checkpoint_dir = PROJECT_ROOT / "checkpoints" / "ABL" / "head" / experiment.experiment_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resume, completed_epoch = checkpoint_state(checkpoint_dir)
        if completed_epoch >= 100:
            runs.append({
                "experiment_id": experiment.experiment_id,
                "status": "skipped_complete",
                "checkpoint": str(resume),
                "epoch": completed_epoch,
            })
            write_manifest(manifest_path, manifest)
            continue

        remaining_seconds = deadline - time.time()
        if remaining_seconds <= reserve_hours * 3600.0:
            runs.append({"experiment_id": experiment.experiment_id, "status": "not_started_budget"})
            break

        command = build_train_command(experiment, resume=resume)
        console_log = QUEUE_ROOT / f"{experiment.experiment_id}_console.log"
        run_record: Dict[str, object] = {
            "experiment_id": experiment.experiment_id,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "resume": str(resume) if resume else None,
            "resume_epoch": completed_epoch,
            "command": command,
            "console_log": str(console_log),
        }
        runs.append(run_record)
        write_manifest(manifest_path, manifest)

        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with console_log.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write("\n=== " + run_record["started_at"] + " ===\n")
            stream.write(subprocess.list2cmdline(command) + "\n")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            run_record["pid"] = process.pid
            write_manifest(manifest_path, manifest)
            timed_out = False
            while process.poll() is None:
                if time.time() >= deadline - reserve_hours * 3600.0:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
                    break
                time.sleep(30)

        _, final_epoch = checkpoint_state(checkpoint_dir)
        run_record.update({
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "return_code": process.returncode,
            "final_epoch": final_epoch,
            "status": "timed_out" if timed_out else ("completed" if process.returncode == 0 else "failed"),
            "split_cache_sha256_after": validate_split_cache(),
        })
        write_manifest(manifest_path, manifest)
        if timed_out or process.returncode != 0:
            manifest["status"] = "stopped_on_timeout" if timed_out else "stopped_on_failure"
            write_manifest(manifest_path, manifest)
            return 124 if timed_out else int(process.returncode or 1)

    manifest["status"] = "completed_selected_queue"
    manifest["queue_finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_hours"] = (time.time() - started_at) / 3600.0
    write_manifest(manifest_path, manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="执行 19 小时内的 A0--A3 单卡训练队列")
    parser.add_argument("--execute", action="store_true", help="实际启动训练；省略时仅 dry-run")
    parser.add_argument("--max-hours", type=float, default=18.75, help="硬墙钟上限，默认 18.75 小时")
    parser.add_argument("--reserve-hours", type=float, default=0.5, help="停止前保留时间，默认 0.5 小时")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    return run_queue(args.max_hours, args.reserve_hours, args.execute)


def main_without_cli() -> None:
    """无参数模式：安全 dry-run。"""
    run_queue(max_hours=18.75, reserve_hours=0.5, execute=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
