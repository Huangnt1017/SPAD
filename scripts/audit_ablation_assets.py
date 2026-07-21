"""审计消融训练协议、split cache、代码哈希和 checkpoint 资产。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\audit_ablation_assets.py --families core

无参数运行：审计 core 并写入 ``outputs/ABL/audit``。该脚本只读训练资产，
不会启动训练或修改 ``.split_cache.json``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

import torch

from scripts.ablation_registry import (
    FIXED_PYTHON,
    PROJECT_ROOT,
    SPLIT_CACHE,
    SPLIT_CACHE_SHA256,
    AblationExperiment,
    expected_checkpoint_dir,
    select_experiments,
)


AUDIT_ROOT = PROJECT_ROOT / "outputs" / "ABL" / "audit"
CODE_FILES = (
    PROJECT_ROOT / "scripts" / "train.py",
    PROJECT_ROOT / "model" / "graph_res_GCN.py",
    PROJECT_ROOT / "model" / "graph_res_GCN_ablation.py",
    PROJECT_ROOT / "utils" / "loss.py",
    PROJECT_ROOT / "utils" / "data.py",
    PROJECT_ROOT / "utils" / "checkpoint.py",
    PROJECT_ROOT / "utils" / "heads.py",
    PROJECT_ROOT / "baseline" / "DGCNN.py",
    PROJECT_ROOT / "scripts" / "test.py",
    PROJECT_ROOT / "scripts" / "run_ablation_training_19h.py",
    PROJECT_ROOT / "scripts" / "ablation_registry.py",
    PROJECT_ROOT / "scripts" / "run_ablation_matrix.py",
    PROJECT_ROOT / "scripts" / "run_ablation_evaluation.py",
    PROJECT_ROOT / "scripts" / "summarize_ablation.py",
    PROJECT_ROOT / "scripts" / "update_ablation_docs.py",
)


def parse_csv(value: str) -> List[str]:
    """解析逗号分隔列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256_bytes(data: bytes) -> str:
    """返回字节串 SHA256。"""
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    """返回文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_output(*args: str) -> str:
    """读取 Git 信息；仓库不可用时返回错误文本。"""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def find_checkpoint(experiment: AblationExperiment, kind: str) -> Optional[Path]:
    """查找注册复用或新目录 checkpoint。"""
    reuse = (
        experiment.reuse_best_checkpoint
        if kind == "best"
        else experiment.reuse_last_checkpoint
    )
    if reuse is not None:
        return reuse if reuse.is_file() else None
    candidates = sorted(
        expected_checkpoint_dir(experiment).glob(f"*_{kind}.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def mapping(value: object) -> Mapping[str, object]:
    """规范 checkpoint args。"""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def audit_checkpoint(experiment: AblationExperiment) -> Dict[str, object]:
    """审计单个实验 checkpoint。"""
    best_path = find_checkpoint(experiment, "best")
    last_path = find_checkpoint(experiment, "last")
    if best_path is None:
        return {
            "experiment_id": experiment.experiment_id,
            "status": "missing_best",
        }
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = (
        torch.load(last_path, map_location="cpu", weights_only=False)
        if last_path is not None
        else best
    )
    args = mapping(best.get("args", {}))
    state = best.get("model_state_dict", {})
    has_seg_head = isinstance(state, Mapping) and any(
        "box_head.seg_mlp." in str(key) for key in state
    )
    effective_head = str(
        args.get("effective_box_head")
        or ("centroid" if has_seg_head else "mlp")
    )
    requested_seg = float(
        args.get(
            "requested_seg_loss_weight",
            args.get("seg_loss_weight", experiment.seg_loss_weight),
        )
    )
    effective_seg = args.get("effective_seg_loss_weight")
    if effective_seg is None:
        effective_seg = requested_seg if has_seg_head else 0.0
    return {
        "experiment_id": experiment.experiment_id,
        "status": "complete" if int(last.get("epoch", 0)) >= 100 else "partial",
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path or ""),
        "best_epoch": int(best.get("epoch", 0)),
        "last_epoch": int(last.get("epoch", 0)),
        "best_val_score": best.get("best_val_score"),
        "best_val_top1": best.get("best_val_top1"),
        "train_seed": args.get("seed"),
        "model_recorded": args.get("model"),
        "requested_box_head": args.get("requested_box_head", args.get("box_head")),
        "effective_box_head": effective_head,
        "requested_seg_loss_weight": requested_seg,
        "effective_seg_loss_weight": float(effective_seg),
        "has_seg_head": has_seg_head,
        "gcn_operator": args.get("gcn_operator", "sage"),
        "gcn_aggregation": args.get("gcn_aggregation"),
        "gcn_exclude_self": args.get("gcn_exclude_self"),
        "gcn_feature_residual": args.get("gcn_feature_residual"),
        "gcn_use_physical_branch": args.get("gcn_use_physical_branch", True),
        "gcn_use_se_gate": args.get("gcn_use_se_gate", True),
        "gcn_use_coord_residual": args.get("gcn_use_coord_residual", True),
        "best_sha256": sha256_file(best_path),
        "last_sha256": sha256_file(last_path) if last_path is not None else None,
    }


def build_split_manifest() -> Dict[str, object]:
    """把不可变旧 cache 转成带比例和成员哈希的审计 manifest。"""
    raw = SPLIT_CACHE.read_bytes()
    cache = json.loads(raw.decode("utf-8"))
    manifest: Dict[str, object] = {
        "source_path": str(SPLIT_CACHE),
        "source_sha256": sha256_bytes(raw),
        "expected_sha256": SPLIT_CACHE_SHA256,
        "fingerprint": cache.get("fingerprint"),
        "label_mode": cache.get("label_mode"),
        "split_seed": cache.get("seed"),
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
    }
    for split_name in ("train", "val", "test"):
        paths = cache.get(f"{split_name}_paths", [])
        canonical = json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        manifest[f"{split_name}_count"] = len(paths)
        manifest[f"{split_name}_paths_sha256"] = sha256_bytes(canonical)
    return manifest


def run_audit(experiments: Sequence[AblationExperiment]) -> Path:
    """执行并保存审计。"""
    if Path(sys.executable).resolve() != FIXED_PYTHON.resolve():
        raise RuntimeError(f"Expected fixed Python {FIXED_PYTHON}, got {sys.executable}")
    split_manifest = build_split_manifest()
    if split_manifest["source_sha256"] != SPLIT_CACHE_SHA256:
        raise RuntimeError("Split cache SHA256 mismatch")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "split_manifest": split_manifest,
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in CODE_FILES
            if path.is_file()
        },
        "checkpoints": [audit_checkpoint(experiment) for experiment in experiments],
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = AUDIT_ROOT / f"ablation_asset_audit_{timestamp}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest_path = AUDIT_ROOT / "ablation_asset_audit_latest.json"
    latest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    split_path = AUDIT_ROOT / "split_manifest_seed42.json"
    split_path.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"audit={output_path}")
    print(f"split_manifest={split_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI。"""
    parser = argparse.ArgumentParser(description="审计 SPAD 消融资产")
    parser.add_argument(
        "--families",
        type=parse_csv,
        default=["core"],
        help="逗号分隔：core,robustness,structure_core,structure_appendix,operator,lambda；structure 为兼容别名；默认 core",
    )
    parser.add_argument("--experiments", type=parse_csv, default=None, help="逗号分隔实验 ID")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    run_audit(select_experiments(args.families, args.experiments))
    return 0


def main_without_cli() -> None:
    """无参数模式：审计 core。"""
    run_audit(select_experiments(["core"]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
