"""汇总 SPAD 消融训练、统一测试、两 seed 统计与配对差值。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\summarize_ablation.py --families core

无参数运行：允许缺失项并生成当前快照。正式落论文前请省略 ``--allow-incomplete``，
确保 A0--A3 × seed42/43 均有 100 epoch checkpoint 和统一无增强指标 JSON。
输出位于 ``outputs/ABL/summary``：逐 run CSV、mean/std、paired delta、Markdown 与 LaTeX。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

import torch

from scripts.ablation_registry import (
    EVAL_SEED,
    PROJECT_ROOT,
    SPLIT_CACHE_SHA256,
    SPLIT_SEED,
    AblationExperiment,
    expected_checkpoint_dir,
    expected_log_dir,
    expected_output_dir,
    select_experiments,
)
from scripts.train import build_model


SUMMARY_ROOT = PROJECT_ROOT / "outputs" / "ABL" / "summary"
UNIFIED_TEST_ROOT = PROJECT_ROOT / "outputs" / "ABL" / "unified_test"
TIMESTAMP_PATTERN = re.compile(
    r"(?m)^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)

METRIC_FIELDS = (
    "top1",
    "f1_macro",
    "box_z_mae",
    "box_center_mae",
    "mean_iou_matched_cls",
    "AP50",
    "AP@50:5:95",
)


@dataclass
class RunRecord:
    """逐 run 审计与结果记录。"""

    experiment_id: str
    family: str
    factor_id: str
    train_seed: int
    split_seed: int
    split_manifest_sha256: str
    model: str
    gcn_operator: str
    requested_box_head: str
    effective_box_head: str
    requested_seg_loss_weight: float
    effective_seg_loss_weight: float
    checkpoint_path: str
    checkpoint_epoch: int
    best_epoch: int
    train_log: str
    metrics_json: str
    parameter_count: int
    peak_vram_mb: Optional[float]
    train_seconds: Optional[float]
    eval_seed: Optional[int]
    augment_eval: Optional[bool]
    box_space: Optional[str]
    top1: Optional[float]
    f1_macro: Optional[float]
    box_z_mae: Optional[float]
    box_center_mae: Optional[float]
    mean_iou_matched_cls: Optional[float]
    AP50: Optional[float]
    ap_50_95: Optional[float]
    status: str
    issues: str


def parse_csv(value: str) -> List[str]:
    """解析逗号分隔 CLI 列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def factor_id(experiment_id: str) -> str:
    """从 A0_seed42、B1_*、C_lambda_* 提取分组因子编号。"""
    if experiment_id.startswith("A"):
        return experiment_id.split("_", 1)[0]
    if experiment_id.startswith("B"):
        return experiment_id.split("_", 1)[0]
    if experiment_id.startswith("C_lambda_"):
        return experiment_id.rsplit("_seed", 1)[0]
    return experiment_id


def find_checkpoint(experiment: AblationExperiment, kind: str) -> Optional[Path]:
    """查找 best/last checkpoint，历史复用资产优先。"""
    if kind not in {"best", "last"}:
        raise ValueError(f"Unsupported checkpoint kind: {kind}")
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


def find_train_log(experiment: AblationExperiment) -> Optional[Path]:
    """查找训练日志。"""
    if experiment.reuse_train_log is not None:
        return experiment.reuse_train_log if experiment.reuse_train_log.is_file() else None
    candidates = sorted(
        expected_log_dir(experiment).glob("train_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def metrics_matches_checkpoint(metrics_path: Path, checkpoint: Path) -> bool:
    """验证指标 JSON 是否由指定 checkpoint 生成。"""
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        recorded = Path(str(payload.get("checkpoint", ""))).resolve()
        return recorded == checkpoint.resolve()
    except (OSError, json.JSONDecodeError):
        return False


def unified_metrics_dirs(experiment: AblationExperiment) -> List[Path]:
    """返回当前规范目录和 2026-07-17 统一复测目录。

    主矩阵统一复测资产历史上写入 ``outputs/ABL/unified_test/<ID>``，
    新评估脚本则按 family 写入 ``expected_output_dir``。复用型实验（B0、
    lambda=0/0.5）还应复用锚点实验的统一测试结果。
    """
    experiment_ids = [experiment.experiment_id]
    if experiment.reuse_of:
        experiment_ids.append(experiment.reuse_of)
    directories = [expected_output_dir(experiment)]
    directories.extend(UNIFIED_TEST_ROOT / item for item in experiment_ids)
    return list(dict.fromkeys(directories))


def is_unified_metrics_path(
    experiment: AblationExperiment,
    metrics_path: Path,
) -> bool:
    """判断指标 JSON 是否位于任一认可的统一无增强测试目录。"""
    resolved = metrics_path.resolve()
    return any(root.resolve() in resolved.parents for root in unified_metrics_dirs(experiment))


def find_metrics(experiment: AblationExperiment, checkpoint: Path) -> Optional[Path]:
    """优先选择统一评估目录，缺失时回退到登记的历史无增强结果。"""
    candidates = sorted(
        (
            metrics_path
            for directory in unified_metrics_dirs(experiment)
            for metrics_path in directory.glob("metrics_*.json")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for metrics_path in candidates:
        if metrics_matches_checkpoint(metrics_path, checkpoint):
            return metrics_path
    legacy = experiment.reuse_test_metrics
    if legacy is not None and legacy.is_file() and metrics_matches_checkpoint(legacy, checkpoint):
        return legacy
    return None


def load_checkpoint(path: Path) -> Mapping[str, object]:
    """CPU 加载 checkpoint。"""
    return torch.load(path, map_location="cpu", weights_only=False)


def as_mapping(value: object) -> Mapping[str, object]:
    """把 checkpoint args 规范成 Mapping。"""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def infer_effective_config(
    experiment: AblationExperiment,
    checkpoint: Mapping[str, object],
) -> Tuple[str, float, List[str]]:
    """读取或从 state_dict 推断有效 head 与目标性权重。"""
    issues: List[str] = []
    args = as_mapping(checkpoint.get("args", {}))
    state_dict = checkpoint.get("model_state_dict", {})
    has_seg_head = isinstance(state_dict, Mapping) and any(
        "box_head.seg_mlp." in str(key) for key in state_dict
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
    effective_seg_raw = args.get("effective_seg_loss_weight")
    effective_seg = (
        float(effective_seg_raw)
        if effective_seg_raw is not None
        else (requested_seg if has_seg_head else 0.0)
    )
    if effective_head != experiment.box_head:
        issues.append(
            f"effective head mismatch: expected={experiment.box_head}, actual={effective_head}"
        )
    if not math.isclose(effective_seg, experiment.seg_loss_weight, abs_tol=1e-12):
        issues.append(
            "effective seg mismatch: "
            f"expected={experiment.seg_loss_weight}, actual={effective_seg}"
        )
    return effective_head, effective_seg, issues


def parameter_count(
    experiment: AblationExperiment,
    checkpoint: Mapping[str, object],
) -> int:
    """按 checkpoint 模型配置重建网络并统计可训练参数。"""
    args_dict = dict(as_mapping(checkpoint.get("args", {})))
    args_dict["model"] = experiment.model
    args_dict.setdefault("box_head", experiment.box_head)
    args_dict.setdefault("gcn_use_physical_branch", experiment.gcn_use_physical_branch)
    args_dict.setdefault("gcn_use_se_gate", experiment.gcn_use_se_gate)
    args_dict.setdefault("gcn_use_coord_residual", experiment.gcn_use_coord_residual)
    args_dict.setdefault("gcn_aggregation", experiment.gcn_aggregation)
    args_dict.setdefault("gcn_exclude_self", experiment.gcn_exclude_self)
    args_dict.setdefault("gcn_feature_residual", experiment.gcn_feature_residual)
    namespace = argparse.Namespace(**args_dict)
    class_to_idx = checkpoint.get("class_to_idx", {})
    num_classes = len(class_to_idx) if isinstance(class_to_idx, Mapping) else 26
    model = build_model(
        experiment.model,
        num_classes=num_classes or 26,
        project_root=PROJECT_ROOT,
        args=namespace,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def parse_train_seconds(path: Optional[Path]) -> Optional[float]:
    """从训练日志首尾时间戳计算墙钟秒数。"""
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    stamps = TIMESTAMP_PATTERN.findall(text)
    if len(stamps) < 2:
        return None
    start = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S,%f")
    end = datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S,%f")
    return (end - start).total_seconds()


def optional_float(payload: Mapping[str, object], key: str) -> Optional[float]:
    """读取可空浮点字段。"""
    value = payload.get(key)
    return None if value is None else float(value)


def build_run_record(experiment: AblationExperiment) -> RunRecord:
    """汇总单个实验 checkpoint、训练日志和测试 JSON。"""
    issues: List[str] = []
    best_path = find_checkpoint(experiment, "best")
    last_path = find_checkpoint(experiment, "last")
    train_log = find_train_log(experiment)
    if best_path is None:
        return RunRecord(
            experiment_id=experiment.experiment_id,
            family=experiment.family,
            factor_id=factor_id(experiment.experiment_id),
            train_seed=experiment.seed,
            split_seed=SPLIT_SEED,
            split_manifest_sha256=SPLIT_CACHE_SHA256,
            model=experiment.model,
            gcn_operator=experiment.gcn_operator,
            requested_box_head=experiment.box_head,
            effective_box_head="",
            requested_seg_loss_weight=experiment.seg_loss_weight,
            effective_seg_loss_weight=0.0,
            checkpoint_path="",
            checkpoint_epoch=0,
            best_epoch=0,
            train_log=str(train_log or ""),
            metrics_json="",
            parameter_count=0,
            peak_vram_mb=None,
            train_seconds=parse_train_seconds(train_log),
            eval_seed=None,
            augment_eval=None,
            box_space=None,
            top1=None,
            f1_macro=None,
            box_z_mae=None,
            box_center_mae=None,
            mean_iou_matched_cls=None,
            AP50=None,
            ap_50_95=None,
            status="missing_checkpoint",
            issues="best checkpoint missing",
        )

    best_checkpoint = load_checkpoint(best_path)
    last_checkpoint = load_checkpoint(last_path) if last_path is not None else best_checkpoint
    best_epoch = int(best_checkpoint.get("epoch", 0))
    checkpoint_epoch = int(last_checkpoint.get("epoch", 0))
    if checkpoint_epoch < 100:
        issues.append(f"last checkpoint epoch={checkpoint_epoch}, expected 100")

    args = as_mapping(best_checkpoint.get("args", {}))
    requested_head = str(args.get("requested_box_head", args.get("box_head", experiment.box_head)))
    requested_seg = float(
        args.get(
            "requested_seg_loss_weight",
            args.get("seg_loss_weight", experiment.seg_loss_weight),
        )
    )
    effective_head, effective_seg, config_issues = infer_effective_config(
        experiment, best_checkpoint
    )
    issues.extend(config_issues)
    effective_operator = str(args.get("gcn_operator", "sage"))
    if effective_operator != experiment.gcn_operator:
        issues.append(
            "effective operator mismatch: "
            f"expected={experiment.gcn_operator}, actual={effective_operator}"
        )

    metrics_path = find_metrics(experiment, best_path)
    metrics: Mapping[str, object] = {}
    if metrics_path is None:
        issues.append("uniform no-augment metrics missing")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not is_unified_metrics_path(experiment, metrics_path):
            issues.append("only legacy no-augment metrics available; unified evaluation not run")
        if metrics.get("augment_eval") is not False:
            issues.append("metrics do not explicitly record augment_eval=false")
        if metrics.get("eval_seed") != EVAL_SEED:
            issues.append(f"eval seed mismatch or missing: {metrics.get('eval_seed')}")
        if metrics.get("box_space") != "normalized":
            issues.append(f"box space mismatch or missing: {metrics.get('box_space')}")
        split_hash = metrics.get("split_cache_sha256")
        if split_hash != SPLIT_CACHE_SHA256:
            issues.append(f"split hash mismatch or missing: {split_hash}")
        for required_metric in ("box_z_mae", "box_center_mae"):
            if metrics.get(required_metric) is None:
                issues.append(f"required metric missing: {required_metric}")

    peak_vram = args.get("peak_vram_mb")
    status = "complete" if checkpoint_epoch >= 100 and metrics_path is not None and not issues else "incomplete"
    return RunRecord(
        experiment_id=experiment.experiment_id,
        family=experiment.family,
        factor_id=factor_id(experiment.experiment_id),
        train_seed=experiment.seed,
        split_seed=int(args.get("split_seed", SPLIT_SEED)),
        split_manifest_sha256=str(
            metrics.get("split_cache_sha256") or args.get("split_cache_sha256") or SPLIT_CACHE_SHA256
        ),
        model=experiment.model,
        gcn_operator=effective_operator,
        requested_box_head=requested_head,
        effective_box_head=effective_head,
        requested_seg_loss_weight=requested_seg,
        effective_seg_loss_weight=effective_seg,
        checkpoint_path=str(best_path),
        checkpoint_epoch=checkpoint_epoch,
        best_epoch=best_epoch,
        train_log=str(train_log or ""),
        metrics_json=str(metrics_path or ""),
        parameter_count=parameter_count(experiment, best_checkpoint),
        peak_vram_mb=None if peak_vram is None else float(peak_vram),
        train_seconds=parse_train_seconds(train_log),
        eval_seed=None if metrics.get("eval_seed") is None else int(metrics["eval_seed"]),
        augment_eval=None if metrics.get("augment_eval") is None else bool(metrics["augment_eval"]),
        box_space=None if metrics.get("box_space") is None else str(metrics["box_space"]),
        top1=optional_float(metrics, "top1"),
        f1_macro=optional_float(metrics, "f1_macro"),
        box_z_mae=optional_float(metrics, "box_z_mae"),
        box_center_mae=optional_float(metrics, "box_center_mae"),
        mean_iou_matched_cls=optional_float(metrics, "mean_iou_matched_cls"),
        AP50=optional_float(metrics, "AP50"),
        ap_50_95=optional_float(metrics, "AP@50:5:95"),
        status=status,
        issues="; ".join(issues),
    )


def record_metric(record: RunRecord, metric: str) -> Optional[float]:
    """统一处理 AP@50:5:95 的字段名差异。"""
    if metric == "AP@50:5:95":
        return record.ap_50_95
    return getattr(record, metric)


def summarize_groups(records: Sequence[RunRecord]) -> List[Dict[str, object]]:
    """按 factor_id 输出 count/mean/std。"""
    rows: List[Dict[str, object]] = []
    for group_id in dict.fromkeys(record.factor_id for record in records):
        group = [record for record in records if record.factor_id == group_id]
        row: Dict[str, object] = {
            "factor_id": group_id,
            "count": len(group),
            "complete_count": sum(record.status == "complete" for record in group),
        }
        for metric in METRIC_FIELDS:
            values = [
                value
                for record in group
                if record.status == "complete"
                and (value := record_metric(record, metric)) is not None
            ]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else (0.0 if len(values) == 1 else None)
        rows.append(row)
    return rows


def paired_deltas(records: Sequence[RunRecord]) -> List[Dict[str, object]]:
    """计算核心 A0--A3 的逐 seed 配对差值及 mean/std。"""
    comparisons = (
        ("backbone", "A1", "A0"),
        ("centroid_head", "A2", "A1"),
        ("objectness_bce", "A3", "A2"),
        ("full_method", "A3", "A0"),
    )
    core_seeds = sorted(
        {
            record.train_seed
            for record in records
            if record.family == "core" and record.factor_id in {"A0", "A1", "A2", "A3"}
        }
    )
    lookup = {(record.factor_id, record.train_seed): record for record in records}
    rows: List[Dict[str, object]] = []
    for name, upper, lower in comparisons:
        metric_values: Dict[str, List[float]] = {metric: [] for metric in METRIC_FIELDS}
        for seed in core_seeds:
            upper_record = lookup.get((upper, seed))
            lower_record = lookup.get((lower, seed))
            if (
                upper_record is None
                or lower_record is None
                or upper_record.status != "complete"
                or lower_record.status != "complete"
            ):
                continue
            row: Dict[str, object] = {
                "comparison": name,
                "upper": upper,
                "lower": lower,
                "seed": seed,
            }
            for metric in METRIC_FIELDS:
                upper_value = record_metric(upper_record, metric)
                lower_value = record_metric(lower_record, metric)
                delta = (
                    None
                    if upper_value is None or lower_value is None
                    else upper_value - lower_value
                )
                row[metric] = delta
                if delta is not None:
                    metric_values[metric].append(delta)
            rows.append(row)
        summary: Dict[str, object] = {
            "comparison": name,
            "upper": upper,
            "lower": lower,
            "seed": "mean±std",
        }
        for metric, values in metric_values.items():
            summary[metric] = statistics.fmean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else (0.0 if len(values) == 1 else None)
        rows.append(summary)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """写入 CSV；字段按首次出现顺序并集。"""
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: Optional[float], *, percent: bool = False) -> str:
    """Markdown/LaTeX 数值格式。"""
    if value is None:
        return "--"
    return f"{value * 100:.2f}" if percent else f"{value:.4f}"


def write_markdown(
    records: Sequence[RunRecord],
    groups: Sequence[Mapping[str, object]],
    deltas: Sequence[Mapping[str, object]],
) -> Path:
    """生成可审计 Markdown 摘要。"""
    path = SUMMARY_ROOT / "ablation_summary.md"
    lines = [
        "# SPAD 消融结果自动汇总",
        "",
        f"> 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"> Split seed/hash：{SPLIT_SEED} / `{SPLIT_CACHE_SHA256}`",
        f"> 正式测试：无增强，eval seed={EVAL_SEED}，1024 点。",
        "",
        "## 逐 run",
        "",
        "| ID | seed | operator | effective head | effective λobj | epoch | Top-1 | F1 | z-MAE | center-MAE | mIoU | AP50 | AP50:95 | 状态 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| {id} | {seed} | {operator} | {head} | {seg:.2f} | {epoch} | {top1} | {f1} | {z} | {center} | {iou} | {ap50} | {ap} | {status} |".format(
                id=record.experiment_id,
                seed=record.train_seed,
                operator=record.gcn_operator,
                head=record.effective_box_head or "--",
                seg=record.effective_seg_loss_weight if math.isfinite(record.effective_seg_loss_weight) else 0.0,
                epoch=record.checkpoint_epoch,
                top1=format_metric(record.top1, percent=True),
                f1=format_metric(record.f1_macro, percent=True),
                z=format_metric(record.box_z_mae),
                center=format_metric(record.box_center_mae),
                iou=format_metric(record.mean_iou_matched_cls),
                ap50=format_metric(record.AP50, percent=True),
                ap=format_metric(record.ap_50_95, percent=True),
                status=record.status,
            )
        )

    lines.extend(
        [
            "",
            "## 分组 mean ± std",
            "",
            "| ID | n | Top-1 (%) | F1 (%) | z-MAE | center-MAE | mIoU | AP50 (%) | AP50:95 (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups:
        def mean_std(metric: str, percent: bool = False) -> str:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            if mean is None or std is None:
                return "--"
            scale = 100.0 if percent else 1.0
            digits = 2 if percent else 4
            return f"{float(mean) * scale:.{digits}f} ± {float(std) * scale:.{digits}f}"

        lines.append(
            f"| {row['factor_id']} | {row['complete_count']}/{row['count']} | "
            f"{mean_std('top1', True)} | {mean_std('f1_macro', True)} | "
            f"{mean_std('box_z_mae')} | {mean_std('box_center_mae')} | "
            f"{mean_std('mean_iou_matched_cls')} | {mean_std('AP50', True)} | "
            f"{mean_std('AP@50:5:95', True)} |"
        )

    lines.extend(["", "## 核心配对差值", ""])
    for row in deltas:
        if row.get("seed") != "mean±std":
            continue
        lines.append(
            "- **{name} ({upper}-{lower})**：Top-1 {top1}，mIoU {iou}，AP50 {ap50}，AP50:95 {ap}.".format(
                name=row["comparison"],
                upper=row["upper"],
                lower=row["lower"],
                top1=format_metric(row.get("top1"), percent=True),
                iou=format_metric(row.get("mean_iou_matched_cls")),
                ap50=format_metric(row.get("AP50"), percent=True),
                ap=format_metric(row.get("AP@50:5:95"), percent=True),
            )
        )

    incomplete = [record for record in records if record.status != "complete"]
    lines.extend(["", "## 审计问题", ""])
    if not incomplete:
        lines.append("- 无。")
    else:
        for record in incomplete:
            lines.append(f"- `{record.experiment_id}`：{record.issues or record.status}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_latex(groups: Sequence[Mapping[str, object]]) -> Path:
    """生成核心 mean±std LaTeX 行，不覆盖论文源文件。"""
    path = SUMMARY_ROOT / "ablation_tables.tex"
    lines = [
        "% Auto-generated by scripts/summarize_ablation.py",
        "% Columns: ID, Top-1, Macro-F1, mIoU, AP50, AP50:95",
    ]
    for row in groups:
        if not str(row["factor_id"]).startswith("A"):
            continue
        values = []
        for metric in ("top1", "f1_macro", "mean_iou_matched_cls", "AP50", "AP@50:5:95"):
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            if mean is None or std is None:
                values.append("--")
            elif metric in {"top1", "f1_macro", "AP50", "AP@50:5:95"}:
                values.append(f"{float(mean) * 100:.2f} $\\pm$ {float(std) * 100:.2f}")
            else:
                values.append(f"{float(mean):.4f} $\\pm$ {float(std):.4f}")
        lines.append(f"{row['factor_id']} & " + " & ".join(values) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_summary(
    experiments: Iterable[AblationExperiment],
    *,
    allow_incomplete: bool,
) -> int:
    """生成所有汇总产物。"""
    records = [build_run_record(experiment) for experiment in experiments]
    group_rows = summarize_groups(records)
    delta_rows = paired_deltas(records) if any(record.family == "core" for record in records) else []

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    run_rows = [asdict(record) for record in records]
    write_csv(SUMMARY_ROOT / "ablation_runs.csv", run_rows)
    write_csv(SUMMARY_ROOT / "ablation_group_summary.csv", group_rows)
    write_csv(SUMMARY_ROOT / "ablation_paired_deltas.csv", delta_rows)
    markdown_path = write_markdown(records, group_rows, delta_rows)
    latex_path = write_latex(group_rows)
    audit_path = SUMMARY_ROOT / "ablation_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "split_seed": SPLIT_SEED,
                "split_cache_sha256": SPLIT_CACHE_SHA256,
                "eval_seed": EVAL_SEED,
                "records": run_rows,
                "group_summary": group_rows,
                "paired_deltas": delta_rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"runs_csv={SUMMARY_ROOT / 'ablation_runs.csv'}")
    print(f"markdown={markdown_path}")
    print(f"latex={latex_path}")

    incomplete = [record for record in records if record.status != "complete"]
    if incomplete and not allow_incomplete:
        print("Incomplete experiments:")
        for record in incomplete:
            print(f"  {record.experiment_id}: {record.issues or record.status}")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="汇总 SPAD 消融结果")
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
        help="可选，逗号分隔实验 ID",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="缺 checkpoint/统一测试时仍生成当前快照",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    experiments = select_experiments(args.families, args.experiments)
    return run_summary(experiments, allow_incomplete=args.allow_incomplete)


def main_without_cli() -> None:
    """无参数模式：生成允许缺失的 core 当前快照。"""
    run_summary(select_experiments(["core"]), allow_incomplete=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
