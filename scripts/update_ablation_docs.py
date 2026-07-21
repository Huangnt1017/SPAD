"""把消融自动汇总同步回两个长期维护 Markdown。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\update_ablation_docs.py

默认要求 core 8 个 run 均为 complete；任何 checkpoint、统一无增强测试或审计字段
缺失时拒绝覆盖结果区。训练尚未结束时可用 ``--allow-incomplete`` 刷新进度快照。
只替换 ``ABLATION_CORE_RESULTS_START/END`` 标记之间的内容，不改动人工说明。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

from scripts.ablation_registry import (
    CORE_EXPERIMENTS,
    EVAL_SEED,
    PROJECT_ROOT,
    SPLIT_CACHE_SHA256,
)


SUMMARY_JSON = PROJECT_ROOT / "outputs" / "ABL" / "summary" / "ablation_audit.json"
TRAINING_RESULTS_MD = PROJECT_ROOT / "logs" / "training_results_since_202607.md"
MASTER_PLAN_MD = PROJECT_ROOT / "model" / "ablation_training_master_plan.md"
START_MARKER = "<!-- ABLATION_CORE_RESULTS_START -->"
END_MARKER = "<!-- ABLATION_CORE_RESULTS_END -->"
CORE_RUN_COUNT = len(CORE_EXPERIMENTS)


def load_summary() -> Mapping[str, object]:
    """读取 summarize_ablation.py 的审计 JSON。"""
    if not SUMMARY_JSON.is_file():
        raise FileNotFoundError(
            f"Summary not found: {SUMMARY_JSON}. Run scripts/summarize_ablation.py first."
        )
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def replace_marked_block(path: Path, block: str) -> None:
    """原子替换 Markdown 标记区。"""
    text = path.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(f"Result markers missing or invalid in {path}")
    end += len(END_MARKER)
    replacement = f"{START_MARKER}\n{block.rstrip()}\n{END_MARKER}"
    updated = text[:start] + replacement + text[end:]
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(updated, encoding="utf-8")
    temp_path.replace(path)


def value(record: Mapping[str, object], key: str) -> Optional[float]:
    """读取可空浮点。"""
    raw = record.get(key)
    return None if raw is None or raw == "" else float(raw)


def fmt(raw: Optional[float], *, percent: bool = False, digits: int = 4) -> str:
    """格式化指标。"""
    if raw is None or not math.isfinite(raw):
        return "--"
    if percent:
        return f"{raw * 100:.2f}%"
    return f"{raw:.{digits}f}"


def duration(seconds: object) -> str:
    """秒数转 H:MM:SS。"""
    if seconds is None or seconds == "":
        return "--"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def status_label(status: object) -> str:
    """把内部状态转成文档用中文。"""
    return {
        "complete": "完成",
        "incomplete": "待统一测试/审计",
        "missing_checkpoint": "缺 checkpoint",
    }.get(str(status), str(status))


def humanize_issues(issues: object) -> str:
    """把常见严格审计错误转换成简洁中文。"""
    text = str(issues or "")
    replacements = {
        "only legacy no-augment metrics available; unified evaluation not run": "仅有历史无增强结果，尚未统一复测",
        "metrics do not explicitly record augment_eval=false": "未显式记录 augment_eval=false",
        "eval seed mismatch or missing: None": "未记录 eval seed",
        "box space mismatch or missing: None": "未记录 box 坐标空间",
        "split hash mismatch or missing: None": "未记录 split hash",
        "required metric missing: box_z_mae": "缺 box_z_mae",
        "required metric missing: box_center_mae": "缺 box_center_mae",
        "uniform no-augment metrics missing": "缺统一无增强测试 JSON",
        "best checkpoint missing": "缺 best checkpoint",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("last checkpoint epoch=", "last checkpoint epoch=")
    text = text.replace(", expected 100", "，应为 100")
    return text


def grouped_lookup(rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    """按 factor_id 建立汇总索引。"""
    return {str(row["factor_id"]): row for row in rows}


def mean_std(
    row: Mapping[str, object],
    metric: str,
    *,
    percent: bool = False,
) -> str:
    """格式化 mean ± std。"""
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None or std is None:
        return "--"
    scale = 100.0 if percent else 1.0
    digits = 2 if percent else 4
    suffix = "%" if percent else ""
    return (
        f"{float(mean) * scale:.{digits}f} ± "
        f"{float(std) * scale:.{digits}f}{suffix}"
    )


def paired_summary(rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    """只保留 paired delta 的 mean±std 行。"""
    return [row for row in rows if row.get("seed") == "mean±std"]


def build_block(summary: Mapping[str, object]) -> str:
    """生成两个 Markdown 共用的核心结果区。"""
    records = [
        record
        for record in summary.get("records", [])
        if isinstance(record, Mapping) and record.get("family") == "core"
    ]
    groups = [
        row
        for row in summary.get("group_summary", [])
        if isinstance(row, Mapping) and str(row.get("factor_id", "")).startswith("A")
    ]
    deltas = paired_summary(
        [row for row in summary.get("paired_deltas", []) if isinstance(row, Mapping)]
    )
    complete_count = sum(record.get("status") == "complete" for record in records)
    training_complete_count = sum(int(record.get("checkpoint_epoch") or 0) >= 100 for record in records)
    is_complete = len(records) == CORE_RUN_COUNT and complete_count == CORE_RUN_COUNT
    title = (
        "### A0--A3 两 seed 最终结果"
        if is_complete
        else "### A0--A3 当前进度快照"
    )
    lines = [
        title,
        "",
        f"> 同步时间：{datetime.now().isoformat(timespec='seconds')}  ",
        f"> 训练完成：{training_complete_count}/{len(records)}；统一测试与完整审计：{complete_count}/{len(records)}；split SHA256 `{SPLIT_CACHE_SHA256}`；eval seed={EVAL_SEED}、无增强。",
        "",
        "| ID | seed | 有效 head | 有效 λobj | epoch | best epoch | 训练耗时 | Top-1 | F1 | z-MAE | center-MAE | mIoU | AP50 | AP50:95 | 状态 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        experiment_id = str(record.get("experiment_id", ""))
        document_status = status_label(record.get("status", ""))
        lines.append(
            "| {id} | {seed} | {head} | {seg:.2f} | {epoch} | {best} | {train_time} | {top1} | {f1} | {z} | {center} | {iou} | {ap50} | {ap} | {status} |".format(
                id=experiment_id,
                seed=record.get("train_seed", ""),
                head=record.get("effective_box_head") or "--",
                seg=float(record.get("effective_seg_loss_weight") or 0.0),
                epoch=record.get("checkpoint_epoch", 0),
                best=record.get("best_epoch", 0),
                train_time=duration(record.get("train_seconds")),
                top1=fmt(value(record, "top1"), percent=True),
                f1=fmt(value(record, "f1_macro"), percent=True),
                z=fmt(value(record, "box_z_mae"), digits=6),
                center=fmt(value(record, "box_center_mae"), digits=6),
                iou=fmt(value(record, "mean_iou_matched_cls")),
                ap50=fmt(value(record, "AP50"), percent=True),
                ap=fmt(value(record, "ap_50_95"), percent=True),
                status=document_status,
            )
        )

    lines.extend(
        [
            "",
            "#### 两 seed mean ± std",
            "",
            "| ID | n | Top-1 | F1 | z-MAE | center-MAE | mIoU | AP50 | AP50:95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lookup = grouped_lookup(groups)
    for factor in ("A0", "A1", "A2", "A3"):
        row = lookup.get(factor, {"factor_id": factor, "count": 0, "complete_count": 0})
        lines.append(
            f"| {factor} | {row.get('complete_count', 0)}/{row.get('count', 0)} | "
            f"{mean_std(row, 'top1', percent=True)} | "
            f"{mean_std(row, 'f1_macro', percent=True)} | "
            f"{mean_std(row, 'box_z_mae')} | "
            f"{mean_std(row, 'box_center_mae')} | "
            f"{mean_std(row, 'mean_iou_matched_cls')} | "
            f"{mean_std(row, 'AP50', percent=True)} | "
            f"{mean_std(row, 'AP@50:5:95', percent=True)} |"
        )

    lines.extend(["", "#### 同 seed 配对差值（上组减下组）", ""])
    if not deltas:
        lines.append("- 尚无完整 paired delta。")
    else:
        for row in deltas:
            lines.append(
                "- **{name} ({upper}-{lower})**：Top-1 {top1}，F1 {f1}，z-MAE {z}，center-MAE {center}，mIoU {iou}，AP50 {ap50}，AP50:95 {ap}.".format(
                    name=row.get("comparison"),
                    upper=row.get("upper"),
                    lower=row.get("lower"),
                    top1=fmt(value(row, "top1"), percent=True),
                    f1=fmt(value(row, "f1_macro"), percent=True),
                    z=fmt(value(row, "box_z_mae"), digits=6),
                    center=fmt(value(row, "box_center_mae"), digits=6),
                    iou=fmt(value(row, "mean_iou_matched_cls")),
                    ap50=fmt(value(row, "AP50"), percent=True),
                    ap=fmt(value(row, "AP@50:5:95"), percent=True),
                )
            )

    incomplete = [record for record in records if record.get("status") != "complete"]
    if incomplete:
        lines.extend(["", "#### 尚未满足最终汇总的项目", ""])
        for record in incomplete:
            issue_text = humanize_issues(record.get("issues") or record.get("status"))
            lines.append(f"- `{record.get('experiment_id')}`：{issue_text}")
    else:
        lines.extend(
            [
                "",
                "8 个核心 run 已全部通过 checkpoint、统一无增强测试、有效配置和 split hash 审计。",
            ]
        )
    return "\n".join(lines)


def finalize_static_text(*, is_complete: bool) -> None:
    """最终 8/8 完成时同步母版静态状态和完成勾选。"""
    if not is_complete:
        return

    master = MASTER_PLAN_MD.read_text(encoding="utf-8")
    status_lines = [line for line in master.splitlines() if line.startswith("> 当前状态：")]
    if status_lines:
        new_status = (
            "> 当前状态：**核心 A0--A3 已完成 8/8 次 100 epoch 训练，并完成 "
            "8/8 统一无增强测试、两 seed 汇总和 paired delta；seed44 只作为额外稳健性资产，"
            "不进入主 mean/std。**"
        )
        master = master.replace(status_lines[0], new_status, 1)

    replacements = {
        "- [ ] A0--A3 均有 seed42/43 两次 100 epoch 正式训练；":
            "- [x] A0--A3 均有 seed42/43 两次 100 epoch 正式训练；",
        "- [ ] 8 个 core best checkpoint 全部完成统一无增强测试；":
            "- [x] 8 个 core best checkpoint 全部完成统一无增强测试；",
        "- [ ] 输出两 seed mean/std 和同 seed paired delta；":
            "- [x] 输出两 seed mean/std 和同 seed paired delta；",
        "- [ ] 将最终结果同步到 `logs/training_results_since_202607.md`；":
            "- [x] 将最终结果同步到 `logs/training_results_since_202607.md`；",
    }
    for old, new in replacements.items():
        master = master.replace(old, new)
    MASTER_PLAN_MD.write_text(master, encoding="utf-8")


def update_docs(*, allow_incomplete: bool) -> int:
    """同步文档；完整模式拒绝不完整结果。"""
    summary = load_summary()
    records = [
        record
        for record in summary.get("records", [])
        if isinstance(record, Mapping) and record.get("family") == "core"
    ]
    incomplete = [record for record in records if record.get("status") != "complete"]
    if (len(records) != CORE_RUN_COUNT or incomplete) and not allow_incomplete:
        details = ", ".join(
            f"{record.get('experiment_id')}={record.get('status')}"
            for record in incomplete
        )
        raise RuntimeError(
            f"Core summary is incomplete ({len(records)} records; {details}). "
            "Use --allow-incomplete only for a progress snapshot."
        )

    block = build_block(summary)
    replace_marked_block(TRAINING_RESULTS_MD, block)
    replace_marked_block(MASTER_PLAN_MD, block)
    finalize_static_text(is_complete=len(records) == CORE_RUN_COUNT and not incomplete)
    print(f"updated={TRAINING_RESULTS_MD}")
    print(f"updated={MASTER_PLAN_MD}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI。"""
    parser = argparse.ArgumentParser(description="同步 SPAD 消融结果到 Markdown")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="允许把未完成状态写成进度快照；正式最终同步不要使用",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    return update_docs(allow_incomplete=args.allow_incomplete)


def main_without_cli() -> None:
    """无参数模式：仅在 8/8 完整时同步最终结果。"""
    update_docs(allow_incomplete=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
