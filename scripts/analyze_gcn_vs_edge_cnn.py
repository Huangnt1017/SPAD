"""分析参数匹配 EdgeCNN 与 GraphSAGE 的逐 seed 差异。

CLI 示例：
    D:\Anaconda3\envs\torchnew\python.exe scripts\analyze_gcn_vs_edge_cnn.py
    D:\Anaconda3\envs\torchnew\python.exe scripts\analyze_gcn_vs_edge_cnn.py --require-complete

无参数运行会生成当前进度快照；B8 尚未训练时明确标记 pending，不启动训练。
输出默认写入 ``outputs/ABL/summary/gcn_vs_edge_cnn_analysis.{json,md}``。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

from model.graph_res_GCN_ablation import GraphResidualGCNAblationNet
from scripts.ablation_registry import (
    OPERATOR_EXPERIMENTS,
    PROJECT_ROOT,
    STRUCTURE_CORE_EXPERIMENTS,
)
from scripts.summarize_ablation import RunRecord, build_run_record


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "ABL" / "summary" / "gcn_vs_edge_cnn_analysis.json"
METRICS = (
    "top1",
    "f1_macro",
    "box_z_mae",
    "box_center_mae",
    "mean_iou_matched_cls",
    "AP50",
    "ap_50_95",
)


def experiment_by_id(experiment_id: str):
    """从结构锚点或算子对照注册表读取唯一实验。"""
    experiments = STRUCTURE_CORE_EXPERIMENTS + OPERATOR_EXPERIMENTS
    return next(item for item in experiments if item.experiment_id == experiment_id)


def parameter_count(operator: str) -> int:
    """返回 MLP 定位头下完整结构的参数量。"""
    model = GraphResidualGCNAblationNet(
        seg_centroid_box=False,
        operator=operator,
        use_checkpoint=False,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def metric_value(record: RunRecord, metric: str) -> Optional[float]:
    """读取 RunRecord 指标。"""
    value = getattr(record, metric)
    return None if value is None else float(value)


def paired_delta(edge_record: RunRecord, sage_record: RunRecord) -> Dict[str, Optional[float]]:
    """计算同 seed EdgeCNN-GraphSAGE；MAE 为负表示 EdgeCNN 误差更低。"""
    result: Dict[str, Optional[float]] = {}
    for metric in METRICS:
        edge_value = metric_value(edge_record, metric)
        sage_value = metric_value(sage_record, metric)
        result[metric] = (
            None if edge_value is None or sage_value is None else edge_value - sage_value
        )
    return result


def format_value(value: Optional[float], *, percent: bool = False) -> str:
    """格式化可空结果。"""
    if value is None:
        return "--"
    return f"{value * 100:.2f}" if percent else f"{value:.6f}"


def write_markdown(path: Path, payload: Mapping[str, object]) -> None:
    """写入便于直接检查的 Markdown 分析表。"""
    architecture = payload["architecture"]
    runs = payload["runs"]
    deltas = payload["paired_deltas"]
    lines = [
        "# GraphSAGE 与参数匹配 EdgeCNN 对照",
        "",
        f"> GraphSAGE 参数量：{architecture['sage_parameters']:,}",
        f"> EdgeCNN 参数量：{architecture['edge_cnn_parameters']:,}",
        f"> 参数量完全一致：{architecture['parameter_matched']}",
        "",
        "两组保持相同 KNN、双分支、SE、融合、feature residual、coordinate residual、head 和训练协议；只替换局部消息传递算子。",
        "",
        "| seed | B0 GraphSAGE | B8 EdgeCNN | 状态 |",
        "|---:|---|---|---|",
    ]
    for seed in (42, 43):
        sage = runs[str(seed)]["sage"]
        edge = runs[str(seed)]["edge_cnn"]
        state = "complete" if sage["status"] == edge["status"] == "complete" else "pending"
        lines.append(
            f"| {seed} | {sage['status']} | {edge['status']} | {state} |"
        )

    lines.extend(
        [
            "",
            "## 同 seed 差值：EdgeCNN - GraphSAGE",
            "",
            "| seed | Top-1 (pp) | F1 (pp) | z-MAE | center-MAE | mIoU | AP50 (pp) | AP50:95 (pp) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in (42, 43):
        row = deltas.get(str(seed), {})
        lines.append(
            "| {seed} | {top1} | {f1} | {z} | {center} | {iou} | {ap50} | {ap} |".format(
                seed=seed,
                top1=format_value(row.get("top1"), percent=True),
                f1=format_value(row.get("f1_macro"), percent=True),
                z=format_value(row.get("box_z_mae")),
                center=format_value(row.get("box_center_mae")),
                iou=format_value(row.get("mean_iou_matched_cls")),
                ap50=format_value(row.get("AP50"), percent=True),
                ap=format_value(row.get("ap_50_95"), percent=True),
            )
        )
    lines.extend(
        [
            "",
            "解释：Top-1/F1/mIoU/AP 为正表示 EdgeCNN 更高；z-MAE/center-MAE 为负表示 EdgeCNN 误差更低。只有两个 seed 都完成后才报告 mean/std，不作显著性声明。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(output_path: Path, *, require_complete: bool) -> int:
    """生成参数与测试指标对照，不启动训练或评估。"""
    sage_parameters = parameter_count("sage")
    edge_parameters = parameter_count("edge_cnn")
    runs: Dict[str, Dict[str, object]] = {}
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    incomplete = []

    for seed in (42, 43):
        sage_record = build_run_record(experiment_by_id(f"B0_seed{seed}"))
        edge_record = build_run_record(experiment_by_id(f"B8_edge_cnn_seed{seed}"))
        runs[str(seed)] = {
            "sage": asdict(sage_record),
            "edge_cnn": asdict(edge_record),
        }
        if sage_record.status == edge_record.status == "complete":
            deltas[str(seed)] = paired_delta(edge_record, sage_record)
        else:
            deltas[str(seed)] = {metric: None for metric in METRICS}
            incomplete.append(seed)

    payload = {
        "architecture": {
            "sage_parameters": sage_parameters,
            "edge_cnn_parameters": edge_parameters,
            "parameter_matched": sage_parameters == edge_parameters,
            "controlled_factors": [
                "same KNN graphs",
                "same block channels",
                "same physical and feature branches",
                "same SE and fusion",
                "same feature and coordinate residuals",
                "same heads and training protocol",
            ],
            "changed_factor": "local operator: GraphSAGE vs EdgeCNN edge MLP",
        },
        "runs": runs,
        "paired_deltas": deltas,
        "complete_seeds": [seed for seed in (42, 43) if seed not in incomplete],
        "pending_seeds": incomplete,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path = output_path.with_suffix(".md")
    write_markdown(markdown_path, payload)
    print(f"json={output_path}")
    print(f"markdown={markdown_path}")
    print(f"parameter_matched={sage_parameters == edge_parameters}")
    print(f"pending_seeds={incomplete}")
    return 2 if require_complete and incomplete else 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI。"""
    parser = argparse.ArgumentParser(description="分析 GraphSAGE 与参数匹配 EdgeCNN 差异")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="分析 JSON 输出路径；同名 .md 会同时生成",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="B8 两 seed 未完成时返回退出码 2",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    return run_analysis(args.output, require_complete=args.require_complete)


def main_without_cli() -> None:
    """无参数模式：生成允许 B8 缺失的当前快照。"""
    run_analysis(DEFAULT_OUTPUT, require_complete=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
