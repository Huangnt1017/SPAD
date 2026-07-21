"""消融实验唯一编号、冻结配置与可复用资产注册表。

2026-07-17 起采用精简方案：
- A0--A3 主矩阵固定 seed 42/43；seed44 资产转入 robustness，不进入主统计；
- 结构核心为 B0/B1/B3/B4/B6/B7 × seed42/43；
- 算子对照 B8 使用参数匹配 EdgeCNN 与 GraphSAGE 比较，单列 operator family；
- B2（无 SE）和 B5（包含自身）仅做 seed42 附录；
- 目标性权重为 lambda_obj∈{0, 0.25, 0.5, 1.0} × seed42/43，
  其中 0/0.5 复用 A2/A3。

训练、统一测试、汇总和 Markdown 必须读取本注册表，避免配置漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(r"D:\PYproject\SPAD")
DATA_ROOT = Path(r"D:\PYproject\SPADdata\2025-04-30-dpc")
SPLIT_CACHE = DATA_ROOT / ".split_cache.json"
SPLIT_CACHE_SHA256 = (
    "AB94E67744AC3C73FC45A2D3E3E389773661E3EEBA85A6F8EF2C3025220A9F22"
)
SPLIT_SEED = 42
EVAL_SEED = 42
FIXED_PYTHON = Path(r"D:\Anaconda3\envs\torchnew\python.exe")
CORE_SEEDS = (42, 43)
STRUCTURE_CORE_SEEDS = (42, 43)
LAMBDA_SEEDS = (42, 43)


@dataclass(frozen=True)
class AblationExperiment:
    """一次可训练、可测试、可汇总的正式消融实验。"""

    experiment_id: str
    family: str
    seed: int
    model: str
    box_head: str
    seg_loss_weight: float
    description: str
    gcn_aggregation: str = "max"
    gcn_operator: str = "sage"
    gcn_exclude_self: bool = True
    gcn_feature_residual: bool = True
    gcn_use_physical_branch: bool = True
    gcn_use_se_gate: bool = True
    gcn_use_coord_residual: bool = True
    estimated_hours: float = 2.0
    reuse_best_checkpoint: Optional[Path] = None
    reuse_last_checkpoint: Optional[Path] = None
    reuse_train_log: Optional[Path] = None
    reuse_test_metrics: Optional[Path] = None
    reuse_of: Optional[str] = None

    @property
    def output_group(self) -> str:
        """返回日志、checkpoint 和测试输出的稳定分组目录。"""
        return {
            "core": "head",
            "robustness": "robustness",
            "structure_core": "structure/core",
            "structure_appendix": "structure/appendix",
            "operator": "operator",
            "lambda": "lambda",
        }[self.family]

    @property
    def is_gcn(self) -> bool:
        """是否使用 GraphResidual-GCN 系列模型。"""
        return self.model.startswith("graph_residual_gcn")


def _legacy_cls_path(file_name: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / "CLS" / file_name


def _legacy_log_path(file_name: str) -> Path:
    return PROJECT_ROOT / "logs" / "CLS" / file_name


def _legacy_metrics_path(directory: str, file_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "CLS" / directory / file_name


def _abl_head_checkpoint(experiment_id: str, file_name: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / "ABL" / "head" / experiment_id / file_name


def _abl_head_log(experiment_id: str, file_name: str) -> Path:
    return PROJECT_ROOT / "logs" / "ABL" / "head" / experiment_id / file_name


# ---------------------------------------------------------------------------
# A0--A3 主矩阵：仅 seed42/43 进入论文主统计。
# ---------------------------------------------------------------------------
CORE_EXPERIMENTS: List[AblationExperiment] = [
    AblationExperiment(
        "A0_seed42",
        "core",
        42,
        "dgcnn",
        "mlp",
        0.0,
        "DGCNN + 标准 MLP 定位头；骨干参照。",
        estimated_hours=2.85,
        reuse_best_checkpoint=_legacy_cls_path("dgcnn_20260710_165958_587028_best.pth"),
        reuse_last_checkpoint=_legacy_cls_path("dgcnn_20260710_165958_587028_last.pth"),
        reuse_train_log=_legacy_log_path("train_dgcnn_20260710_165958_587028.log"),
        reuse_test_metrics=_legacy_metrics_path(
            "dgcnn_20260710_165958_587028_best_noaug",
            "metrics_dgcnn_20260710_222635_907101.json",
        ),
    ),
    AblationExperiment(
        "A1_seed42",
        "core",
        42,
        "graph_residual_gcn_ablation",
        "mlp",
        0.0,
        "完整 GraphResidual-GCN 骨干 + 标准 MLP 定位头。",
        reuse_best_checkpoint=_legacy_cls_path(
            "graph_residual_gcn_20260711_173454_489156_best.pth"
        ),
        reuse_last_checkpoint=_legacy_cls_path(
            "graph_residual_gcn_20260711_173454_489156_last.pth"
        ),
        reuse_train_log=_legacy_log_path(
            "train_graph_residual_gcn_20260711_173454_489156.log"
        ),
        reuse_test_metrics=_legacy_metrics_path(
            "graph_residual_gcn_20260711_173454_489156_best_noaug",
            "metrics_graph_residual_gcn_20260711_202649_885237.json",
        ),
    ),
    AblationExperiment(
        "A2_seed42",
        "core",
        42,
        "graph_residual_gcn_ablation",
        "centroid",
        0.0,
        "完整 GCN + 质心头，不加入目标性 BCE。",
        reuse_best_checkpoint=_abl_head_checkpoint(
            "A2_seed42", "graph_residual_gcn_ablation_20260717_010700_891111_best.pth"
        ),
        reuse_last_checkpoint=_abl_head_checkpoint(
            "A2_seed42", "graph_residual_gcn_ablation_20260717_010700_891111_last.pth"
        ),
        reuse_train_log=_abl_head_log(
            "A2_seed42", "train_graph_residual_gcn_ablation_20260717_010700_891111.log"
        ),
    ),
    AblationExperiment(
        "A3_seed42",
        "core",
        42,
        "graph_residual_gcn_ablation",
        "centroid",
        0.5,
        "完整方法：质心头 + 0.5×目标性 BCE。",
        reuse_best_checkpoint=_legacy_cls_path(
            "graph_residual_gcn_20260710_200627_305941_best.pth"
        ),
        reuse_last_checkpoint=_legacy_cls_path(
            "graph_residual_gcn_20260710_200627_305941_last.pth"
        ),
        reuse_train_log=_legacy_log_path(
            "train_graph_residual_gcn_20260710_200627_305941.log"
        ),
        reuse_test_metrics=_legacy_metrics_path(
            "graph_residual_gcn_20260710_200627_305941_best_noaug",
            "metrics_graph_residual_gcn_20260710_222623_643385.json",
        ),
    ),
]

CORE_EXPERIMENTS.extend(
    [
        AblationExperiment(
            "A0_seed43", "core", 43, "dgcnn", "mlp", 0.0,
            "DGCNN + 标准 MLP 定位头；骨干参照。", estimated_hours=2.85,
            reuse_best_checkpoint=_abl_head_checkpoint(
                "A0_seed43", "dgcnn_20260717_030131_027566_best.pth"
            ),
            reuse_last_checkpoint=_abl_head_checkpoint(
                "A0_seed43", "dgcnn_20260717_030131_027566_last.pth"
            ),
            reuse_train_log=_abl_head_log(
                "A0_seed43", "train_dgcnn_20260717_030131_027566.log"
            ),
        ),
        AblationExperiment(
            "A1_seed43", "core", 43,
            "graph_residual_gcn_ablation", "mlp", 0.0,
            "完整 GraphResidual-GCN 骨干 + 标准 MLP 定位头。",
            reuse_best_checkpoint=_abl_head_checkpoint(
                "A1_seed43", "graph_residual_gcn_ablation_20260717_054231_137403_best.pth"
            ),
            reuse_last_checkpoint=_abl_head_checkpoint(
                "A1_seed43", "graph_residual_gcn_ablation_20260717_054231_137403_last.pth"
            ),
            reuse_train_log=_abl_head_log(
                "A1_seed43", "train_graph_residual_gcn_ablation_20260717_054231_137403.log"
            ),
        ),
        AblationExperiment(
            "A2_seed43", "core", 43,
            "graph_residual_gcn_ablation", "centroid", 0.0,
            "完整 GCN + 质心头，不加入目标性 BCE。",
            reuse_best_checkpoint=_abl_head_checkpoint(
                "A2_seed43", "graph_residual_gcn_ablation_20260717_073431_247421_best.pth"
            ),
            reuse_last_checkpoint=_abl_head_checkpoint(
                "A2_seed43", "graph_residual_gcn_ablation_20260717_073431_247421_last.pth"
            ),
            reuse_train_log=_abl_head_log(
                "A2_seed43", "train_graph_residual_gcn_ablation_20260717_073431_247421.log"
            ),
        ),
        AblationExperiment(
            "A3_seed43", "core", 43,
            "graph_residual_gcn_ablation", "centroid", 0.5,
            "完整方法：质心头 + 0.5×目标性 BCE。",
            reuse_best_checkpoint=_abl_head_checkpoint(
                "A3_seed43", "graph_residual_gcn_ablation_20260717_092801_375376_best.pth"
            ),
            reuse_last_checkpoint=_abl_head_checkpoint(
                "A3_seed43", "graph_residual_gcn_ablation_20260717_092801_375376_last.pth"
            ),
            reuse_train_log=_abl_head_log(
                "A3_seed43", "train_graph_residual_gcn_ablation_20260717_092801_375376.log"
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# seed44 已有资产：只做额外稳健性观察，不进入主 2-seed mean/std。
# ---------------------------------------------------------------------------
ROBUSTNESS_EXPERIMENTS: List[AblationExperiment] = [
    AblationExperiment(
        "A0_seed44", "robustness", 44, "dgcnn", "mlp", 0.0,
        "A0 的额外 seed44 稳健性资产，不进入主统计。", estimated_hours=2.85,
        reuse_best_checkpoint=_abl_head_checkpoint(
            "A0_seed44", "dgcnn_20260717_112231_524334_best.pth"
        ),
        reuse_last_checkpoint=_abl_head_checkpoint(
            "A0_seed44", "dgcnn_20260717_112231_524334_last.pth"
        ),
        reuse_train_log=_abl_head_log(
            "A0_seed44", "train_dgcnn_20260717_112231_524334.log"
        ),
    ),
    AblationExperiment(
        "A1_seed44", "robustness", 44, "graph_residual_gcn_ablation", "mlp", 0.0,
        "A1 的额外 seed44 稳健性资产，不进入主统计。",
        reuse_best_checkpoint=_abl_head_checkpoint(
            "A1_seed44", "graph_residual_gcn_ablation_20260717_141431_698479_best.pth"
        ),
        reuse_last_checkpoint=_abl_head_checkpoint(
            "A1_seed44", "graph_residual_gcn_ablation_20260717_141431_698479_last.pth"
        ),
        reuse_train_log=_abl_head_log(
            "A1_seed44", "train_graph_residual_gcn_ablation_20260717_141431_698479.log"
        ),
    ),
    AblationExperiment(
        "A2_seed44", "robustness", 44, "graph_residual_gcn_ablation", "centroid", 0.0,
        "A2 的额外 seed44 稳健性资产，不进入主统计。",
        reuse_best_checkpoint=_abl_head_checkpoint(
            "A2_seed44", "graph_residual_gcn_ablation_20260717_160601_821719_best.pth"
        ),
        reuse_last_checkpoint=_abl_head_checkpoint(
            "A2_seed44", "graph_residual_gcn_ablation_20260717_160601_821719_last.pth"
        ),
        reuse_train_log=_abl_head_log(
            "A2_seed44", "train_graph_residual_gcn_ablation_20260717_160601_821719.log"
        ),
    ),
    AblationExperiment(
        "A3_seed44", "robustness", 44, "graph_residual_gcn_ablation", "centroid", 0.5,
        "可选额外 seed44；主 2-seed 方案不要求训练。",
    ),
]


def _core_by_id(experiment_id: str) -> AblationExperiment:
    return next(item for item in CORE_EXPERIMENTS if item.experiment_id == experiment_id)


def _structure_base(seed: int) -> AblationExperiment:
    anchor = _core_by_id(f"A1_seed{seed}")
    return AblationExperiment(
        experiment_id=f"B0_seed{seed}",
        family="structure_core",
        seed=seed,
        model="graph_residual_gcn_ablation",
        box_head="mlp",
        seg_loss_weight=0.0,
        description=f"结构锚点：完整 GCN，复用 A1_seed{seed}。",
        reuse_best_checkpoint=anchor.reuse_best_checkpoint,
        reuse_last_checkpoint=anchor.reuse_last_checkpoint,
        reuse_train_log=anchor.reuse_train_log,
        reuse_test_metrics=anchor.reuse_test_metrics,
        reuse_of=f"A1_seed{seed}",
    )


# ---------------------------------------------------------------------------
# 核心结构消融：B0/B1/B3/B4/B6/B7 × seed42/43。
# ---------------------------------------------------------------------------
STRUCTURE_CORE_EXPERIMENTS: List[AblationExperiment] = []
for structure_seed in STRUCTURE_CORE_SEEDS:
    base = _structure_base(structure_seed)
    STRUCTURE_CORE_EXPERIMENTS.extend(
        [
            base,
            replace(
                base,
                experiment_id=f"B1_no_physical_seed{structure_seed}",
                description="关闭坐标图 GraphSAGE 分支，仅保留动态特征图。",
                gcn_use_physical_branch=False,
                reuse_best_checkpoint=None,
                reuse_last_checkpoint=None,
                reuse_train_log=None,
                reuse_test_metrics=None,
                reuse_of=None,
            ),
            replace(
                base,
                experiment_id=f"B3_no_coord_residual_seed{structure_seed}",
                description="硬关闭坐标门控、坐标编码器及坐标残差。",
                gcn_use_coord_residual=False,
                reuse_best_checkpoint=None,
                reuse_last_checkpoint=None,
                reuse_train_log=None,
                reuse_test_metrics=None,
                reuse_of=None,
            ),
            replace(
                base,
                experiment_id=f"B4_mean_aggregation_seed{structure_seed}",
                description="GraphSAGE 聚合由 max 改为 mean。",
                gcn_aggregation="mean",
                reuse_best_checkpoint=None,
                reuse_last_checkpoint=None,
                reuse_train_log=None,
                reuse_test_metrics=None,
                reuse_of=None,
            ),
            replace(
                base,
                experiment_id=f"B6_no_feature_residual_seed{structure_seed}",
                description="关闭特征残差捷径。",
                gcn_feature_residual=False,
                reuse_best_checkpoint=None,
                reuse_last_checkpoint=None,
                reuse_train_log=None,
                reuse_test_metrics=None,
                reuse_of=None,
            ),
            replace(
                base,
                experiment_id=f"B7_no_coordinate_pathways_seed{structure_seed}",
                description="同时关闭坐标图分支和坐标残差，检查两条显式坐标增强路径。",
                gcn_use_physical_branch=False,
                gcn_use_coord_residual=False,
                reuse_best_checkpoint=None,
                reuse_last_checkpoint=None,
                reuse_train_log=None,
                reuse_test_metrics=None,
                reuse_of=None,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 单 seed 附录：B2/B5 仅 seed42。
# ---------------------------------------------------------------------------
_APPENDIX_BASE = replace(_structure_base(42), family="structure_appendix")
STRUCTURE_APPENDIX_EXPERIMENTS: List[AblationExperiment] = [
    replace(
        _APPENDIX_BASE,
        experiment_id="B2_no_se_seed42",
        description="附录：关闭 SE 通道门控，仅 seed42。",
        gcn_use_se_gate=False,
        reuse_best_checkpoint=None,
        reuse_last_checkpoint=None,
        reuse_train_log=None,
        reuse_test_metrics=None,
        reuse_of=None,
    ),
    replace(
        _APPENDIX_BASE,
        experiment_id="B5_include_self_seed42",
        description="附录：kNN 邻域包含自身点，仅 seed42。",
        gcn_exclude_self=False,
        reuse_best_checkpoint=None,
        reuse_last_checkpoint=None,
        reuse_train_log=None,
        reuse_test_metrics=None,
        reuse_of=None,
    ),
]

# 向后兼容旧命令中的 ``--families structure`` 和旧测试/工具导入。
STRUCTURE_EXPERIMENTS: List[AblationExperiment] = (
    STRUCTURE_CORE_EXPERIMENTS + STRUCTURE_APPENDIX_EXPERIMENTS
)


# ---------------------------------------------------------------------------
# 算子对照：保持全部结构不变，仅将 GraphSAGE 替换为参数匹配 EdgeCNN。
# ---------------------------------------------------------------------------
OPERATOR_EXPERIMENTS: List[AblationExperiment] = []
for operator_seed in STRUCTURE_CORE_SEEDS:
    OPERATOR_EXPERIMENTS.append(
        replace(
            _structure_base(operator_seed),
            experiment_id=f"B8_edge_cnn_seed{operator_seed}",
            family="operator",
            description=(
                "保持相同 KNN、双分支、融合与残差结构，将 GraphSAGE 替换为"
                "参数量匹配的 EdgeCNN 边卷积。"
            ),
            gcn_operator="edge_cnn",
            reuse_best_checkpoint=None,
            reuse_last_checkpoint=None,
            reuse_train_log=None,
            reuse_test_metrics=None,
            reuse_of=None,
        )
    )


# ---------------------------------------------------------------------------
# lambda_obj 敏感性：{0, 0.25, 0.5, 1.0} × seed42/43。
# 0 与 0.5 分别复用 A2/A3；只新增 0.25 和 1.0 的四次训练。
# ---------------------------------------------------------------------------
LAMBDA_EXPERIMENTS: List[AblationExperiment] = []
for lambda_seed in LAMBDA_SEEDS:
    a2_anchor = _core_by_id(f"A2_seed{lambda_seed}")
    a3_anchor = _core_by_id(f"A3_seed{lambda_seed}")
    LAMBDA_EXPERIMENTS.extend(
        [
            AblationExperiment(
                f"C_lambda_0_seed{lambda_seed}", "lambda", lambda_seed,
                "graph_residual_gcn_ablation", "centroid", 0.0,
                f"lambda_obj=0，复用 A2_seed{lambda_seed}。",
                reuse_best_checkpoint=a2_anchor.reuse_best_checkpoint,
                reuse_last_checkpoint=a2_anchor.reuse_last_checkpoint,
                reuse_train_log=a2_anchor.reuse_train_log,
                reuse_test_metrics=a2_anchor.reuse_test_metrics,
                reuse_of=f"A2_seed{lambda_seed}",
            ),
            AblationExperiment(
                f"C_lambda_0p25_seed{lambda_seed}", "lambda", lambda_seed,
                "graph_residual_gcn_ablation", "centroid", 0.25,
                "目标性 BCE 权重敏感性：lambda_obj=0.25。",
            ),
            AblationExperiment(
                f"C_lambda_0p5_seed{lambda_seed}", "lambda", lambda_seed,
                "graph_residual_gcn_ablation", "centroid", 0.5,
                f"lambda_obj=0.5，复用 A3_seed{lambda_seed}。",
                reuse_best_checkpoint=a3_anchor.reuse_best_checkpoint,
                reuse_last_checkpoint=a3_anchor.reuse_last_checkpoint,
                reuse_train_log=a3_anchor.reuse_train_log,
                reuse_test_metrics=a3_anchor.reuse_test_metrics,
                reuse_of=f"A3_seed{lambda_seed}",
            ),
            AblationExperiment(
                f"C_lambda_1p0_seed{lambda_seed}", "lambda", lambda_seed,
                "graph_residual_gcn_ablation", "centroid", 1.0,
                "目标性 BCE 权重敏感性：lambda_obj=1.0。",
            ),
        ]
    )


ALL_EXPERIMENTS: List[AblationExperiment] = (
    CORE_EXPERIMENTS
    + ROBUSTNESS_EXPERIMENTS
    + STRUCTURE_CORE_EXPERIMENTS
    + STRUCTURE_APPENDIX_EXPERIMENTS
    + OPERATOR_EXPERIMENTS
    + LAMBDA_EXPERIMENTS
)
EXPERIMENT_BY_ID: Dict[str, AblationExperiment] = {
    experiment.experiment_id: experiment for experiment in ALL_EXPERIMENTS
}
if len(EXPERIMENT_BY_ID) != len(ALL_EXPERIMENTS):
    raise RuntimeError("Duplicate ablation experiment IDs detected")


def select_experiments(
    families: Sequence[str],
    experiment_ids: Optional[Iterable[str]] = None,
) -> List[AblationExperiment]:
    """按注册顺序筛选实验，并拒绝未知编号。"""
    valid_families = {
        "core",
        "robustness",
        "structure",
        "structure_core",
        "structure_appendix",
        "operator",
        "lambda",
    }
    unknown_families = sorted(set(families) - valid_families)
    if unknown_families:
        raise ValueError(f"Unknown ablation families: {', '.join(unknown_families)}")

    family_set = set(families)
    if "structure" in family_set:
        family_set.remove("structure")
        family_set.update({"structure_core", "structure_appendix"})
    selected = [item for item in ALL_EXPERIMENTS if item.family in family_set]
    if experiment_ids is None:
        return selected

    requested = list(experiment_ids)
    unknown = [item for item in requested if item not in EXPERIMENT_BY_ID]
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {', '.join(unknown)}")
    requested_set = set(requested)
    return [item for item in selected if item.experiment_id in requested_set]


def expected_checkpoint_dir(experiment: AblationExperiment) -> Path:
    """返回新训练使用的 checkpoint 目录。"""
    return (
        PROJECT_ROOT
        / "checkpoints"
        / "ABL"
        / experiment.output_group
        / experiment.experiment_id
    )


def expected_log_dir(experiment: AblationExperiment) -> Path:
    """返回新训练/测试使用的日志目录。"""
    return PROJECT_ROOT / "logs" / "ABL" / experiment.output_group / experiment.experiment_id


def expected_output_dir(experiment: AblationExperiment) -> Path:
    """返回统一无增强测试输出目录。"""
    return (
        PROJECT_ROOT
        / "outputs"
        / "ABL"
        / experiment.output_group
        / experiment.experiment_id
        / f"test_noaug_eval_seed{EVAL_SEED}"
    )
