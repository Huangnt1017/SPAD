"""消融注册表、结构开关和正式模型兼容性测试。"""

from __future__ import annotations

import argparse
import unittest
from collections import Counter

import torch

from scripts.ablation_registry import (
    CORE_EXPERIMENTS,
    LAMBDA_EXPERIMENTS,
    OPERATOR_EXPERIMENTS,
    PROJECT_ROOT,
    ROBUSTNESS_EXPERIMENTS,
    STRUCTURE_APPENDIX_EXPERIMENTS,
    STRUCTURE_CORE_EXPERIMENTS,
    STRUCTURE_EXPERIMENTS,
    select_experiments,
)
from scripts.train import build_model, set_seed


class AblationRegistryTests(unittest.TestCase):
    """核心、结构、附录和敏感性矩阵必须保持唯一编号来源。"""

    @staticmethod
    def _factor(experiment_id: str) -> str:
        return experiment_id.split("_", 1)[0]

    def test_core_matrix_contains_four_factors_and_two_seeds(self) -> None:
        self.assertEqual(len(CORE_EXPERIMENTS), 8)
        self.assertEqual(len({item.experiment_id for item in CORE_EXPERIMENTS}), 8)
        counts = Counter(self._factor(item.experiment_id) for item in CORE_EXPERIMENTS)
        self.assertEqual(counts, {"A0": 2, "A1": 2, "A2": 2, "A3": 2})
        for factor in ("A0", "A1", "A2", "A3"):
            seeds = {
                item.seed
                for item in CORE_EXPERIMENTS
                if item.experiment_id.startswith(factor + "_")
            }
            self.assertEqual(seeds, {42, 43})

    def test_completed_core_assets_are_registered_for_reuse(self) -> None:
        for experiment in CORE_EXPERIMENTS:
            with self.subTest(experiment=experiment.experiment_id):
                self.assertIsNotNone(experiment.reuse_best_checkpoint)
                self.assertIsNotNone(experiment.reuse_last_checkpoint)
                self.assertIsNotNone(experiment.reuse_train_log)

    def test_seed44_is_separate_robustness_family(self) -> None:
        self.assertEqual(
            {item.experiment_id for item in ROBUSTNESS_EXPERIMENTS},
            {"A0_seed44", "A1_seed44", "A2_seed44", "A3_seed44"},
        )
        self.assertEqual({item.seed for item in ROBUSTNESS_EXPERIMENTS}, {44})
        for experiment in ROBUSTNESS_EXPERIMENTS:
            if experiment.experiment_id == "A3_seed44":
                self.assertIsNone(experiment.reuse_last_checkpoint)
            else:
                self.assertIsNotNone(experiment.reuse_last_checkpoint)

    def test_structure_core_and_appendix_seed_policy(self) -> None:
        core_counts = Counter(
            self._factor(item.experiment_id) for item in STRUCTURE_CORE_EXPERIMENTS
        )
        self.assertEqual(
            core_counts,
            {"B0": 2, "B1": 2, "B3": 2, "B4": 2, "B6": 2, "B7": 2},
        )
        for factor in ("B0", "B1", "B3", "B4", "B6", "B7"):
            seeds = {
                item.seed
                for item in STRUCTURE_CORE_EXPERIMENTS
                if item.experiment_id.startswith(factor + "_")
            }
            self.assertEqual(seeds, {42, 43})

        self.assertEqual(
            {item.experiment_id for item in STRUCTURE_APPENDIX_EXPERIMENTS},
            {"B2_no_se_seed42", "B5_include_self_seed42"},
        )
        self.assertEqual({item.seed for item in STRUCTURE_APPENDIX_EXPERIMENTS}, {42})
        self.assertEqual(
            STRUCTURE_EXPERIMENTS,
            STRUCTURE_CORE_EXPERIMENTS + STRUCTURE_APPENDIX_EXPERIMENTS,
        )
        self.assertEqual(select_experiments(["structure"]), STRUCTURE_EXPERIMENTS)

    def test_structure_rows_change_the_intended_registered_factors(self) -> None:
        baselines = {
            item.seed: item
            for item in STRUCTURE_CORE_EXPERIMENTS
            if self._factor(item.experiment_id) == "B0"
        }
        factor_fields = (
            "gcn_aggregation",
            "gcn_exclude_self",
            "gcn_feature_residual",
            "gcn_use_physical_branch",
            "gcn_use_se_gate",
            "gcn_use_coord_residual",
        )
        expected_by_factor = {
            "B1": ["gcn_use_physical_branch"],
            "B2": ["gcn_use_se_gate"],
            "B3": ["gcn_use_coord_residual"],
            "B4": ["gcn_aggregation"],
            "B5": ["gcn_exclude_self"],
            "B6": ["gcn_feature_residual"],
            "B7": ["gcn_use_physical_branch", "gcn_use_coord_residual"],
        }
        rows = STRUCTURE_CORE_EXPERIMENTS + STRUCTURE_APPENDIX_EXPERIMENTS
        for experiment in rows:
            factor = self._factor(experiment.experiment_id)
            if factor == "B0":
                continue
            baseline = baselines[experiment.seed]
            changed = [
                field
                for field in factor_fields
                if getattr(experiment, field) != getattr(baseline, field)
            ]
            self.assertEqual(changed, expected_by_factor[factor])


    def test_operator_comparison_has_two_parameter_matched_seeds(self) -> None:
        self.assertEqual(
            {item.experiment_id for item in OPERATOR_EXPERIMENTS},
            {"B8_edge_cnn_seed42", "B8_edge_cnn_seed43"},
        )
        self.assertEqual({item.seed for item in OPERATOR_EXPERIMENTS}, {42, 43})
        for experiment in OPERATOR_EXPERIMENTS:
            self.assertEqual(experiment.family, "operator")
            self.assertEqual(experiment.gcn_operator, "edge_cnn")
        self.assertEqual(select_experiments(["operator"]), OPERATOR_EXPERIMENTS)

    def test_lambda_sensitivity_has_four_weights_and_two_seeds(self) -> None:
        self.assertEqual(len(LAMBDA_EXPERIMENTS), 8)
        self.assertEqual(
            Counter(item.seg_loss_weight for item in LAMBDA_EXPERIMENTS),
            {0.0: 2, 0.25: 2, 0.5: 2, 1.0: 2},
        )
        for weight in (0.0, 0.25, 0.5, 1.0):
            self.assertEqual(
                {item.seed for item in LAMBDA_EXPERIMENTS if item.seg_loss_weight == weight},
                {42, 43},
            )
        for experiment in LAMBDA_EXPERIMENTS:
            with self.subTest(experiment=experiment.experiment_id):
                if experiment.seg_loss_weight == 0.0:
                    self.assertEqual(experiment.reuse_of, f"A2_seed{experiment.seed}")
                    self.assertIsNotNone(experiment.reuse_last_checkpoint)
                elif experiment.seg_loss_weight == 0.5:
                    self.assertEqual(experiment.reuse_of, f"A3_seed{experiment.seed}")
                    self.assertIsNotNone(experiment.reuse_last_checkpoint)
                else:
                    self.assertIsNone(experiment.reuse_of)
                    self.assertIsNone(experiment.reuse_last_checkpoint)

    def test_unknown_family_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown ablation families"):
            select_experiments(["unknown"])


class FullAblationCompatibilityTests(unittest.TestCase):
    """全开消融模型必须与正式模型同初始化、同参数名、同输出。"""

    @staticmethod
    def _args(box_head: str) -> argparse.Namespace:
        return argparse.Namespace(
            box_head=box_head,
            gcn_k=None,
            gcn_use_checkpoint=False,
            gcn_aggregation="max",
            gcn_operator="sage",
            gcn_exclude_self=True,
            gcn_feature_residual=True,
            gcn_coord_scale_init=0.1,
            gcn_legacy_mode=False,
            gcn_use_physical_branch=True,
            gcn_use_se_gate=True,
            gcn_use_coord_residual=True,
        )

    def _assert_equivalent(self, box_head: str) -> None:
        set_seed(20260717)
        formal = build_model(
            "graph_residual_gcn",
            num_classes=26,
            project_root=PROJECT_ROOT,
            args=self._args(box_head),
        )
        set_seed(20260717)
        ablation = build_model(
            "graph_residual_gcn_ablation",
            num_classes=26,
            project_root=PROJECT_ROOT,
            args=self._args(box_head),
        )
        formal_state = formal.state_dict()
        ablation_state = ablation.state_dict()
        self.assertEqual(list(formal_state), list(ablation_state))
        for key in formal_state:
            torch.testing.assert_close(formal_state[key], ablation_state[key])

        formal.eval()
        ablation.eval()
        points = torch.randn(2, 32, 4)
        with torch.no_grad():
            formal_out = formal(points)
            ablation_out = ablation(points)
        self.assertEqual(formal_out.keys(), ablation_out.keys())
        for key in formal_out:
            torch.testing.assert_close(formal_out[key], ablation_out[key])

    def test_mlp_head_all_on_is_exactly_equivalent(self) -> None:
        self._assert_equivalent("mlp")

    def test_centroid_head_all_on_is_exactly_equivalent(self) -> None:
        self._assert_equivalent("centroid")


if __name__ == "__main__":
    unittest.main()
