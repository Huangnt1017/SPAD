"""消融汇总必须识别统一复测目录和复用锚点。"""

from __future__ import annotations

import unittest

from scripts.ablation_registry import CORE_EXPERIMENTS, STRUCTURE_CORE_EXPERIMENTS
from scripts.summarize_ablation import UNIFIED_TEST_ROOT, is_unified_metrics_path


class UnifiedMetricsPathTests(unittest.TestCase):
    """统一无增强 JSON 的历史目录必须继续作为正式资产。"""

    @staticmethod
    def _by_id(experiments, experiment_id: str):
        return next(item for item in experiments if item.experiment_id == experiment_id)

    def test_core_unified_test_directory_is_accepted(self) -> None:
        experiment = self._by_id(CORE_EXPERIMENTS, "A2_seed42")
        metrics_path = UNIFIED_TEST_ROOT / experiment.experiment_id / "metrics_example.json"
        self.assertTrue(is_unified_metrics_path(experiment, metrics_path))

    def test_reused_structure_anchor_directory_is_accepted(self) -> None:
        experiment = self._by_id(STRUCTURE_CORE_EXPERIMENTS, "B0_seed42")
        metrics_path = UNIFIED_TEST_ROOT / "A1_seed42" / "metrics_example.json"
        self.assertTrue(is_unified_metrics_path(experiment, metrics_path))

    def test_unregistered_legacy_directory_is_rejected(self) -> None:
        experiment = self._by_id(CORE_EXPERIMENTS, "A0_seed42")
        metrics_path = UNIFIED_TEST_ROOT.parent / "legacy" / "metrics_example.json"
        self.assertFalse(is_unified_metrics_path(experiment, metrics_path))


if __name__ == "__main__":
    unittest.main()
