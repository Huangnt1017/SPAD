"""GraphSAGE 与参数匹配 EdgeCNN 消融算子测试。"""

from __future__ import annotations

import argparse
import unittest

import torch

from model.graph_res_GCN import (
    BatchedEdgeCNNConv,
    BatchedSAGEConv,
    batched_knn_edge_index,
)
from scripts.ablation_registry import OPERATOR_EXPERIMENTS, PROJECT_ROOT
from scripts.run_ablation_matrix import build_train_command
from scripts.train import build_model


class EdgeCNNOperatorTests(unittest.TestCase):
    """EdgeCNN 必须保持 GraphSAGE 的接口、形状和参数预算。"""

    def test_parameter_count_matches_sage(self) -> None:
        for in_channels, out_channels in ((4, 64), (32, 64), (64, 128), (128, 256)):
            with self.subTest(in_channels=in_channels, out_channels=out_channels):
                sage = BatchedSAGEConv(in_channels, out_channels, aggregation="max")
                edge_cnn = BatchedEdgeCNNConv(
                    in_channels,
                    out_channels,
                    aggregation="max",
                )
                self.assertEqual(
                    sum(parameter.numel() for parameter in sage.parameters()),
                    sum(parameter.numel() for parameter in edge_cnn.parameters()),
                )

    def test_forward_shape_and_backward(self) -> None:
        points = torch.randn(2, 8, 6, requires_grad=True)
        knn_idx = torch.tensor(
            [
                [[1, 2], [0, 2], [0, 1], [2, 4], [3, 5], [3, 4]],
                [[1, 3], [0, 2], [1, 4], [0, 5], [2, 5], [3, 4]],
            ],
            dtype=torch.long,
        )
        edge_index = batched_knn_edge_index(knn_idx, batch_size=2, num_nodes=6)
        layer = BatchedEdgeCNNConv(8, 12, aggregation="max")
        output = layer(points, edge_index)
        self.assertEqual(tuple(output.shape), (2, 12, 6))
        output.square().mean().backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_mean_aggregation_is_supported(self) -> None:
        points = torch.randn(1, 4, 4)
        knn_idx = torch.tensor([[[1, 2], [0, 2], [0, 3], [1, 2]]])
        edge_index = batched_knn_edge_index(knn_idx, batch_size=1, num_nodes=4)
        output = BatchedEdgeCNNConv(4, 5, aggregation="mean")(points, edge_index)
        self.assertEqual(tuple(output.shape), (1, 5, 4))
        self.assertTrue(torch.isfinite(output).all())


class EdgeCNNAblationWiringTests(unittest.TestCase):
    """B8 注册表、训练命令和模型构造必须统一使用 edge_cnn。"""

    @staticmethod
    def _experiment(seed: int):
        return next(
            item
            for item in OPERATOR_EXPERIMENTS
            if item.experiment_id == f"B8_edge_cnn_seed{seed}"
        )

    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            box_head="mlp",
            gcn_k=4,
            gcn_operator="edge_cnn",
            gcn_use_checkpoint=False,
            gcn_aggregation="max",
            gcn_exclude_self=True,
            gcn_feature_residual=True,
            gcn_coord_scale_init=0.1,
            gcn_legacy_mode=False,
            gcn_use_physical_branch=True,
            gcn_use_se_gate=True,
            gcn_use_coord_residual=True,
        )

    def test_b8_changes_only_operator(self) -> None:
        for seed in (42, 43):
            experiment = self._experiment(seed)
            self.assertEqual(experiment.gcn_operator, "edge_cnn")
            command = build_train_command(
                experiment,
                None,
                batch_size=32,
                grad_accum_steps=1,
            )
            operator_index = command.index("--gcn-operator")
            self.assertEqual(command[operator_index + 1], "edge_cnn")

    def test_model_forward_contract(self) -> None:
        model = build_model(
            "graph_residual_gcn_ablation",
            num_classes=26,
            project_root=PROJECT_ROOT,
            args=self._args(),
        )
        self.assertEqual(model.effective_config()["gcn_operator"], "edge_cnn")
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 16, 4))
        self.assertEqual(tuple(output["logits"].shape), (2, 26))
        self.assertEqual(tuple(output["box_pred"].shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
