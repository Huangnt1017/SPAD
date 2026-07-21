"""消融评估模型识别与框坐标一致性测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scripts.test import decode_unit_boxes_3d, evaluate, infer_model_name_from_checkpoint
from utils.loss import box_iou_3d_aligned


class AblationCheckpointInferenceTests(unittest.TestCase):
    """消融 checkpoint 必须优先于正式 GCN 前缀识别。"""

    def test_ablation_checkpoint_is_not_misclassified_as_full_model(self) -> None:
        checkpoint = Path(
            "graph_residual_gcn_ablation_20260717_010700_891111_best.pth"
        )
        self.assertEqual(
            infer_model_name_from_checkpoint(checkpoint),
            "graph_residual_gcn_ablation",
        )


class BoxCoordinateConsistencyTests(unittest.TestCase):
    """预测框和 GT 同时仿射变换后 IoU 必须保持不变。"""

    def test_decode_both_sides_preserves_iou(self) -> None:
        predicted = torch.tensor(
            [[0.10, 0.40, 0.20, 0.50, 0.60, 0.80]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[0.20, 0.50, 0.10, 0.40, 0.65, 0.85]],
            dtype=torch.float32,
        )
        normalized_iou = box_iou_3d_aligned(predicted, target)
        absolute_iou = box_iou_3d_aligned(
            decode_unit_boxes_3d(predicted),
            decode_unit_boxes_3d(target),
        )
        torch.testing.assert_close(normalized_iou, absolute_iou)


class EvaluationCenterMetricTests(unittest.TestCase):
    """统一测试必须输出与训练口径一致的归一化中心误差。"""

    def test_center_and_z_mae_are_accumulated(self) -> None:
        class DummyModel(nn.Module):
            def forward(self, points: torch.Tensor):
                batch_size = points.shape[0]
                logits = torch.zeros(batch_size, 2)
                logits[:, 0] = 5.0
                centers = torch.tensor([[0.5, 0.4, 0.8]], dtype=points.dtype)
                return {
                    "logits": logits,
                    "box_pred": centers.repeat(batch_size, 1),
                }

        points = torch.zeros(2, 8, 4)
        targets = {
            "cls_targets": torch.zeros(2, 1, dtype=torch.long),
            "bbox_targets": torch.tensor(
                [
                    [[0.3, 0.5, 0.3, 0.5, 0.7, 0.9]],
                    [[0.3, 0.5, 0.3, 0.5, 0.7, 0.9]],
                ],
                dtype=torch.float32,
            ),
            "mask": torch.ones(2, 1, dtype=torch.bool),
        }
        result = evaluate(
            DummyModel(),
            [(points, targets)],
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            box_space="normalized",
        )
        self.assertLess(result["box_z_mae"], 1e-6)
        self.assertAlmostEqual(result["box_center_mae"], 0.1 / 3.0, places=6)
        self.assertEqual(result["box_eval_samples"], 2)


if __name__ == "__main__":
    unittest.main()
