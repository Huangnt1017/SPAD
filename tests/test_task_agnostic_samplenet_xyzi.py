"""强度感知任务无关 SampleNet-XYZI 数据、损失、恢复和导出测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from downsampling import SampleNetXYZI, assert_unique_indices
from downsampling.common import gather_points
from downsampling.task_agnostic_xyzi import (
    FormalXYZICandidateDataset,
    scan_formal_point_files,
    split_formal_files,
    task_agnostic_samplenet_loss,
)
from scripts.train_task_agnostic_samplenet_xyzi import (
    TaskAgnosticConfig,
    export_frozen_model,
    restore_training_checkpoint,
    save_training_checkpoint,
)


def make_raw_points(count: int, offset: float = 0.0) -> np.ndarray:
    """生成每行互异、强度非负的测试 XYZI。"""

    row = np.arange(count, dtype=np.float32)
    return np.stack(
        (
            row + offset,
            row * 2.0 + offset,
            row * 3.0 + offset,
            row % 7.0 + 1.0,
        ),
        axis=1,
    )


class TaskAgnosticSampleNetXYZITests(unittest.TestCase):
    """验证数据契约、可微损失、checkpoint 和硬点导出。"""

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "A").mkdir()
        (self.root / "B").mkdir()
        self.file_a = (
            self.root
            / "A"
            / "2025-04-30_18-47-35_Delay-0_Width-200-1-3.txt"
        )
        self.file_b = (
            self.root
            / "B"
            / "2025-04-30_18-48-37_Delay-0_Width-200-2-4.txt"
        )
        np.savetxt(self.file_a, make_raw_points(32), delimiter=",", fmt="%.6g")
        np.savetxt(
            self.file_b,
            make_raw_points(36, offset=100.0),
            delimiter=",",
            fmt="%.6g",
        )
        np.savetxt(
            self.root / "A" / "A.txt",
            make_raw_points(32),
            delimiter=",",
            fmt="%.6g",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def make_config(root: Path) -> TaskAgnosticConfig:
        return TaskAgnosticConfig(
            mode="sanity",
            data_root=root,
            output_root=root / "outputs",
            candidate_points=16,
            num_samples=4,
            projection_neighbors=2,
            batch_size=2,
            epochs=2,
            feature_dim=16,
            hidden_dim=32,
            distance_chunk_size=2,
            intensity_chunk_size=5,
            device="cpu",
            max_files=2,
            seed=19,
        )

    def make_model(self) -> SampleNetXYZI:
        return SampleNetXYZI(
            num_samples=4,
            projection_neighbors=2,
            feature_dim=16,
            hidden_dim=32,
            distance_chunk_size=2,
        )

    def test_strict_scan_and_dataset_have_no_labels(self) -> None:
        files = scan_formal_point_files(self.root)
        self.assertEqual(files, [self.file_a.resolve(), self.file_b.resolve()])
        train_files, val_files = split_formal_files(files, val_ratio=0.5, seed=3)
        self.assertEqual(len(train_files), 1)
        self.assertEqual(len(val_files), 1)

        dataset = FormalXYZICandidateDataset(
            data_root=self.root,
            files=files,
            candidate_points=16,
            seed=11,
        )
        sample = dataset[0]
        self.assertEqual(
            set(sample),
            {"points", "source_indices", "relative_path"},
        )
        self.assertEqual(sample["points"].shape, (16, 4))
        self.assertEqual(sample["source_indices"].shape, (16,))
        self.assertEqual(
            torch.unique(sample["source_indices"]).numel(),
            16,
        )
        self.assertEqual(sample["relative_path"], self.file_a.relative_to(self.root).as_posix())

    def test_loss_uses_differentiable_generated_and_projected_points(self) -> None:
        dataset = FormalXYZICandidateDataset(
            data_root=self.root,
            files=[self.file_a],
            candidate_points=16,
            seed=5,
        )
        points = dataset[0]["points"].unsqueeze(0)
        model = self.make_model().train()
        output = model(points)
        self.assertTrue(output.generated_points.requires_grad)
        self.assertTrue(output.projected_points.requires_grad)

        losses = task_agnostic_samplenet_loss(
            input_points=points,
            output=output,
            intensity_chunk_size=5,
        )
        self.assertEqual(
            set(losses),
            {
                "total",
                "geometry",
                "intensity_coverage",
                "projection_temperature",
            },
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)
        self.assertIsNotNone(model.decoder[-1].weight.grad)
        self.assertIsNotNone(model._temperature_unconstrained.grad)
        self.assertGreater(float(model.encoder[0].weight.grad.norm()), 0.0)
        self.assertGreater(float(model.decoder[-1].weight.grad.norm()), 0.0)
        self.assertGreater(float(model._temperature_unconstrained.grad.abs()), 0.0)

        assert_unique_indices(output.indices)
        self.assertTrue(
            torch.equal(output.points, gather_points(points, output.indices))
        )

    def test_checkpoint_restore_and_txt_export(self) -> None:
        config = self.make_config(self.root)
        model = self.make_model().train()
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
        dataset = FormalXYZICandidateDataset(
            data_root=self.root,
            files=[self.file_a],
            candidate_points=config.candidate_points,
            seed=config.seed,
        )
        points = dataset[0]["points"].unsqueeze(0)
        output = model(points)
        losses = task_agnostic_samplenet_loss(
            input_points=points,
            output=output,
            intensity_chunk_size=config.intensity_chunk_size,
        )
        losses["total"].backward()
        optimizer.step()
        scheduler.step()

        checkpoint_path = self.root / "run" / "checkpoints" / "last.pth"
        save_training_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            best_val_loss=float(losses["total"].detach()),
            config=config,
        )

        restored_model = self.make_model()
        restored_optimizer = AdamW(
            restored_model.parameters(), lr=config.learning_rate
        )
        restored_scheduler = CosineAnnealingLR(
            restored_optimizer, T_max=config.epochs
        )
        next_epoch, best_val_loss = restore_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            config=config,
            device=torch.device("cpu"),
        )
        self.assertEqual(next_epoch, 2)
        self.assertTrue(np.isfinite(best_val_loss))

        export_dir = self.root / "export"
        summary = export_frozen_model(
            model=restored_model,
            data_root=self.root,
            files=[self.file_a, self.file_b],
            export_dir=export_dir,
            candidate_points=config.candidate_points,
            batch_size=2,
            seed=config.seed,
            device=torch.device("cpu"),
        )
        self.assertEqual(summary["exported_files"], 2)
        exported_files = sorted(export_dir.rglob("*"))
        exported_regular_files = [path for path in exported_files if path.is_file()]
        self.assertEqual(
            [path.relative_to(export_dir) for path in exported_regular_files],
            [self.file_a.relative_to(self.root), self.file_b.relative_to(self.root)],
        )
        self.assertTrue(all(path.suffix == ".txt" for path in exported_regular_files))

        for source_path in (self.file_a, self.file_b):
            exported_path = export_dir / source_path.relative_to(self.root)
            source_rows = {
                tuple(row.tolist()) for row in np.loadtxt(source_path, delimiter=",")
            }
            exported = np.loadtxt(exported_path, delimiter=",")
            self.assertEqual(exported.shape, (config.num_samples, 4))
            self.assertTrue(
                all(tuple(row.tolist()) in source_rows for row in exported)
            )
            self.assertEqual(len({tuple(row.tolist()) for row in exported}), config.num_samples)


if __name__ == "__main__":
    unittest.main()
