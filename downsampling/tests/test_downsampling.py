"""SPAD ``xyzi`` 降采样器的 CPU 单元测试。"""

from __future__ import annotations

import copy
import unittest

import torch

from downsampling import (
    APESLocalXYZI,
    IntensityWeightedRandomSampler,
    SampleNetXYZI,
    assert_unique_indices,
    available_downsamplers,
    build_downsampler,
    normalize_xyzi,
)
from downsampling.common import deduplicate_and_fill_indices


def make_points(batch_size: int = 2, num_points: int = 48) -> torch.Tensor:
    """生成坐标和强度量级不同的确定性测试点云。"""

    generator = torch.Generator().manual_seed(20260715)
    points = torch.rand(batch_size, num_points, 4, generator=generator)
    points[..., 0] *= 63.0
    points[..., 1] *= 63.0
    points[..., 2] *= 108.0
    points[..., 3] *= 800.0
    return points


class CommonToolTests(unittest.TestCase):
    """公共归一化与索引工具测试。"""

    def test_normalize_xyzi_is_finite_and_bounded(self) -> None:
        points = make_points()
        points[:, :, 0] = 7.0
        normalized = normalize_xyzi(points)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertGreaterEqual(float(normalized.amin()), 0.0)
        self.assertLessEqual(float(normalized.amax()), 1.0)
        self.assertTrue(torch.equal(normalized[:, :, 0], torch.zeros_like(points[:, :, 0])))

    def test_deduplicate_and_fill_indices(self) -> None:
        primary = torch.tensor([[1, 1, 3, 3]], dtype=torch.long)
        priority = torch.tensor([[0.5, 0.4, 0.1, 0.9, 0.2]])
        indices = deduplicate_and_fill_indices(primary, priority, num_samples=4)
        assert_unique_indices(indices)
        self.assertEqual(indices.shape, (1, 4))
        self.assertIn(1, indices[0].tolist())
        self.assertIn(3, indices[0].tolist())


class IntensityWeightedRandomSamplerTests(unittest.TestCase):
    """I-WRS 测试。"""

    def test_shape_uniqueness_source_and_seed(self) -> None:
        points = make_points()
        sampler = IntensityWeightedRandomSampler(num_samples=12, gamma=0.5)

        output_a = sampler(
            points,
            generator=torch.Generator().manual_seed(42),
        )
        output_b = sampler(
            points,
            generator=torch.Generator().manual_seed(42),
        )

        self.assertEqual(output_a.points.shape, (2, 12, 4))
        self.assertEqual(output_a.indices.shape, (2, 12))
        self.assertTrue(torch.equal(output_a.indices, output_b.indices))
        assert_unique_indices(output_a.indices)

        expected = torch.gather(
            points,
            1,
            output_a.indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        self.assertTrue(torch.equal(output_a.points, expected))


class SampleNetXYZITests(unittest.TestCase):
    """SampleNet-XYZI 的硬输出、软投影和梯度测试。"""

    @staticmethod
    def make_model() -> SampleNetXYZI:
        return SampleNetXYZI(
            num_samples=12,
            projection_neighbors=4,
            feature_dim=32,
            hidden_dim=64,
            distance_chunk_size=5,
        )

    def test_forward_backward_and_source_points(self) -> None:
        points = make_points()
        model = self.make_model().train()
        output = model(points)

        self.assertEqual(output.points.shape, (2, 12, 4))
        self.assertEqual(output.projected_points.shape, (2, 12, 4))
        self.assertEqual(output.generated_points.shape, (2, 12, 3))
        assert_unique_indices(output.indices)

        expected = torch.gather(
            points,
            1,
            output.indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        self.assertTrue(torch.equal(output.points, expected))

        loss = output.projected_points.mean() + model.sampler_loss(output)
        loss.backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)
        self.assertIsNotNone(model.decoder[-1].weight.grad)
        self.assertIsNotNone(model._temperature_unconstrained.grad)

    def test_state_dict_roundtrip_is_deterministic_in_eval(self) -> None:
        points = make_points()
        model = self.make_model().eval()
        clone = self.make_model().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()))

        with torch.no_grad():
            output_a = model(points)
            output_b = clone(points)
        self.assertTrue(torch.equal(output_a.indices, output_b.indices))
        self.assertTrue(torch.equal(output_a.points, output_b.points))


class APESLocalXYZITests(unittest.TestCase):
    """APES-Local-XYZI 的硬输出和可微特征测试。"""

    @staticmethod
    def make_model() -> APESLocalXYZI:
        return APESLocalXYZI(
            num_samples=12,
            num_neighbors=8,
            embedding_dim=32,
            knn_chunk_size=13,
        )

    def test_forward_backward_and_source_points(self) -> None:
        points = make_points()
        model = self.make_model().train()
        output = model(points)

        self.assertEqual(output.points.shape, (2, 12, 4))
        self.assertEqual(output.features.shape, (2, 32, 12))
        self.assertEqual(output.scores.shape, (2, 48))
        assert_unique_indices(output.indices)

        expected = torch.gather(
            points,
            1,
            output.indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        self.assertTrue(torch.equal(output.points, expected))

        output.features.square().mean().backward()
        self.assertIsNotNone(model.embedding[0].weight.grad)
        self.assertIsNotNone(model.query_conv.weight.grad)
        self.assertIsNotNone(model.key_conv.weight.grad)
        self.assertIsNotNone(model.value_conv.weight.grad)

    def test_state_dict_roundtrip_is_deterministic_in_eval(self) -> None:
        points = make_points()
        model = self.make_model().eval()
        clone = self.make_model().eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()))

        with torch.no_grad():
            output_a = model(points)
            output_b = clone(points)
        self.assertTrue(torch.equal(output_a.indices, output_b.indices))
        self.assertTrue(torch.equal(output_a.points, output_b.points))


class FactoryTests(unittest.TestCase):
    """统一工厂测试。"""

    def test_available_names_and_aliases(self) -> None:
        self.assertEqual(
            available_downsamplers(),
            ("i_wrs", "samplenet_xyzi", "apes_local_xyzi"),
        )
        self.assertIsInstance(
            build_downsampler("i-wrs", num_samples=8),
            IntensityWeightedRandomSampler,
        )
        self.assertIsInstance(
            build_downsampler(
                "samplenet",
                num_samples=8,
                feature_dim=16,
                hidden_dim=32,
            ),
            SampleNetXYZI,
        )
        self.assertIsInstance(
            build_downsampler(
                "apes-local",
                num_samples=8,
                num_neighbors=4,
                embedding_dim=16,
            ),
            APESLocalXYZI,
        )

    def test_unknown_name_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported downsampler"):
            build_downsampler("not-a-sampler", num_samples=8)


if __name__ == "__main__":
    unittest.main()
