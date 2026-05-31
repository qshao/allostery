from __future__ import annotations

import unittest


class ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from allostery.models.encoders import PairEncoder, ResidueEncoder
            from allostery.models.relational import RelationalScoreModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"torch is required for model tests: {exc}") from exc

        cls.torch = torch
        cls.PairEncoder = PairEncoder
        cls.ResidueEncoder = ResidueEncoder
        cls.RelationalScoreModel = RelationalScoreModel

    def test_encoders_preserve_batch_axes_and_project_features(self) -> None:
        residue_encoder = self.ResidueEncoder(input_dim=10, hidden_dim=8)
        pair_encoder = self.PairEncoder(input_dim=5, hidden_dim=8)

        residue_features = self.torch.randn(2, 4, 10)
        pair_features = self.torch.randn(2, 6, 5)

        residue_embedding = residue_encoder(residue_features)
        pair_embedding = pair_encoder(pair_features)

        self.assertEqual(residue_embedding.shape, (2, 4, 8))
        self.assertEqual(pair_embedding.shape, (2, 6, 8))

    def test_relational_model_outputs_scores_and_compact_target_predictions(self) -> None:
        model = self.RelationalScoreModel(residue_dim=10, pair_dim=5, hidden_dim=8)
        residue_features = self.torch.randn(2, 3, 10)
        pair_index = self.torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=self.torch.int64)
        pair_features = self.torch.randn(2, 3, 5)

        output = model(residue_features, pair_index, pair_features)

        self.assertEqual(set(output), {"scores", "target_pred"})
        self.assertEqual(output["scores"].shape, (2, 3))
        self.assertEqual(output["target_pred"].shape, (2, 3, 3))

    def test_relational_model_is_symmetric_with_swapped_pair_order(self) -> None:
        model = self.RelationalScoreModel(residue_dim=10, pair_dim=5, hidden_dim=8)
        residue_features = self.torch.randn(1, 3, 10)
        pair_features = self.torch.randn(1, 2, 5)
        pair_index = self.torch.tensor([[0, 1], [0, 2]], dtype=self.torch.int64)
        swapped_pair_index = self.torch.tensor([[1, 0], [2, 0]], dtype=self.torch.int64)

        output = model(residue_features, pair_index, pair_features)
        swapped_output = model(residue_features, swapped_pair_index, pair_features)

        self.assertTrue(self.torch.allclose(output["scores"], swapped_output["scores"]))
        self.assertTrue(
            self.torch.allclose(output["target_pred"], swapped_output["target_pred"])
        )


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromTestCase(ModelTestCase)
