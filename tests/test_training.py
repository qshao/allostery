from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TrainingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch is required for training tests: {exc}")
        self.torch = torch
        self.fixture_path = Path(__file__).parent / "fixtures" / "tiny_trajectory.pdb"

    def test_training_objectives_return_scalar_losses(self) -> None:
        from allostery.training.objectives import consistency_loss, future_summary_loss

        prediction = self.torch.randn(1, 3, 3)
        target = self.torch.randn(1, 3, 3)
        current_scores = self.torch.randn(1, 3)
        next_scores = self.torch.randn(1, 3)

        self.assertEqual(future_summary_loss(prediction, target).ndim, 0)
        self.assertEqual(consistency_loss(current_scores, next_scores).ndim, 0)

    def test_train_model_smoke(self) -> None:
        from allostery.data import build_training_samples
        from allostery.io.pdb import load_multimodel_pdb
        from allostery.pipeline.train import train_model

        trajectory = load_multimodel_pdb(self.fixture_path)
        samples = build_training_samples(
            trajectory.coordinates,
            window_size=1,
            horizon_size=1,
            stride=1,
        )

        result = train_model(
            pdb_path=self.fixture_path,
            window_size=1,
            horizon_size=1,
            stride=1,
            hidden_dim=8,
            epochs=1,
            learning_rate=1e-3,
            consistency_weight=0.25,
        )

        self.assertEqual(result.num_samples, 2)
        self.assertGreaterEqual(result.last_loss, 0.0)
        self.assertEqual(
            result.model.residue_encoder.network[0].in_features,
            samples[0].residue_features.shape[-1],
        )
        self.assertEqual(
            result.model.pair_encoder.network[0].in_features,
            samples[0].pair_features.shape[-1],
        )
        self.assertEqual(result.model.target_head[-1].out_features, samples[0].targets.shape[-1])

    def test_train_model_writes_checkpoint_and_loaded_model_scores_same_pairs(self) -> None:
        from allostery.io.checkpoint import load_checkpoint
        from allostery.pipeline.score import load_scoring_model, score_trajectory
        from allostery.pipeline.train import train_model

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            result = train_model(
                pdb_path=self.fixture_path,
                window_size=1,
                horizon_size=1,
                stride=1,
                hidden_dim=8,
                residue_layers=3,
                pair_layers=4,
                dropout=0.15,
                epochs=1,
                learning_rate=1e-3,
                consistency_weight=0.25,
                checkpoint_path=checkpoint_path,
                config_snapshot={
                    "mode": "train",
                    "model": {
                        "hidden_dim": 8,
                        "residue_layers": 3,
                        "pair_layers": 4,
                        "dropout": 0.15,
                    },
                },
            )

            checkpoint = load_checkpoint(checkpoint_path)
            loaded_model = load_scoring_model(checkpoint_path)

            self.assertEqual(result.num_samples, 2)
            self.assertEqual(checkpoint.residue_dim, 10)
            self.assertEqual(checkpoint.pair_dim, 5)
            self.assertEqual(checkpoint.hidden_dim, 8)
            self.assertEqual(checkpoint.residue_layers, 3)
            self.assertEqual(checkpoint.pair_layers, 4)
            self.assertEqual(checkpoint.dropout, 0.15)
            self.assertEqual(checkpoint.target_dim, 3)
            self.assertEqual(
                checkpoint.config,
                {
                    "mode": "train",
                    "model": {
                        "hidden_dim": 8,
                        "residue_layers": 3,
                        "pair_layers": 4,
                        "dropout": 0.15,
                    },
                },
            )

            direct_scores = score_trajectory(
                model=result.model,
                pdb_path=self.fixture_path,
                window_size=1,
                horizon_size=1,
                stride=1,
            )
            loaded_scores = score_trajectory(
                model=loaded_model,
                pdb_path=self.fixture_path,
                window_size=1,
                horizon_size=1,
                stride=1,
            )
            self.assertEqual(direct_scores, loaded_scores)

    def test_train_model_errors_when_no_windows_are_available(self) -> None:
        from allostery.pipeline.train import train_model

        with self.assertRaisesRegex(ValueError, "did not yield any training windows"):
            train_model(
                pdb_path=self.fixture_path,
                window_size=3,
                horizon_size=1,
                stride=1,
                hidden_dim=8,
                epochs=1,
            )

    def test_score_trajectory_ranks_average_pair_scores(self) -> None:
        from allostery.data import build_training_samples
        from allostery.io.pdb import load_multimodel_pdb
        from allostery.pipeline.score import score_trajectory
        from allostery.pipeline.train import train_model

        training_result = train_model(
            pdb_path=self.fixture_path,
            window_size=1,
            horizon_size=1,
            stride=1,
            hidden_dim=8,
            epochs=1,
            learning_rate=1e-3,
            consistency_weight=0.25,
        )

        ranked_scores = score_trajectory(
            model=training_result.model,
            pdb_path=self.fixture_path,
            window_size=1,
            horizon_size=1,
            stride=1,
        )

        trajectory = load_multimodel_pdb(self.fixture_path)
        samples = build_training_samples(
            trajectory.coordinates,
            window_size=1,
            horizon_size=1,
            stride=1,
        )
        pair_index = self.torch.as_tensor(samples[0].pair_index, dtype=self.torch.int64)

        training_result.model.eval()
        with self.torch.no_grad():
            window_scores = []
            for sample in samples:
                output = training_result.model(
                    self.torch.as_tensor(sample.residue_features[None, ...], dtype=self.torch.float32),
                    pair_index,
                    self.torch.as_tensor(sample.pair_features[None, ...], dtype=self.torch.float32),
                )
                window_scores.append(output["scores"].squeeze(0))
        averaged_scores = self.torch.stack(window_scores, dim=0).mean(dim=0)
        expected_pairs = sorted(
            (
                {
                    "residue_i": {
                        "index": int(left_index),
                        "chain_id": trajectory.residues[int(left_index)].chain_id,
                        "residue_number": trajectory.residues[int(left_index)].residue_number,
                        "name": trajectory.residues[int(left_index)].name,
                    },
                    "residue_j": {
                        "index": int(right_index),
                        "chain_id": trajectory.residues[int(right_index)].chain_id,
                        "residue_number": trajectory.residues[int(right_index)].residue_number,
                        "name": trajectory.residues[int(right_index)].name,
                    },
                    "score": float(score.item()),
                }
                for (left_index, right_index), score in zip(samples[0].pair_index, averaged_scores, strict=True)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )

        self.assertEqual(len(ranked_scores), len(expected_pairs))
        self.assertEqual(
            [entry["score"] for entry in ranked_scores],
            sorted((entry["score"] for entry in ranked_scores), reverse=True),
        )
        for actual, expected in zip(ranked_scores, expected_pairs, strict=True):
            self.assertEqual(actual["residue_i"], expected["residue_i"])
            self.assertEqual(actual["residue_j"], expected["residue_j"])
            self.assertAlmostEqual(actual["score"], expected["score"], places=6)
