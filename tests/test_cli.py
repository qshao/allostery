from __future__ import annotations

import csv
from pathlib import Path
import sys
from types import ModuleType


def _ensure_yaml_module() -> None:
    try:
        __import__("yaml")
        return
    except ModuleNotFoundError:
        pass

    yaml_module = ModuleType("yaml")

    def safe_load(text: str) -> dict[str, object]:
        root: dict[str, object] = {}
        current_section: dict[str, object] | None = None

        for raw_line in text.splitlines():
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if ":" not in line:
                raise ValueError("config must be a YAML mapping")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if indent == 0:
                if value == "":
                    section: dict[str, object] = {}
                    root[key] = section
                    current_section = section
                else:
                    root[key] = value
                    current_section = None
                continue

            if indent == 2 and current_section is not None:
                current_section[key] = value
                continue

            raise ValueError("config must be a YAML mapping")

        return root

    yaml_module.safe_load = safe_load
    sys.modules["yaml"] = yaml_module


_ensure_yaml_module()


def _write_config(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join([*lines, ""]), encoding="utf-8")


def test_cli_run_mode_writes_checkpoint_and_scores_csv(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    config_path = tmp_path / "run.yaml"
    checkpoint_path = tmp_path / "artifacts" / "model.pt"
    score_csv_path = tmp_path / "artifacts" / "scores.csv"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 3",
            "  pair_layers: 4",
            "  dropout: 0.15",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "  verbose: false",
            "scoring:",
            "  top_k: 2",
            "output:",
            f"  model_path: {checkpoint_path}",
            f"  score_csv_path: {score_csv_path}",
        ],
    )

    exit_code = main([str(config_path)])

    assert exit_code == 0
    assert checkpoint_path.exists()
    assert score_csv_path.exists()

    rows = list(csv.DictReader(score_csv_path.open(encoding="utf-8")))
    assert len(rows) == 3

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"trained samples=2 checkpoint={checkpoint_path}",
        f"scored pairs=3 csv={score_csv_path} top_k=2",
        "completed mode=run",
    ]


def test_cli_train_mode_writes_checkpoint_only(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    config_path = tmp_path / "train.yaml"
    checkpoint_path = tmp_path / "artifacts" / "train_model.pt"
    _write_config(
        config_path,
        [
            "mode: train",
            "data:",
            f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 3",
            "  pair_layers: 4",
            "  dropout: 0.15",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "  verbose: false",
            "output:",
            f"  model_path: {checkpoint_path}",
        ],
    )

    exit_code = main([str(config_path)])

    assert exit_code == 0
    assert checkpoint_path.exists()
    assert not (tmp_path / "artifacts" / "scores.csv").exists()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"trained samples=2 checkpoint={checkpoint_path}",
        "completed mode=train",
    ]


def test_cli_score_mode_uses_checkpoint_and_writes_csv(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    train_config_path = tmp_path / "train.yaml"
    checkpoint_path = tmp_path / "artifacts" / "train_model.pt"
    _write_config(
        train_config_path,
        [
            "mode: train",
            "data:",
            f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 3",
            "  pair_layers: 4",
            "  dropout: 0.15",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "  verbose: false",
            "output:",
            f"  model_path: {checkpoint_path}",
        ],
    )
    train_exit_code = main([str(train_config_path)])
    assert train_exit_code == 0
    assert checkpoint_path.exists()

    score_config_path = tmp_path / "score.yaml"
    score_csv_path = tmp_path / "artifacts" / "scores.csv"
    _write_config(
        score_config_path,
        [
            "mode: score",
            "data:",
            f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 3",
            "  pair_layers: 4",
            "  dropout: 0.15",
            "scoring:",
            "  top_k: 2",
            "output:",
            f"  model_path: {checkpoint_path}",
            f"  score_csv_path: {score_csv_path}",
        ],
    )

    exit_code = main([str(score_config_path)])

    assert exit_code == 0
    assert score_csv_path.exists()

    rows = list(csv.DictReader(score_csv_path.open(encoding="utf-8")))
    assert len(rows) == 3

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"trained samples=2 checkpoint={checkpoint_path}",
        "completed mode=train",
        f"scored pairs=3 csv={score_csv_path} top_k=2",
        "completed mode=score",
    ]


def test_cli_cri_run_mode_writes_checkpoint_and_scores_csv(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    config_path = tmp_path / "cri_run.yaml"
    checkpoint_path = tmp_path / "artifacts" / "cri_model.pt"
    score_csv_path = tmp_path / "artifacts" / "cri_scores.csv"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
            "  window_size: 3",
            "  horizon_size: 1",
            "  stride: 1",
            "  time_step: 1.0",
            "  distance_cutoff: 20.0",
            "  max_neighbors: 2",
            "model:",
            "  family: cri",
            "  hidden_dim: 8",
            "  residue_layers: 1",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "  edge_types: 2",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.0",
            "  entropy_weight: 0.0",
            "  no_edge_weight: 0.0",
            "  verbose: false",
            "scoring:",
            "  top_k: 2",
            "output:",
            f"  model_path: {checkpoint_path}",
            f"  score_csv_path: {score_csv_path}",
        ],
    )

    exit_code = main([str(config_path)])

    assert exit_code == 0
    assert checkpoint_path.exists()
    assert score_csv_path.exists()

    rows = list(csv.DictReader(score_csv_path.open(encoding="utf-8")))
    assert len(rows) == 3
    assert "edge_type_probabilities" in rows[0]

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"trained samples=1 checkpoint={checkpoint_path}",
        f"scored pairs=3 csv={score_csv_path} top_k=2",
        "completed mode=run",
    ]


def test_cli_influence_run_mode_writes_checkpoint_and_scores_csv(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main
    from allostery.io.pdb import load_multimodel_pdb

    config_path = tmp_path / "influence_run.yaml"
    checkpoint_path = tmp_path / "artifacts" / "influence_model.pt"
    score_csv_path = tmp_path / "artifacts" / "influence_scores.csv"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
            "  window_size: 3",
            "  horizon_size: 1",
            "  stride: 1",
            "  time_step: 1.0",
            "model:",
            "  family: influence",
            "  hidden_dim: 8",
            "  residue_layers: 1",
            "  pair_layers: 1",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.0",
            "  sparsity_weight: 0.0",
            "scoring:",
            "  top_k: 2",
            "output:",
            f"  model_path: {checkpoint_path}",
            f"  score_csv_path: {score_csv_path}",
        ],
    )

    exit_code = main([str(config_path)])

    assert exit_code == 0
    assert checkpoint_path.exists()
    assert score_csv_path.exists()

    trajectory = load_multimodel_pdb(fixture_path / 'tiny_trajectory.pdb')
    num_residues = trajectory.coordinates.shape[1]
    expected_pairs = num_residues * (num_residues - 1) // 2
    rows = list(csv.DictReader(score_csv_path.open(encoding="utf-8")))
    assert len(rows) == expected_pairs

    captured = capsys.readouterr()
    assert "trained samples=1" in captured.out
    assert "completed mode=run" in captured.out
