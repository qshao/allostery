from pathlib import Path
import sys
from types import ModuleType

import pytest


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

from allostery.config import load_config


FIXTURE_PDB = (Path(__file__).resolve().parent / "fixtures" / "tiny_trajectory.pdb").resolve()


def _write_config(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join([*lines, ""]), encoding="utf-8")


def test_load_config_parses_minimal_run_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 3",
            "  pair_layers: 4",
            "  dropout: 0.1",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )

    config = load_config(config_path)

    assert config.mode == "run"
    assert config.data.window_size == 1
    assert config.model.hidden_dim == 8
    assert config.model.residue_layers == 3
    assert config.model.pair_layers == 4
    assert config.model.dropout == 0.1
    assert config.training is not None
    assert config.training.epochs == 1
    assert config.scoring is not None
    assert config.scoring.top_k == 5
    assert config.output.model_path == config_path.parent.joinpath("outputs/model.pt").resolve()


def test_load_config_rejects_invalid_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _write_config(
        config_path,
        [
            "mode: nope",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )

    with pytest.raises(ValueError, match="mode"):
        load_config(config_path)


def test_load_config_requires_model_path_for_score_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "score.yaml"
    _write_config(
        config_path,
        [
            "mode: score",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  score_csv_path: outputs/scores.csv",
        ],
    )

    with pytest.raises(ValueError, match="model_path"):
        load_config(config_path)


def test_load_config_rejects_missing_pdb_path(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_pdb.yaml"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            "  pdb_path: does-not-exist.pdb",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )

    with pytest.raises(ValueError, match="pdb_path"):
        load_config(config_path)


def test_load_config_allows_score_mode_without_training_section(tmp_path: Path) -> None:
    pdb_path = tmp_path / "score_input.pdb"
    pdb_path.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")
    config_path = tmp_path / "score.yaml"
    _write_config(
        config_path,
        [
            "mode: score",
            "data:",
            "  pdb_path: score_input.pdb",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )

    config = load_config(config_path)

    assert config.mode == "score"
    assert config.training is None
    assert config.scoring is not None
    assert config.scoring.top_k == 5


def test_load_config_allows_train_mode_without_scoring_section(tmp_path: Path) -> None:
    pdb_path = tmp_path / "train_input.pdb"
    pdb_path.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")
    config_path = tmp_path / "train.yaml"
    _write_config(
        config_path,
        [
            "mode: train",
            "data:",
            "  pdb_path: train_input.pdb",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "output:",
            "  model_path: outputs/model.pt",
        ],
    )

    config = load_config(config_path)

    assert config.mode == "train"
    assert config.training is not None
    assert config.training.epochs == 1
    assert config.scoring is None


def test_load_config_resolves_relative_paths_from_config_location(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    config_dir.mkdir()
    data_dir.mkdir()
    output_dir.mkdir()
    pdb_path = data_dir / "input.pdb"
    pdb_path.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")
    config_path = config_dir / "config.yaml"
    _write_config(
        config_path,
        [
            "mode: score",
            "data:",
            "  pdb_path: ../data/input.pdb",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: ../outputs/model.pt",
            "  score_csv_path: ../outputs/scores.csv",
        ],
    )

    config = load_config(config_path)

    assert config.data.pdb_path == pdb_path.resolve()
    assert config.output.model_path == (output_dir / "model.pt").resolve()
    assert config.output.score_csv_path == (output_dir / "scores.csv").resolve()
