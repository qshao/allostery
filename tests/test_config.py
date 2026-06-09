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
    assert config.training.batch_size == 4
    assert config.scoring is not None
    assert config.scoring.top_k == 5
    assert config.output.model_path == config_path.parent.joinpath("outputs/model.pt").resolve()


def test_load_config_parses_cri_model_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "cri.yaml"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 3",
            "  horizon_size: 1",
            "  stride: 1",
            "  time_step: 1.0",
            "  distance_cutoff: 20.0",
            "  max_neighbors: 2",
            "  min_sequence_separation: 2",
            "  preprocess: center",
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
            "  batch_size: 6",
            "  entropy_weight: 0.0",
            "  no_edge_weight: 0.0",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/cri.pt",
            "  score_csv_path: outputs/cri_scores.csv",
        ],
    )

    config = load_config(config_path)

    assert config.model.family == "cri"
    assert config.model.edge_types == 2
    assert config.data.time_step == 1.0
    assert config.data.distance_cutoff == 20.0
    assert config.data.max_neighbors == 2
    assert config.data.min_sequence_separation == 2
    assert config.data.preprocess == "center"
    assert config.training is not None
    assert config.training.batch_size == 6
    assert config.training.entropy_weight == 0.0
    assert config.training.no_edge_weight == 0.0


def test_config_error_message_includes_filename(tmp_path: Path) -> None:
    from allostery.config import load_config
    bad_config = tmp_path / "bad.yaml"
    _write_config(
        bad_config,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 0",
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
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_config)
    assert "bad.yaml" in str(exc_info.value)


def test_config_error_includes_got_value(tmp_path: Path) -> None:
    from allostery.config import load_config
    bad_config = tmp_path / "got.yaml"
    _write_config(
        bad_config,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 0",
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
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_config)
    assert "got 0" in str(exc_info.value)


def test_config_error_reports_multiple_errors_at_once(tmp_path: Path) -> None:
    from allostery.config import load_config
    bad_config = tmp_path / "multi.yaml"
    _write_config(
        bad_config,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 0",
            "  horizon_size: 0",
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
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_config)
    msg = str(exc_info.value)
    assert "window_size" in msg
    assert "horizon_size" in msg


def test_config_error_is_value_error_subclass() -> None:
    from allostery.config import ConfigError
    assert issubclass(ConfigError, ValueError)


def test_unknown_key_in_training_prints_warning_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from allostery.config import load_config
    config_path = tmp_path / "typo.yaml"
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
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learnig_rate: 0.001",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )
    load_config(config_path)
    captured = capsys.readouterr()
    assert "learnig_rate" in captured.err
    assert "warning" in captured.err.lower()


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
