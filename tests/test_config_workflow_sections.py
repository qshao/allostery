from __future__ import annotations

from pathlib import Path

import pytest

from allostery.config import ConfigError, load_config


def _base(tmp_path: Path, fixture_path: Path, extra: list[str]) -> Path:
    checkpoint = tmp_path / "model.pt"
    scores = tmp_path / "scores.csv"
    lines = [
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 2",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 1",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {checkpoint}",
        f"  score_csv_path: {scores}",
    ] + extra
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_sections_parse(tmp_path: Path, fixture_path: Path) -> None:
    cfg = load_config(_base(tmp_path, fixture_path, [
        "analyze:",
        "  top_k: 10",
        "  top_hubs: 4",
        "interpret:",
        "  llm: none",
        "  top_hubs: 6",
    ]))
    assert cfg.analyze.top_k == 10
    assert cfg.analyze.top_hubs == 4
    assert cfg.interpret.llm == "none"
    assert cfg.interpret.top_hubs == 6


def test_no_sections_default_to_none(tmp_path: Path, fixture_path: Path) -> None:
    cfg = load_config(_base(tmp_path, fixture_path, []))
    assert cfg.analyze is None
    assert cfg.interpret is None


def test_bad_llm_enum_rejected(tmp_path: Path, fixture_path: Path) -> None:
    with pytest.raises(ConfigError, match="interpret.llm"):
        load_config(_base(tmp_path, fixture_path, ["interpret:", "  llm: gpt5"]))


def test_lone_source_rejected(tmp_path: Path, fixture_path: Path) -> None:
    with pytest.raises(ConfigError, match="source"):
        load_config(_base(tmp_path, fixture_path, ["analyze:", "  source: A:1 GLY"]))


def test_train_mode_with_analyze_section_rejected(tmp_path: Path, fixture_path: Path) -> None:
    lines = [
        "mode: train",  # train only, no scoring
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 2",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 1",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "output:",
        f"  model_path: {tmp_path / 'model.pt'}",
        "analyze:",
        "  top_k: 10",
    ]
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="train"):
        load_config(path)
