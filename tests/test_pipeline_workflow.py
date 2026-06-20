from __future__ import annotations

from pathlib import Path

import pytest

from allostery.config import load_config
from allostery.pipeline.workflow import WorkflowError, run_workflow


def _cfg(tmp_path: Path, fixture_path: Path, extra: list[str]) -> Path:
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
        "  verbose: false",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {tmp_path / 'model.pt'}",
        f"  score_csv_path: {tmp_path / 'scores.csv'}",
    ] + extra
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _FakeBackend:
    def generate_json(self, system, user, schema):
        return {"summary": "s", "mechanism_hypothesis": "m", "key_residues": [],
                "confidence": "low", "parametric": False, "caveats": "c"}


def test_full_workflow_runs_all_stages(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_cfg(tmp_path, fixture_path, [
        "analyze:", "  top_k: 3",
        "interpret:", "  llm: ollama", "  top_hubs: 3",
    ]))
    stages: list[str] = []
    result = run_workflow(config, backend=_FakeBackend(), progress=stages.append)
    assert result.command == "workflow"
    assert result.data["stages"] == ["train", "score", "analyze", "interpret"]
    assert stages == ["train", "score", "analyze", "interpret"]
    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "scores.network.txt").exists()
    assert (tmp_path / "scores.interpret.json").exists()
    assert (tmp_path / "scores.interpret.md").exists()


def test_workflow_without_post_sections_just_runs_pipeline(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_cfg(tmp_path, fixture_path, []))
    result = run_workflow(config)
    assert result.data["stages"] == ["train", "score"]
    assert (tmp_path / "scores.csv").exists()


def test_workflow_backend_failure_preserves_prior_artifacts(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_cfg(tmp_path, fixture_path, [
        "analyze:", "  top_k: 3",
        "interpret:", "  llm: ollama",
    ]))

    class _Boom:
        def generate_json(self, system, user, schema):
            raise ImportError("backend unavailable")

    with pytest.raises(WorkflowError) as exc:
        run_workflow(config, backend=_Boom())
    assert exc.value.stage == "interpret"
    assert (tmp_path / "scores.csv").exists()         # preserved
    assert (tmp_path / "scores.network.txt").exists()  # preserved
    assert isinstance(exc.value.__cause__, ImportError)
