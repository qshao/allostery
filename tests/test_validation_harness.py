from __future__ import annotations

from allostery.validation.harness import (
    ValidationConfig,
    render_validation_table,
    run_validation,
    validation_report_to_dict,
)


def test_dccm_beats_shuffled_null() -> None:
    config = ValidationConfig(
        n_residues=14, n_couplings=5, coupling_strength=2.0, noise=0.05,
        frames=80, seeds=2, base_seed=0,
    )
    report = run_validation(config, scorers=["dccm", "null"])
    names = {s.name for s in report.scorers}
    assert names == {"dccm", "null"}
    dccm = next(s for s in report.scorers if s.name == "dccm")
    null = next(s for s in report.scorers if s.name == "null")
    assert len(dccm.metrics_per_seed) == 2
    assert dccm.roc_auc_mean > null.roc_auc_mean      # the core rigor claim
    assert report.best_baseline == "dccm"
    assert report.best_scorer == "dccm"


def test_report_dict_and_table_render() -> None:
    config = ValidationConfig(n_residues=12, n_couplings=4, frames=60, seeds=1)
    report = run_validation(config, scorers=["dccm", "contact", "null"])
    data = validation_report_to_dict(report)
    assert data["best_scorer"] in {"dccm", "contact", "null"}
    assert len(data["scorers"]) == 3
    assert "config" in data and data["config"]["n_residues"] == 12
    table = render_validation_table(report)
    assert "dccm" in table and "best scorer" in table


def test_unknown_scorer_raises() -> None:
    import pytest
    config = ValidationConfig(seeds=1)
    with pytest.raises(ValueError, match="unknown scorer"):
        run_validation(config, scorers=["bogus"])
