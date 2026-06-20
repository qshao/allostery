from __future__ import annotations

from allostery.validation.harness import ValidationConfig, run_validation


def test_influence_scorer_runs_and_is_scored() -> None:
    config = ValidationConfig(
        n_residues=12, n_couplings=4, coupling_strength=2.0, noise=0.05,
        frames=48, seeds=1, base_seed=0,
    )
    report = run_validation(config, scorers=["dccm", "influence"])
    names = {s.name for s in report.scorers}
    assert names == {"dccm", "influence"}
    influence = next(s for s in report.scorers if s.name == "influence")
    m = influence.metrics_per_seed[0]
    assert 0.0 <= m.roc_auc <= 1.0          # finite, well-defined metric
    assert isinstance(influence.beats_best_baseline, bool)


def test_cri_and_relational_scorers_complete() -> None:
    config = ValidationConfig(
        n_residues=12, n_couplings=4, frames=48, seeds=1, base_seed=1,
    )
    report = run_validation(config, scorers=["cri", "relational"])
    names = {s.name for s in report.scorers}
    assert names == {"cri", "relational"}
    for s in report.scorers:
        assert 0.0 <= s.roc_auc_mean <= 1.0
        assert len(s.metrics_per_seed) == 1
