from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from allostery.io.trajectory import load_trajectory
from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model
from allostery.pipeline.influence_score import score_influence_trajectory
from allostery.pipeline.influence_train import train_influence_model
from allostery.pipeline.score import score_trajectory
from allostery.pipeline.train import train_model
from allostery.validation.baselines import (
    contact_frequency_scores,
    dccm_scores,
    mutual_information_scores,
    shuffled_null_scores,
)
from allostery.validation.metrics import ScoreMetrics, evaluate_scores
from allostery.validation.synthetic import generate_planted_system

_BASELINE_SCORERS = ("dccm", "mi", "contact", "null")
_MODEL_SCORERS = ("influence", "cri", "relational")
ALL_SCORERS = _BASELINE_SCORERS + _MODEL_SCORERS

_MODEL_WINDOW = 3
_MODEL_STRIDE = 1
_MODEL_HIDDEN = 8
_MODEL_EPOCHS = 2
_MODEL_LR = 1e-2


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    n_residues: int = 24
    n_couplings: int = 8
    coupling_strength: float = 1.0
    noise: float = 0.05
    frames: int = 128
    seeds: int = 3
    base_seed: int = 0
    min_sequence_separation: int = 2


@dataclass(frozen=True, slots=True)
class ScorerResult:
    name: str
    metrics_per_seed: list[ScoreMetrics]
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    precision_at_k_mean: float
    beats_best_baseline: bool = False


@dataclass(frozen=True, slots=True)
class ValidationReport:
    scorers: list[ScorerResult]
    best_scorer: str
    best_baseline: str
    config: dict[str, Any] = field(default_factory=dict)


def _score_influence(pdb_path: Path, config: ValidationConfig, seed: int) -> list[dict]:
    result = train_influence_model(
        pdb_path=pdb_path,
        window_size=_MODEL_WINDOW,
        stride=_MODEL_STRIDE,
        time_step=1.0,
        hidden_dim=_MODEL_HIDDEN,
        num_encoder_layers=2,
        dropout=0.0,
        epochs=_MODEL_EPOCHS,
        learning_rate=_MODEL_LR,
        sparsity_weight=0.0,
        min_sequence_separation=config.min_sequence_separation,
        validation_fraction=0.0,
        patience=0,
        seed=seed,
        verbose=False,
    )
    return score_influence_trajectory(
        model=result.model,
        pdb_path=pdb_path,
        window_size=_MODEL_WINDOW,
        stride=_MODEL_STRIDE,
        time_step=1.0,
        min_sequence_separation=config.min_sequence_separation,
    )


def _score_cri(pdb_path: Path, config: ValidationConfig, seed: int) -> list[dict]:
    # Fully connect the graph so long-range planted couplings are reachable.
    result = train_cri_model(
        pdb_path=pdb_path,
        window_size=_MODEL_WINDOW,
        stride=_MODEL_STRIDE,
        time_step=1.0,
        distance_cutoff=1.0e6,
        max_neighbors=config.n_residues,
        edge_types=2,
        hidden_dim=_MODEL_HIDDEN,
        dropout=0.0,
        epochs=_MODEL_EPOCHS,
        learning_rate=_MODEL_LR,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        min_sequence_separation=config.min_sequence_separation,
        validation_fraction=0.0,
        patience=0,
        seed=seed,
        verbose=False,
    )
    return score_cri_trajectory(
        model=result.model,
        pdb_path=pdb_path,
        window_size=_MODEL_WINDOW,
        stride=_MODEL_STRIDE,
        time_step=1.0,
        distance_cutoff=1.0e6,
        max_neighbors=config.n_residues,
        min_sequence_separation=config.min_sequence_separation,
    )


def _score_relational(pdb_path: Path, config: ValidationConfig, seed: int) -> list[dict]:
    result = train_model(
        pdb_path=pdb_path,
        window_size=_MODEL_WINDOW,
        horizon_size=1,
        stride=_MODEL_STRIDE,
        hidden_dim=_MODEL_HIDDEN,
        residue_layers=2,
        pair_layers=1,
        dropout=0.0,
        epochs=_MODEL_EPOCHS,
        learning_rate=_MODEL_LR,
        consistency_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=seed,
        verbose=False,
    )
    return score_trajectory(
        model=result.model,
        pdb_path=pdb_path,
        window_size=_MODEL_WINDOW,
        horizon_size=1,
        stride=_MODEL_STRIDE,
    )


def _run_one_scorer(
    name: str,
    trajectory: Any,
    pdb_path: Path,
    config: ValidationConfig,
    seed: int,
) -> list[dict]:
    sep = config.min_sequence_separation
    if name == "dccm":
        return dccm_scores(trajectory, min_sequence_separation=sep)
    if name == "mi":
        return mutual_information_scores(trajectory, min_sequence_separation=sep)
    if name == "contact":
        return contact_frequency_scores(trajectory, min_sequence_separation=sep)
    if name == "null":
        return shuffled_null_scores(trajectory, seed=seed, min_sequence_separation=sep)
    if name == "influence":
        return _score_influence(pdb_path, config, seed)
    if name == "cri":
        return _score_cri(pdb_path, config, seed)
    if name == "relational":
        return _score_relational(pdb_path, config, seed)
    raise ValueError(f"scorer {name!r} is not runnable in this build")


def _aggregate(name: str, metrics: list[ScoreMetrics]) -> ScorerResult:
    roc = np.array([m.roc_auc for m in metrics], dtype=np.float64)
    pr = np.array([m.pr_auc for m in metrics], dtype=np.float64)
    prec = np.array([m.precision_at_k for m in metrics], dtype=np.float64)
    return ScorerResult(
        name=name,
        metrics_per_seed=metrics,
        roc_auc_mean=float(roc.mean()),
        roc_auc_std=float(roc.std()),
        pr_auc_mean=float(pr.mean()),
        pr_auc_std=float(pr.std()),
        precision_at_k_mean=float(prec.mean()),
    )


def run_validation(
    config: ValidationConfig,
    *,
    scorers: list[str] | None = None,
) -> ValidationReport:
    selected = list(scorers) if scorers is not None else list(ALL_SCORERS)
    invalid = [name for name in selected if name not in ALL_SCORERS]
    if invalid:
        raise ValueError(f"unknown scorer(s): {invalid}; valid: {sorted(ALL_SCORERS)}")

    per_scorer: dict[str, list[ScoreMetrics]] = {name: [] for name in selected}

    for seed_index in range(config.seeds):
        seed = config.base_seed + seed_index
        with tempfile.TemporaryDirectory() as tempdir:
            pdb_path = Path(tempdir) / "planted.pdb"
            system = generate_planted_system(
                pdb_path,
                n_residues=config.n_residues,
                n_couplings=config.n_couplings,
                coupling_strength=config.coupling_strength,
                noise=config.noise,
                frames=config.frames,
                seed=seed,
                min_sequence_separation=config.min_sequence_separation,
            )
            trajectory = load_trajectory(pdb_path)
            for name in selected:
                pair_scores = _run_one_scorer(name, trajectory, pdb_path, config, seed)
                metrics = evaluate_scores(
                    pair_scores,
                    system.coupling_matrix,
                    min_sequence_separation=config.min_sequence_separation,
                )
                per_scorer[name].append(metrics)

    results = [_aggregate(name, per_scorer[name]) for name in selected]

    baseline_results = [r for r in results if r.name in _BASELINE_SCORERS]
    best_baseline = (
        max(baseline_results, key=lambda r: r.roc_auc_mean).name
        if baseline_results
        else ""
    )
    best_baseline_mean = max(
        (r.roc_auc_mean for r in baseline_results), default=float("-inf")
    )

    final = [
        replace(
            r,
            beats_best_baseline=(
                r.name in _MODEL_SCORERS and r.roc_auc_mean > best_baseline_mean
            ),
        )
        for r in results
    ]
    final.sort(key=lambda r: r.roc_auc_mean, reverse=True)
    best_scorer = final[0].name if final else ""

    return ValidationReport(
        scorers=final,
        best_scorer=best_scorer,
        best_baseline=best_baseline,
        config=asdict(config),
    )


def render_validation_table(report: ValidationReport) -> str:
    lines = [
        f"{'scorer':<12}  {'roc_auc':<15}  {'pr_auc':<15}  {'prec@k':<8}  beats_baseline",
    ]
    for s in report.scorers:
        lines.append(
            f"{s.name:<12}  "
            f"{s.roc_auc_mean:.3f}±{s.roc_auc_std:.3f}    "
            f"{s.pr_auc_mean:.3f}±{s.pr_auc_std:.3f}    "
            f"{s.precision_at_k_mean:.3f}     "
            f"{s.beats_best_baseline}"
        )
    lines.append(f"best scorer: {report.best_scorer}  |  best baseline: {report.best_baseline}")
    return "\n".join(lines)


def validation_report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return {
        "best_scorer": report.best_scorer,
        "best_baseline": report.best_baseline,
        "config": report.config,
        "scorers": [
            {
                "name": s.name,
                "roc_auc_mean": s.roc_auc_mean,
                "roc_auc_std": s.roc_auc_std,
                "pr_auc_mean": s.pr_auc_mean,
                "pr_auc_std": s.pr_auc_std,
                "precision_at_k_mean": s.precision_at_k_mean,
                "beats_best_baseline": s.beats_best_baseline,
                "metrics_per_seed": [
                    {
                        "roc_auc": m.roc_auc,
                        "pr_auc": m.pr_auc,
                        "precision_at_k": m.precision_at_k,
                        "recall_at_k": m.recall_at_k,
                        "n_true": m.n_true,
                        "n_pairs": m.n_pairs,
                    }
                    for m in s.metrics_per_seed
                ],
            }
            for s in report.scorers
        ],
    }


__all__ = [
    "ALL_SCORERS",
    "_BASELINE_SCORERS",
    "_MODEL_SCORERS",
    "ScorerResult",
    "ValidationConfig",
    "ValidationReport",
    "render_validation_table",
    "run_validation",
    "validation_report_to_dict",
]
