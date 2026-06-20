# Validation Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scientific validation harness that measures whether the allostery scorers recover known residue–residue coupling — synthetic planted-coupling systems for exact ground truth, classical baselines for comparison, ranking metrics for accuracy — surfaced as an `allostery validate` subcommand.

**Architecture:** A new `src/allostery/validation/` package with four leaf modules (`synthetic`, `baselines`, `metrics`, `harness`). Every scorer reduces to the existing `{residue_i, residue_j, score}` pair-score shape, so the metrics layer is scorer-agnostic (it keys on residue `index`). The harness generates a planted system per seed, runs each scorer, and aggregates metrics into a `ValidationReport`. A thin `validate` CLI branch renders it through the existing `Result`/`format_result` layer.

**Tech Stack:** Python 3.11+, numpy (generator/baselines/metrics), the existing torch training/scoring pipeline (model scorers), stdlib `argparse`/`json`/`tempfile`/`dataclasses`. No new dependencies.

## Global Constraints

- `from __future__ import annotations` at the top of every new module (repo convention).
- No new dependencies — numpy and torch are already present; `--json` uses stdlib `json`.
- No changes to model architectures, objectives, or scoring math — this round only *measures*.
- Every scorer returns `list[PairScore]` where each item is `{"residue_i": ResidueIdentifier, "residue_j": ResidueIdentifier, "score": float}` and `ResidueIdentifier` is a dict with at least `{"index": int, "chain_id": str, "residue_number": int, "name": str}`. Metrics key on `item["residue_i"]["index"]` / `item["residue_j"]["index"]`.
- Residue label/identifier convention follows the existing `_residue_identifier` helpers exactly.
- Sequence-separation masking threshold default is `2` everywhere (generator ground truth, baselines, metrics) so only non-trivial coupling is scored.
- Synthetic PDBs use the exact fixed-column ATOM format already proven against `load_multimodel_pdb` (see Task 1) — do not invent a new column layout.
- Determinism: all randomness flows through `numpy.random.default_rng(seed)` or the existing `seed_everything`; tests assert reproducibility. No real network, no real MD, no checkpoint files written.
- Default CLI generator knobs: `n_residues=24`, `n_couplings=8`, `noise=0.05`, `frames=128`, `seeds=3`, `seed=0`. Default scorer set: all of `dccm,mi,contact,null,influence,cri,relational`.

---

### Task 1: Synthetic planted-coupling generator

**Files:**
- Create: `src/allostery/validation/__init__.py`
- Create: `src/allostery/validation/synthetic.py`
- Test: `tests/test_validation_synthetic.py`

**Interfaces:**
- Produces: `PlantedSystem(pdb_path: Path, coupling_matrix: np.ndarray, n_residues: int, n_couplings: int)`; `generate_planted_system(out_path, *, n_residues=24, n_couplings=8, coupling_strength=1.0, noise=0.05, frames=128, seed=0, min_sequence_separation=2) -> PlantedSystem`. The PDB is a multi-model CA file readable by `allostery.io.trajectory.load_trajectory`; `coupling_matrix` is a symmetric boolean `[N, N]` with zero diagonal and exactly `n_couplings` planted upper-triangle edges.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_synthetic.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from allostery.io.trajectory import load_trajectory
from allostery.validation.synthetic import generate_planted_system


def test_generates_readable_pdb_and_truth_matrix(tmp_path: Path) -> None:
    pdb = tmp_path / "planted.pdb"
    system = generate_planted_system(
        pdb, n_residues=12, n_couplings=4, frames=40, seed=1,
    )
    assert system.pdb_path == pdb
    assert pdb.exists()

    trajectory = load_trajectory(pdb)
    assert trajectory.coordinates.shape == (40, 12, 3)

    matrix = system.coupling_matrix
    assert matrix.shape == (12, 12)
    assert matrix.dtype == bool
    assert np.array_equal(matrix, matrix.T)              # symmetric
    assert not matrix.diagonal().any()                   # zero diagonal
    assert int(np.triu(matrix).sum()) == 4               # exactly n_couplings edges


def test_planted_pairs_respect_min_separation(tmp_path: Path) -> None:
    system = generate_planted_system(
        tmp_path / "p.pdb", n_residues=16, n_couplings=6, frames=20, seed=2,
        min_sequence_separation=2,
    )
    rows, cols = np.where(np.triu(system.coupling_matrix))
    assert np.all((cols - rows) >= 2)


def test_is_deterministic(tmp_path: Path) -> None:
    a = generate_planted_system(tmp_path / "a.pdb", n_residues=10, n_couplings=3, frames=16, seed=7)
    b = generate_planted_system(tmp_path / "b.pdb", n_residues=10, n_couplings=3, frames=16, seed=7)
    assert np.array_equal(a.coupling_matrix, b.coupling_matrix)
    assert (tmp_path / "a.pdb").read_text() == (tmp_path / "b.pdb").read_text()


def test_rejects_too_many_couplings(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(ValueError, match="n_couplings"):
        generate_planted_system(tmp_path / "x.pdb", n_residues=5, n_couplings=999, frames=10, seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_synthetic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.validation'`

- [ ] **Step 3: Write minimal implementation**

Create the package marker:

```python
# src/allostery/validation/__init__.py
from __future__ import annotations
```

Create the generator. The dynamics is overdamped Langevin on per-residue displacements `u_i` from fixed backbone positions `b_i`: coupled pairs share a `-k_c (u_i - u_j)` force so their *motion* correlates while their equilibrium positions stay spread along the chain (so a distance-only baseline cannot detect them trivially). The ATOM line format is copied verbatim from the proven `benchmark/cri.py` writer.

```python
# src/allostery/validation/synthetic.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RESIDUE_NAMES = ['GLY', 'ALA', 'SER', 'THR', 'LEU', 'VAL', 'ASP', 'ASN', 'GLU', 'GLN']
_BACKBONE_SPACING = 3.8     # Angstrom CA-CA spacing along a straight backbone
_SELF_STIFFNESS = 1.0       # k0: tether of each residue to its backbone position
_INTEGRATION_DT = 0.1       # stable step for overdamped Langevin


@dataclass(frozen=True, slots=True)
class PlantedSystem:
    pdb_path: Path
    coupling_matrix: np.ndarray
    n_residues: int
    n_couplings: int


def _write_pdb(path: Path, coords: np.ndarray, n_residues: int) -> None:
    lines: list[str] = []
    serial = 1
    for frame_index in range(coords.shape[0]):
        lines.append(f'MODEL{frame_index + 1:>9}')
        for residue_index in range(n_residues):
            name = _RESIDUE_NAMES[residue_index % len(_RESIDUE_NAMES)]
            x, y, z = coords[frame_index, residue_index]
            lines.append(
                f'ATOM  {serial:5d}  CA  {name:>3s} A{residue_index + 1:4d}'
                f'{x:11.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C'
            )
            serial += 1
        lines.append('ENDMDL')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def generate_planted_system(
    out_path: str | Path,
    *,
    n_residues: int = 24,
    n_couplings: int = 8,
    coupling_strength: float = 1.0,
    noise: float = 0.05,
    frames: int = 128,
    seed: int = 0,
    min_sequence_separation: int = 2,
) -> PlantedSystem:
    if n_residues < 4:
        raise ValueError(f'n_residues must be >= 4 (got {n_residues})')
    if frames < 2:
        raise ValueError(f'frames must be >= 2 (got {frames})')
    rng = np.random.default_rng(seed)

    candidates = [
        (i, j)
        for i in range(n_residues)
        for j in range(i + min_sequence_separation, n_residues)
    ]
    if n_couplings > len(candidates):
        raise ValueError(
            f'n_couplings={n_couplings} exceeds available non-local pairs ({len(candidates)}) '
            f'for n_residues={n_residues}, min_sequence_separation={min_sequence_separation}'
        )
    chosen = rng.choice(len(candidates), size=n_couplings, replace=False)
    couplings = [candidates[int(k)] for k in chosen]

    coupling_matrix = np.zeros((n_residues, n_residues), dtype=bool)
    for i, j in couplings:
        coupling_matrix[i, j] = True
        coupling_matrix[j, i] = True

    base = np.zeros((n_residues, 3), dtype=np.float64)
    base[:, 0] = np.arange(n_residues) * _BACKBONE_SPACING
    base[:, 1] = 10.0
    base[:, 2] = 10.0

    displacement = np.zeros((n_residues, 3), dtype=np.float64)
    coords = np.empty((frames, n_residues, 3), dtype=np.float64)
    k_c = float(coupling_strength)
    for frame_index in range(frames):
        force = -_SELF_STIFFNESS * displacement
        for i, j in couplings:
            diff = displacement[i] - displacement[j]
            force[i] -= k_c * diff
            force[j] += k_c * diff
        displacement = (
            displacement
            + _INTEGRATION_DT * force
            + noise * rng.standard_normal((n_residues, 3))
        )
        coords[frame_index] = base + displacement

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdb(out_path, coords, n_residues)
    return PlantedSystem(
        pdb_path=out_path,
        coupling_matrix=coupling_matrix,
        n_residues=n_residues,
        n_couplings=n_couplings,
    )


__all__ = ['PlantedSystem', 'generate_planted_system']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_synthetic.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/allostery/validation/__init__.py src/allostery/validation/synthetic.py tests/test_validation_synthetic.py
git commit -m "feat: add synthetic planted-coupling generator for validation"
```

---

### Task 2: Classical baseline scorers

**Files:**
- Create: `src/allostery/validation/baselines.py`
- Test: `tests/test_validation_baselines.py`

**Interfaces:**
- Consumes: `allostery.io.pdb.Trajectory` (has `.residues: tuple[ResidueRecord, ...]` and `.coordinates: np.ndarray` of shape `[frames, N, 3]`); `allostery.pipeline.score.ResidueIdentifier`.
- Produces: `dccm_scores(trajectory, *, min_sequence_separation=2)`, `mutual_information_scores(trajectory, *, bins=8, min_sequence_separation=2)`, `contact_frequency_scores(trajectory, *, cutoff=8.0, min_sequence_separation=2)`, `shuffled_null_scores(trajectory, *, seed=0, min_sequence_separation=2)` — each returns `list[PairScore]` (dicts with `residue_i`/`residue_j`/`score`) sorted by score descending.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_baselines.py
from __future__ import annotations

import numpy as np

from allostery.io.pdb import ResidueRecord, Trajectory
from allostery.validation.baselines import (
    contact_frequency_scores,
    dccm_scores,
    mutual_information_scores,
    shuffled_null_scores,
)


def _trajectory(coords: np.ndarray) -> Trajectory:
    n = coords.shape[1]
    residues = tuple(
        ResidueRecord(index=i, chain_id="A", residue_number=i + 1, name="GLY")
        for i in range(n)
    )
    return Trajectory(residues=residues, coordinates=coords.astype(np.float32))


def _top_pair(scores):
    top = scores[0]
    return tuple(sorted((top["residue_i"]["index"], top["residue_j"]["index"])))


def test_dccm_ranks_correlated_pair_first() -> None:
    rng = np.random.default_rng(0)
    frames, n = 200, 6
    coords = rng.standard_normal((frames, n, 3)).astype(np.float64) * 0.1
    base = np.zeros((n, 3))
    base[:, 0] = np.arange(n) * 3.8
    # residues 0 and 5 share an identical displacement signal -> strongly correlated
    shared = rng.standard_normal((frames, 3))
    coords[:, 0, :] = shared
    coords[:, 5, :] = shared
    coords = coords + base[None, :, :]
    scores = dccm_scores(_trajectory(coords), min_sequence_separation=2)
    assert _top_pair(scores) == (0, 5)


def test_each_baseline_returns_all_nonlocal_pairs() -> None:
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((30, 5, 3))
    traj = _trajectory(coords)
    expected = sum(1 for i in range(5) for j in range(i + 2, 5))  # 6 pairs
    for fn in (dccm_scores, mutual_information_scores, contact_frequency_scores):
        scores = fn(traj, min_sequence_separation=2)
        assert len(scores) == expected
        assert all("score" in s for s in scores)


def test_shuffled_null_breaks_correlation() -> None:
    rng = np.random.default_rng(2)
    frames, n = 200, 6
    base = np.zeros((n, 3))
    base[:, 0] = np.arange(n) * 3.8
    shared = rng.standard_normal((frames, 3))
    coords = (rng.standard_normal((frames, n, 3)) * 0.1)
    coords[:, 0, :] = shared
    coords[:, 5, :] = shared
    coords = coords + base[None, :, :]
    traj = _trajectory(coords)

    true_top = dccm_scores(traj, min_sequence_separation=2)[0]["score"]
    null_top = shuffled_null_scores(traj, seed=0, min_sequence_separation=2)[0]["score"]
    # destroying temporal alignment collapses the strongest correlation
    assert null_top < true_top
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.validation.baselines'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/validation/baselines.py
from __future__ import annotations

import numpy as np

from allostery.io.pdb import ResidueRecord, Trajectory
from allostery.pipeline.score import ResidueIdentifier


def _residue_identifier(residue: ResidueRecord) -> ResidueIdentifier:
    return {
        "index": residue.index,
        "chain_id": residue.chain_id,
        "residue_number": residue.residue_number,
        "name": residue.name,
    }


def _emit_pairs(trajectory: Trajectory, matrix: np.ndarray, sep: int) -> list[dict]:
    residues = trajectory.residues
    n = matrix.shape[0]
    out: list[dict] = []
    for i in range(n):
        for j in range(i + sep, n):
            out.append({
                "residue_i": _residue_identifier(residues[i]),
                "residue_j": _residue_identifier(residues[j]),
                "score": float(matrix[i, j]),
            })
    out.sort(key=lambda item: item["score"], reverse=True)
    return out


def _displacements(trajectory: Trajectory) -> np.ndarray:
    coords = np.asarray(trajectory.coordinates, dtype=np.float64)  # [F, N, 3]
    return coords - coords.mean(axis=0, keepdims=True)


def dccm_scores(trajectory: Trajectory, *, min_sequence_separation: int = 2) -> list[dict]:
    disp = _displacements(trajectory)
    frames = disp.shape[0]
    covariance = np.einsum("tix,tjx->ij", disp, disp) / float(frames)
    diag = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    dccm = covariance / np.outer(diag, diag)
    return _emit_pairs(trajectory, np.abs(dccm), min_sequence_separation)


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    joint = np.zeros((bins, bins), dtype=np.float64)
    for x, y in zip(a.tolist(), b.tolist()):
        joint[x, y] += 1.0
    total = joint.sum()
    if total == 0:
        return 0.0
    joint /= total
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)
    mi = 0.0
    for x in range(bins):
        for y in range(bins):
            if joint[x, y] > 0.0 and p_a[x] > 0.0 and p_b[y] > 0.0:
                mi += joint[x, y] * np.log(joint[x, y] / (p_a[x] * p_b[y]))
    return float(mi)


def mutual_information_scores(
    trajectory: Trajectory, *, bins: int = 8, min_sequence_separation: int = 2
) -> list[dict]:
    disp = _displacements(trajectory)
    magnitude = np.linalg.norm(disp, axis=2)  # [F, N]
    frames, n = magnitude.shape
    digitized = np.empty((frames, n), dtype=int)
    for i in range(n):
        edges = np.histogram_bin_edges(magnitude[:, i], bins=bins)
        digitized[:, i] = np.clip(np.digitize(magnitude[:, i], edges[1:-1]), 0, bins - 1)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + min_sequence_separation, n):
            matrix[i, j] = _mutual_information(digitized[:, i], digitized[:, j], bins)
    return _emit_pairs(trajectory, matrix, min_sequence_separation)


def contact_frequency_scores(
    trajectory: Trajectory, *, cutoff: float = 8.0, min_sequence_separation: int = 2
) -> list[dict]:
    coords = np.asarray(trajectory.coordinates, dtype=np.float64)
    frames, n, _ = coords.shape
    frequency = np.zeros((n, n), dtype=np.float64)
    for frame_index in range(frames):
        frame = coords[frame_index]
        distances = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=2)
        frequency += (distances < cutoff).astype(np.float64)
    frequency /= float(frames)
    return _emit_pairs(trajectory, frequency, min_sequence_separation)


def shuffled_null_scores(
    trajectory: Trajectory, *, seed: int = 0, min_sequence_separation: int = 2
) -> list[dict]:
    coords = np.asarray(trajectory.coordinates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    shuffled = coords.copy()
    frames, n, _ = coords.shape
    for i in range(n):
        shuffled[:, i, :] = coords[rng.permutation(frames), i, :]
    fake = Trajectory(residues=trajectory.residues, coordinates=shuffled.astype(np.float32))
    return dccm_scores(fake, min_sequence_separation=min_sequence_separation)


__all__ = [
    "contact_frequency_scores",
    "dccm_scores",
    "mutual_information_scores",
    "shuffled_null_scores",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_baselines.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/allostery/validation/baselines.py tests/test_validation_baselines.py
git commit -m "feat: add classical baseline scorers (DCCM, MI, contact, null)"
```

---

### Task 3: Ranking metrics

**Files:**
- Create: `src/allostery/validation/metrics.py`
- Test: `tests/test_validation_metrics.py`

**Interfaces:**
- Consumes: a `list[PairScore]` (dicts keyed `residue_i`/`residue_j`/`score`, identifiers carry `index`); a boolean `coupling_matrix: np.ndarray` `[N, N]`.
- Produces: `ScoreMetrics(roc_auc: float, pr_auc: float, precision_at_k: float, recall_at_k: float, n_true: int, n_pairs: int)`; `evaluate_scores(pair_scores, coupling_matrix, *, min_sequence_separation=2) -> ScoreMetrics`. The candidate universe is every unordered pair `(i, j)` with `j - i >= min_sequence_separation`; pairs the scorer omits are ranked last (score `-inf`). Degenerate cases (no positives or no negatives) return `roc_auc=0.5` and zeros without raising.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_metrics.py
from __future__ import annotations

import numpy as np

from allostery.validation.metrics import evaluate_scores


def _pair(i: int, j: int, score: float) -> dict:
    return {
        "residue_i": {"index": i, "chain_id": "A", "residue_number": i + 1, "name": "GLY"},
        "residue_j": {"index": j, "chain_id": "A", "residue_number": j + 1, "name": "GLY"},
        "score": score,
    }


def _truth(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        matrix[i, j] = True
        matrix[j, i] = True
    return matrix


def test_perfect_ranking_scores_one() -> None:
    # n=5, sep>=2 candidate pairs: (0,2)(0,3)(0,4)(1,3)(1,4)(2,4); truths (0,2)(1,3)
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [
        _pair(0, 2, 9.0), _pair(1, 3, 8.0),    # true pairs ranked highest
        _pair(0, 3, 1.0), _pair(0, 4, 0.5), _pair(1, 4, 0.4), _pair(2, 4, 0.3),
    ]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.n_true == 2
    assert m.n_pairs == 6
    assert m.roc_auc == 1.0
    assert m.pr_auc == 1.0
    assert m.precision_at_k == 1.0
    assert m.recall_at_k == 1.0


def test_reversed_ranking_scores_zero_auc() -> None:
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [
        _pair(0, 2, 0.0), _pair(1, 3, 0.1),    # true pairs ranked lowest
        _pair(0, 3, 9.0), _pair(0, 4, 8.0), _pair(1, 4, 7.0), _pair(2, 4, 6.0),
    ]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.roc_auc == 0.0
    assert m.precision_at_k == 0.0


def test_all_tied_scores_give_half_auc() -> None:
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [
        _pair(0, 2, 1.0), _pair(1, 3, 1.0), _pair(0, 3, 1.0),
        _pair(0, 4, 1.0), _pair(1, 4, 1.0), _pair(2, 4, 1.0),
    ]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.roc_auc == 0.5


def test_missing_pairs_rank_last() -> None:
    # scorer omits the true pair (1,3); it must be treated as worst-ranked
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [_pair(0, 2, 9.0), _pair(0, 3, 8.0), _pair(0, 4, 7.0), _pair(2, 4, 6.0)]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.n_pairs == 6                      # full candidate universe, not just provided
    assert m.precision_at_k <= 0.5             # only (0,2) can be recovered in top-2


def test_no_positive_pairs_returns_sentinel() -> None:
    truth = _truth(5, [])                       # no true edges
    scores = [_pair(0, 2, 1.0), _pair(1, 3, 0.5)]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.n_true == 0
    assert m.roc_auc == 0.5
    assert m.pr_auc == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.validation.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/validation/metrics.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScoreMetrics:
    roc_auc: float
    pr_auc: float
    precision_at_k: float
    recall_at_k: float
    n_true: int
    n_pairs: int


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Ranks (1-based) with ties assigned their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # average ranks within tie groups
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                avg = ranks[order[start:end]].mean()
                ranks[order[start:end]] = avg
            start = end
    return ranks


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rank_average(scores)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(np.float64)
    cumulative_tp = np.cumsum(labels_sorted)
    precision = cumulative_tp / (np.arange(len(labels_sorted)) + 1.0)
    return float((precision * labels_sorted).sum() / n_pos)


def evaluate_scores(
    pair_scores: list[dict],
    coupling_matrix: np.ndarray,
    *,
    min_sequence_separation: int = 2,
) -> ScoreMetrics:
    n = coupling_matrix.shape[0]
    score_map: dict[tuple[int, int], float] = {}
    for item in pair_scores:
        i = int(item["residue_i"]["index"])
        j = int(item["residue_j"]["index"])
        score_map[(min(i, j), max(i, j))] = float(item["score"])

    scores: list[float] = []
    labels: list[bool] = []
    for i in range(n):
        for j in range(i + min_sequence_separation, n):
            scores.append(score_map.get((i, j), float("-inf")))
            labels.append(bool(coupling_matrix[i, j]))

    score_array = np.array(scores, dtype=np.float64)
    label_array = np.array(labels, dtype=bool)
    n_true = int(label_array.sum())
    n_pairs = int(len(label_array))

    roc = _roc_auc(score_array, label_array)
    ap = _average_precision(score_array, label_array)

    if n_true == 0:
        precision_at_k = 0.0
        recall_at_k = 0.0
    else:
        top = np.argsort(-score_array, kind="mergesort")[:n_true]
        hits = int(label_array[top].sum())
        precision_at_k = hits / float(n_true)
        recall_at_k = hits / float(n_true)

    return ScoreMetrics(
        roc_auc=roc,
        pr_auc=ap,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        n_true=n_true,
        n_pairs=n_pairs,
    )


__all__ = ["ScoreMetrics", "evaluate_scores"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_metrics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/allostery/validation/metrics.py tests/test_validation_metrics.py
git commit -m "feat: add ranking metrics for validation (ROC-AUC, PR-AUC, precision@k)"
```

---

### Task 4: Harness core with baseline scorers

**Files:**
- Create: `src/allostery/validation/harness.py`
- Test: `tests/test_validation_harness.py`

**Interfaces:**
- Consumes: `generate_planted_system`/`PlantedSystem` (Task 1); `dccm_scores`/`mutual_information_scores`/`contact_frequency_scores`/`shuffled_null_scores` (Task 2); `ScoreMetrics`/`evaluate_scores` (Task 3); `allostery.io.trajectory.load_trajectory`.
- Produces: `ValidationConfig`, `ScorerResult`, `ValidationReport`, `run_validation(config, *, scorers=None) -> ValidationReport`, `render_validation_table(report) -> str`, `validation_report_to_dict(report) -> dict`, and the name tuples `ALL_SCORERS`, `_BASELINE_SCORERS`, `_MODEL_SCORERS`. In this task only the four baseline names are runnable; the three model names are valid (accepted by name validation) but `_run_one_scorer` raises `ValueError` for them — Task 5 fills them in. `ScorerResult.beats_best_baseline` defaults `False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_harness.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.validation.harness'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/validation/harness.py
from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from allostery.io.trajectory import load_trajectory
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


def _run_one_scorer(name, trajectory, pdb_path, config, seed):
    sep = config.min_sequence_separation
    if name == "dccm":
        return dccm_scores(trajectory, min_sequence_separation=sep)
    if name == "mi":
        return mutual_information_scores(trajectory, min_sequence_separation=sep)
    if name == "contact":
        return contact_frequency_scores(trajectory, min_sequence_separation=sep)
    if name == "null":
        return shuffled_null_scores(trajectory, seed=seed, min_sequence_separation=sep)
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


def run_validation(config: ValidationConfig, *, scorers: list[str] | None = None) -> ValidationReport:
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
                    pair_scores, system.coupling_matrix,
                    min_sequence_separation=config.min_sequence_separation,
                )
                per_scorer[name].append(metrics)

    results = [_aggregate(name, per_scorer[name]) for name in selected]

    baseline_results = [r for r in results if r.name in _BASELINE_SCORERS]
    best_baseline = (
        max(baseline_results, key=lambda r: r.roc_auc_mean).name if baseline_results else ""
    )
    best_baseline_mean = max(
        (r.roc_auc_mean for r in baseline_results), default=float("-inf")
    )

    final = [
        replace(
            r,
            beats_best_baseline=(r.name in _MODEL_SCORERS and r.roc_auc_mean > best_baseline_mean),
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
    "ScorerResult",
    "ValidationConfig",
    "ValidationReport",
    "render_validation_table",
    "run_validation",
    "validation_report_to_dict",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_harness.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/allostery/validation/harness.py tests/test_validation_harness.py
git commit -m "feat: add validation harness core with baseline scorers"
```

---

### Task 5: Model-family scorer adapters

**Files:**
- Modify: `src/allostery/validation/harness.py` (extend `_run_one_scorer`; add adapter helpers)
- Test: `tests/test_validation_model_scorers.py`

**Interfaces:**
- Consumes: `train_influence_model`+`score_influence_trajectory`, `train_cri_model`+`score_cri_trajectory`, `train_model`+`score_trajectory` from the existing pipeline.
- Produces: `_run_one_scorer` now also handles `"influence"`, `"cri"`, `"relational"` by training a small model on the synthetic PDB (no checkpoint written) and scoring it. Model scorers set `beats_best_baseline` via the existing aggregation in `run_validation` (already implemented in Task 4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_model_scorers.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_model_scorers.py -v`
Expected: FAIL — `ValueError: scorer 'influence' is not runnable in this build`

- [ ] **Step 3: Write minimal implementation**

Add the imports at the top of `src/allostery/validation/harness.py` (alongside the existing imports):

```python
from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model
from allostery.pipeline.influence_score import score_influence_trajectory
from allostery.pipeline.influence_train import train_influence_model
from allostery.pipeline.score import score_trajectory
from allostery.pipeline.train import train_model
```

Add module constants (small, fast hyperparameters) after the scorer-name tuples:

```python
_MODEL_WINDOW = 3
_MODEL_STRIDE = 1
_MODEL_HIDDEN = 8
_MODEL_EPOCHS = 2
_MODEL_LR = 1e-2
```

Add the three adapter functions (above `_run_one_scorer`):

```python
def _score_influence(pdb_path, config, seed):
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


def _score_cri(pdb_path, config, seed):
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


def _score_relational(pdb_path, config, seed):
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
```

Replace the body of `_run_one_scorer` so the model names dispatch to the adapters (keep the baseline branches; replace the final `raise`):

```python
def _run_one_scorer(name, trajectory, pdb_path, config, seed):
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
```

- [ ] **Step 4: Run test to verify it passes, and confirm no regression**

Run: `pytest tests/test_validation_model_scorers.py tests/test_validation_harness.py -v`
Expected: PASS (new model-scorer tests plus the Task 4 harness tests).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/validation/harness.py tests/test_validation_model_scorers.py
git commit -m "feat: add model-family scorer adapters to validation harness"
```

---

### Task 6: `validate` CLI subcommand

**Files:**
- Modify: `src/allostery/cli.py`
- Test: `tests/test_cli_validate.py`

**Interfaces:**
- Consumes: `run_validation`, `render_validation_table`, `validation_report_to_dict`, `ValidationConfig` (Tasks 4–5); existing `Result` (already imported in cli.py).
- Produces: a `validate` subcommand (`allostery validate [--scorers ...] [--n-residues N] [--couplings M] [--noise σ] [--frames F] [--seeds R] [--seed S] [--out-json PATH]`) dispatched in `_dispatch`, returning a `Result(command="validate", summary=<table>, data=<report dict>, artifacts=[out_json?])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_validate.py
from __future__ import annotations

import json
from pathlib import Path

from allostery.cli import main


def test_validate_runs_and_prints_table(capsys) -> None:
    code = main([
        "validate", "--scorers", "dccm,null",
        "--n-residues", "12", "--couplings", "4", "--frames", "48", "--seeds", "1",
    ])
    captured = capsys.readouterr()
    assert code == 0
    assert "best scorer" in captured.out
    assert "dccm" in captured.out


def test_validate_json_mode_is_parseable(tmp_path: Path, capsys) -> None:
    out_json = tmp_path / "report.json"
    code = main([
        "--json", "validate", "--scorers", "dccm,contact,null",
        "--n-residues", "12", "--couplings", "4", "--frames", "48", "--seeds", "1",
        "--out-json", str(out_json),
    ])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["command"] == "validate"
    assert payload["data"]["best_scorer"] in {"dccm", "contact", "null"}
    assert str(out_json) in payload["artifacts"]
    assert out_json.exists()                      # report written to disk
    on_disk = json.loads(out_json.read_text())
    assert len(on_disk["scorers"]) == 3


def test_validate_unknown_scorer_exits_1(capsys) -> None:
    code = main(["validate", "--scorers", "bogus", "--seeds", "1"])
    captured = capsys.readouterr()
    assert code == 1
    assert "unknown scorer" in captured.err
    assert "Traceback" not in captured.err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_validate.py -v`
Expected: FAIL — argparse error: `invalid choice: 'validate'`.

- [ ] **Step 3: Write minimal implementation**

In `src/allostery/cli.py`, add `'validate'` to the subcommands set:

```python
_SUBCOMMANDS = frozenset({'run', 'analyze', 'check', 'interpret', 'workflow', 'validate'})
```

Ensure `import json` is present near the top of `cli.py` (add it if missing). Add the import alongside the other pipeline imports:

```python
from allostery.validation.harness import (
    ValidationConfig,
    render_validation_table,
    run_validation,
    validation_report_to_dict,
)
```

Add the subparser in `build_parser()` (before `return parser`):

```python
    validate_parser = subparsers.add_parser(
        'validate',
        help='Measure scorer accuracy against synthetic planted-coupling ground truth')
    validate_parser.add_argument(
        '--scorers', default=None,
        help='Comma-separated scorers (default: all). '
             'Choices: dccm,mi,contact,null,influence,cri,relational')
    validate_parser.add_argument('--n-residues', type=int, default=24)
    validate_parser.add_argument('--couplings', type=int, default=8)
    validate_parser.add_argument('--noise', type=float, default=0.05)
    validate_parser.add_argument('--frames', type=int, default=128)
    validate_parser.add_argument('--seeds', type=int, default=3)
    validate_parser.add_argument('--seed', type=int, default=0)
    validate_parser.add_argument('--out-json', default=None, help='Write the full JSON report here')
```

Add a branch in `_dispatch` (place it alongside the other command branches, e.g. after the `workflow` branch):

```python
    if args.command == 'validate':
        scorers = None
        if args.scorers:
            scorers = [name.strip() for name in args.scorers.split(',') if name.strip()]
        config = ValidationConfig(
            n_residues=args.n_residues,
            n_couplings=args.couplings,
            noise=args.noise,
            frames=args.frames,
            seeds=args.seeds,
            base_seed=args.seed,
        )
        report = run_validation(config, scorers=scorers)
        data = validation_report_to_dict(report)
        artifacts: list[Path] = []
        if args.out_json:
            out_json = Path(args.out_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(data, indent=2), encoding='utf-8')
            artifacts.append(out_json)
        return Result(
            command='validate',
            summary=render_validation_table(report),
            data=data,
            artifacts=artifacts,
        )
```

(The unknown-scorer `ValueError` from `run_validation` propagates to the `main()` wrapper, which renders it cleanly to stderr and exits 1.)

- [ ] **Step 4: Run test to verify it passes, and confirm no regression**

Run: `pytest tests/test_cli_validate.py tests/test_cli.py tests/test_cli_presentation.py -v`
Expected: PASS (new validate tests plus existing CLI tests — default output for other commands unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/cli.py tests/test_cli_validate.py
git commit -m "feat: add 'validate' CLI subcommand"
```

---

### Task 7: Documentation — README and help

**Files:**
- Modify: `README.md`
- Modify: `tests/test_cli_help.py`

**Interfaces:**
- Produces: README "Commands" section lists `allostery validate`; the help smoke test asserts `validate` appears in `--help`.

- [ ] **Step 1: Write the failing test**

Add these two tests to `tests/test_cli_help.py` (append; do not remove the existing tests):

```python
def test_help_lists_validate_command() -> None:
    import pytest
    from allostery.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    # capsys is unavailable here; re-render help text directly
    text = parser.format_help()
    assert "validate" in text


def test_readme_documents_validate() -> None:
    from pathlib import Path
    readme = Path(__file__).resolve().parent.parent / "README.md"
    body = readme.read_text(encoding="utf-8")
    assert "allostery validate" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_help.py::test_readme_documents_validate -v`
Expected: FAIL — README has no `allostery validate` (the help test passes already since Task 6 added the subparser).

- [ ] **Step 3: Write minimal implementation**

In `README.md`, in the "## Commands" code block, add the `validate` line after the `workflow` line:

```
allostery validate [options]             # measure scorer accuracy vs. synthetic ground truth
```

Then add a new section after the "### Workflow" section:

```markdown
### Validate

Measure how well each scorer recovers *known* residue–residue coupling. The harness
generates synthetic trajectories from a planted coupling graph (exact ground truth),
runs classical baselines (DCCM, mutual information, contact frequency) alongside a
shuffled-trajectory null and the three model families, and reports ranking metrics
(ROC-AUC, PR-AUC, precision@k) averaged over seeds:

```bash
# Baselines only, fast
allostery validate --scorers dccm,mi,contact,null --seeds 5

# Full comparison including the trained model families
allostery validate --n-residues 24 --couplings 8 --seeds 3

# Machine-readable report for scripting / CI
allostery --json validate --out-json outputs/validation.json
```

The report flags whether each model family beats the best classical baseline — a method
is only meaningful if it outperforms trivial correlation on ground truth.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_help.py -v`
Expected: PASS (existing help tests plus the two new ones).

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest -q`
Expected: PASS (entire suite including every new validation test).

```bash
git add README.md tests/test_cli_help.py
git commit -m "docs: document the validate command"
```

---

## Self-Review

**Spec coverage:**
- §4.1 synthetic generator (planted coupling graph, multi-model PDB, tunable strength/noise, `PlantedSystem`) → Task 1. ✓
- §4.2 baselines (DCCM, MI, contact frequency, shuffled null), shared pair-score shape, seq-sep masking → Task 2. ✓
- §4.3 metrics (`ScoreMetrics`, ROC-AUC, PR-AUC, precision@k/recall@k, seq-sep masking, missing-pair handling, degenerate sentinels) → Task 3. ✓
- §4.4 harness (`ValidationConfig`, `ScorerResult`, `ValidationReport`, `run_validation`, mean ± std, `beats_best_baseline`, ranking, model-family adapters reusing existing train/score) → Tasks 4 (core + baselines) and 5 (model adapters). ✓
- §4.5 CLI (`validate` subcommand, flag-driven, `Result`-based, `--json`/`--quiet`, `--out-json` artifact written by the CLI branch, unknown-scorer → exit 1) → Task 6. ✓
- §5 data flow (per-seed generate → score → evaluate → aggregate; global flags never reach the harness) → Tasks 4, 6. ✓
- §6 error handling (unknown scorer → ValueError → exit 1; degenerate metrics → sentinel; too-small/over-coupled system → ValueError) → Tasks 1, 3, 4, 6. ✓
- §7 testing (generator readable + correct truth + deterministic; baselines shape + DCCM detects correlation + null near chance; metrics hand-computed cases; harness DCCM-beats-null + ranking + model smoke; CLI exit codes + json) → Tasks 1–6. ✓
- §8 no new dependencies → honored (numpy + existing torch pipeline + stdlib). ✓

**Refinement vs. spec:** the spec listed `time_step` in the generator signature; the plan drops it from `generate_planted_system` (the generator uses a fixed stable internal integration step `_INTEGRATION_DT`) and keeps `time_step=1.0` only where the model scorers need it (Task 5 adapters). This is a faithfulness refinement — synthetic-trajectory frame spacing does not affect ranking — not a scope change.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test has real assertions. The model-scorer hyperparameters (`_MODEL_*`) are concrete constants. ✓

**Type consistency:**
- `PlantedSystem(pdb_path, coupling_matrix, n_residues, n_couplings)` (Task 1) constructed and consumed identically in Task 4. ✓
- Pair-score dict shape `{residue_i, residue_j, score}` with `index` is produced by baselines (Task 2) and model scorers (Task 5) and consumed by `evaluate_scores` (Task 3). ✓
- `ScoreMetrics(roc_auc, pr_auc, precision_at_k, recall_at_k, n_true, n_pairs)` (Task 3) aggregated in Task 4 and serialized in `validation_report_to_dict`. ✓
- `ValidationConfig` field names (`n_residues`, `n_couplings`, `coupling_strength`, `noise`, `frames`, `seeds`, `base_seed`, `min_sequence_separation`) (Task 4) match the CLI construction in Task 6 (`base_seed=args.seed`). ✓
- `run_validation(config, *, scorers=None) -> ValidationReport` (Task 4) called identically in Task 6. ✓
- `render_validation_table` / `validation_report_to_dict` (Task 4) imported and called in Task 6. ✓
- Model train/score signatures in Task 5 match the verified pipeline signatures (`train_influence_model`/`score_influence_trajectory`, `train_cri_model`/`score_cri_trajectory`, `train_model`/`score_trajectory`). ✓
