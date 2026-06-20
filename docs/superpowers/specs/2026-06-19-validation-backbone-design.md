# Validation Backbone — Design

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan
**Scope:** A scientific validation harness that measures whether the allostery scorers recover *known* residue–residue coupling. Synthetic planted-coupling systems provide exact ground truth; classical baselines provide a comparison floor; standard ranking metrics quantify accuracy. Surfaced as a new `allostery validate` subcommand. This round **measures** accuracy — it does not change the models.

## 1. Motivation

The pipeline trains three model families (`influence`, `cri`, `relational`) that emit residue-pair allosteric scores, but nothing establishes whether those scores are *correct*:

- **The existing benchmark measures only speed.** `src/allostery/benchmark/cri.py` reports train/score seconds and counts. Its synthetic PDB is a smooth global oscillator with **no known coupling graph**, so it cannot measure accuracy.
- **No baselines.** Nothing compares the deep models against classical couplings (dynamical cross-correlation, mutual information, contact frequency). Without a baseline we cannot claim the models add anything over trivial correlation.
- **No ground truth.** There is no system where the true coupled pairs are known, so precision/recall/ROC are uncomputable.

A method whose accuracy has never been measured against ground truth — and never compared to a trivial baseline — is not yet scientifically defensible. This subsystem supplies the measuring stick. Uncertainty quantification, null-model significance/FDR, and score calibration are **separate future cycles that build on this harness** and are out of scope here.

## 2. Goals / Non-Goals

**Goals**
- A synthetic generator that produces a multi-model CA PDB from a *known* coupling graph, with tunable coupling strength and thermal noise.
- Classical baseline scorers (DCCM, mutual information, contact frequency) sharing the existing pair-score output shape.
- Ranking metrics (ROC-AUC, PR-AUC, precision@k, recall@k) that compare any scorer's output to the planted ground truth, with sequence-separation masking so only *non-trivial* coupling is scored.
- A harness that runs the three model families plus the baselines plus a shuffled-trajectory null over multiple seeds and reports mean ± std per metric, flagging whether each deep model beats the best classical baseline.
- An `allostery validate` CLI subcommand that emits the metric report (reusing the existing `Result`/`format_result`, so `--json`/`--quiet` work).

**Non-Goals**
- No real-protein data or experimentally-annotated allosteric sites (data-heavy, fuzzy ground truth).
- No new dependencies — numpy and torch are already present.
- No changes to the models, objectives, or scoring math.
- No uncertainty intervals, null-model significance testing, FDR, or score calibration — deferred to later cycles.
- No directed-pathway generator in this round (the generator is undirected; directed signal-propagation is a noted future variant).

## 3. Architecture

A new `src/allostery/validation/` package with four focused modules plus a thin CLI branch. The unifying abstraction: **every scorer reduces to `trajectory → ranked residue-pair scores`** in the existing `{residue_i, residue_j, score}` shape (residue identifiers carry an `index`). The metrics layer is therefore scorer-agnostic — it keys on residue `index` pairs and compares to the ground-truth matrix.

```
   allostery validate <flags>
            │
            ▼
   harness.run_validation(config) ──► ValidationReport
            │
   ┌────────┼─────────────────────────────────┐
   ▼        ▼                                   ▼
synthetic   scorers (per seed):            metrics
generate    • 3 model families             ROC-AUC / PR-AUC
(pdb +      • DCCM / MI / contact          precision@k / recall@k
 truth)     • shuffled-null floor          (vs. ground-truth matrix,
                                            seq-sep masked)
            │
            ▼
          Result ──► format_result (default | --json | --quiet)
```

**New / changed files**
- `src/allostery/validation/__init__.py` *(new)* — package marker.
- `src/allostery/validation/synthetic.py` *(new)* — planted-coupling generator.
- `src/allostery/validation/baselines.py` *(new)* — classical scorers + shuffled-null.
- `src/allostery/validation/metrics.py` *(new)* — ranking metrics vs. ground truth.
- `src/allostery/validation/harness.py` *(new)* — orchestrator + `ValidationReport`.
- `src/allostery/cli.py` *(modified)* — `validate` subcommand + dispatch branch.
- `README.md` *(modified)* — document `validate`.

## 4. Components

### 4.1 Synthetic generator (`synthetic.py`)

A damped coupled-oscillator / elastic-network model (ENM) — a recognized biophysics construct (ANM/GNM) — produces genuine temporal dynamics (positions → velocities → accelerations), the signal the acceleration-prediction models assume.

```python
@dataclass(frozen=True, slots=True)
class PlantedSystem:
    pdb_path: Path                 # multi-model CA PDB, readable by load_trajectory
    coupling_matrix: np.ndarray    # [N, N] bool — True where pair is truly coupled
    n_residues: int
    n_couplings: int

def generate_planted_system(
    out_path: str | Path,
    *,
    n_residues: int = 24,
    n_couplings: int = 8,        # number of non-backbone planted springs
    coupling_strength: float = 1.0,
    noise: float = 0.05,         # thermal displacement scale
    frames: int = 128,
    time_step: float = 1.0,
    seed: int = 0,
) -> PlantedSystem
```

Behavior:
- Lay `n_residues` CA atoms on a smooth 3D backbone curve (chain `A`, residue numbers `1..N`, rotating residue names from a fixed list — matching the existing synthetic PDB writer's conventions).
- Backbone springs connect sequence neighbors (i, i+1). **Planted** springs connect `n_couplings` randomly chosen non-adjacent pairs (separation ≥ 2) at `coupling_strength`. The planted pairs (and only those) define `coupling_matrix`; backbone neighbors are excluded from ground truth because they are trivially coupled.
- Integrate damped dynamics with per-frame Gaussian thermal kicks of scale `noise` (semi-implicit Euler, fixed `time_step`), starting from small random displacements. Deterministic given `seed`.
- Write CA coordinates per frame as `MODEL`/`ENDMDL` records in the exact ATOM format `load_multimodel_pdb` parses (Angstrom coordinates).
- Return `PlantedSystem` with the boolean `coupling_matrix` (symmetric, zero diagonal).

The generator writes a real PDB file (not an in-memory trajectory) so the full train→score path runs unmodified.

### 4.2 Baselines (`baselines.py`)

Three numpy-only classical scorers plus a null, each consuming a `Trajectory` (from `load_trajectory`) and returning `list[PairScore]` in the existing shape (`residue_i`/`residue_j` as `ResidueIdentifier`, `score: float`), sorted descending:

- **DCCM** — dynamical cross-correlation: for displacement vectors `Δr_i(t) = r_i(t) − mean_t r_i`, score `|⟨Δr_i · Δr_j⟩| / sqrt(⟨|Δr_i|²⟩⟨|Δr_j|²⟩)`. The standard MD coupling metric.
- **MI** — mutual information of per-residue fluctuation magnitude `|Δr_i(t)|`, histogram-binned (fixed bin count), computed from the discrete joint/marginal distributions.
- **Contact frequency** — fraction of frames in which `‖r_i(t) − r_j(t)‖ < cutoff`. A structure-only baseline expected to be *weak* on long-range coupling — useful as a low bar.
- **Shuffled null** — DCCM computed after independently permuting each residue's frame order (destroying cross-residue temporal correlation). Establishes the noise floor any real method must clear.

All four mask pairs within a sequence-separation threshold (default 2) so trivial neighbors are excluded, matching the generator's ground-truth definition.

```python
def dccm_scores(trajectory, *, min_sequence_separation=2) -> list[PairScore]
def mutual_information_scores(trajectory, *, bins=8, min_sequence_separation=2) -> list[PairScore]
def contact_frequency_scores(trajectory, *, cutoff=8.0, min_sequence_separation=2) -> list[PairScore]
def shuffled_null_scores(trajectory, *, seed=0, min_sequence_separation=2) -> list[PairScore]
```

### 4.3 Metrics (`metrics.py`)

Compares any scorer's ranked pairs to the boolean ground-truth matrix. Scorer-agnostic: it reads each item's `residue_i['index']`, `residue_j['index']`, `score`.

```python
@dataclass(frozen=True, slots=True)
class ScoreMetrics:
    roc_auc: float
    pr_auc: float
    precision_at_k: float    # k = number of true edges
    recall_at_k: float
    n_true: int
    n_pairs: int

def evaluate_scores(
    pair_scores: list[PairScore],
    coupling_matrix: np.ndarray,
    *,
    min_sequence_separation: int = 2,
) -> ScoreMetrics
```

- Build aligned arrays of `(score, label)` over all unordered index pairs with separation ≥ threshold, where `label = coupling_matrix[i, j]`.
- **ROC-AUC** via the rank-sum (Mann–Whitney U) identity — exact, ties averaged, no sklearn.
- **PR-AUC** as average precision over the score-sorted ranking — the honest metric for sparse positives.
- **precision@k / recall@k** with `k = n_true` (number of planted edges).
- Hand-implemented in numpy; degenerate cases (no positives, no negatives, all-tied scores) return well-defined values (e.g. ROC-AUC 0.5 when undefined) rather than raising.

### 4.4 Harness (`harness.py`)

Pure orchestration; returns data, prints nothing.

```python
@dataclass(frozen=True, slots=True)
class ScorerResult:
    name: str                         # "influence" | "cri" | "relational" | "dccm" | ...
    metrics_per_seed: list[ScoreMetrics]
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    beats_best_baseline: bool         # set for model scorers; False for baselines

@dataclass(frozen=True, slots=True)
class ValidationReport:
    scorers: list[ScorerResult]       # ranked by roc_auc_mean, desc
    best_scorer: str
    best_baseline: str
    config: dict[str, Any]            # echoed knobs for reproducibility

def run_validation(config: ValidationConfig, *, scorers: list[str] | None = None) -> ValidationReport
```

Per seed: generate a fresh planted system, then run each selected scorer:
- **Baselines / null** consume the loaded `Trajectory` directly.
- **Model families** train a small model on the synthetic PDB and score it, reusing the existing family functions directly — `train_influence_model`→`score_influence_trajectory`, `train_cri_model`→`score_cri_trajectory`, `train_model`→`score_trajectory` — with fixed small hyperparameters (low hidden dim, few epochs) chosen for speed. No checkpoint files are written (in-memory model handoff).

Aggregate each scorer's per-seed metrics into mean ± std. Mark a model scorer `beats_best_baseline` when its `roc_auc_mean` exceeds the best classical baseline's. Rank scorers by `roc_auc_mean`.

`ValidationConfig` is a frozen dataclass of the generator knobs plus `seeds: int` and `scorers` selection. Default scorer set: all three baselines + the null + the three model families.

### 4.5 CLI (`validate` subcommand)

```
allostery validate [--scorers dccm,mi,influence,...] [--n-residues N] [--couplings M]
                    [--noise σ] [--frames F] [--seeds R] [--seed S] [--out-json PATH]
```

- Add `'validate'` to `_SUBCOMMANDS`; add a subparser; add a `validate` branch in `_dispatch` that builds a `ValidationConfig`, calls `run_validation`, and returns a `Result`.
- `Result.summary`: a ranked text table (scorer, ROC-AUC mean±std, PR-AUC mean±std, precision@k, "beats baseline?").
- `Result.data`: the full report (per-scorer, per-seed) for `--json`.
- `Result.artifacts`: `[out_json]` when `--out-json` is given. The harness stays pure (returns data only); the `validate` CLI branch serializes `ValidationReport` to the JSON path.
- Errors (e.g. unknown scorer name) raise `ValueError` → exit 1 via the existing wrapper.

## 5. Data Flow

```
validate <flags>
  → _dispatch builds ValidationConfig
  → run_validation:
        for seed in range(seeds):
            PlantedSystem = generate_planted_system(...)      # pdb + coupling_matrix
            trajectory = load_trajectory(pdb)
            for scorer in selected:
                pair_scores = scorer(trajectory | trained-model)
                m = evaluate_scores(pair_scores, coupling_matrix)
        aggregate → ValidationReport
  → Result(command="validate", summary=table, data=report, artifacts=[out_json?])
  → format_result → stdout (default | --json | --quiet)
```

The global flags never reach the harness; the harness returns pure data and the CLI renders it.

## 6. Error Handling

- **Unknown scorer name** in `--scorers` → `ValueError` listing valid names → exit 1.
- **Degenerate metric inputs** (no positive or no negative pairs after masking) → metrics return defined sentinels (ROC-AUC 0.5), never raise; the harness still completes.
- **Too-small system** (`n_residues` too low for `min_sequence_separation` to leave valid pairs, or `n_couplings` exceeding available non-adjacent pairs) → `ValueError` from the generator with guidance → exit 1.
- **Non-finite training loss** on a synthetic system → surfaces the existing trainer `ValueError` → exit 1; the message names the failing scorer.

## 7. Testing Strategy

- **synthetic**: generated PDB is readable by `load_trajectory` and yields `[frames, n_residues, 3]`; `coupling_matrix` is symmetric, zero-diagonal, has exactly `n_couplings` planted edges, and excludes backbone neighbors; identical `seed` ⇒ identical output.
- **baselines**: each scorer returns the correct pair count and shape; a hand-built two-cluster trajectory where two residues move together yields a high DCCM score for that pair; the shuffled null scores near chance.
- **metrics**: against a tiny hand-computed case — a perfect ranking gives ROC-AUC 1.0 and precision@k 1.0; a reversed ranking gives ROC-AUC 0.0; an all-tied scorer gives ROC-AUC 0.5; degenerate (no positives) returns the sentinel without raising.
- **harness**: end-to-end on a small planted system with `seeds=2` running DCCM + the null asserts DCCM's `roc_auc_mean` clears the null's (the core rigor claim), and that the report ranks scorers and identifies `best_baseline`. Model families covered by one fast smoke run (tiny hidden dim, 1–2 epochs) asserting the path completes and produces finite metrics — not a quality bar.
- **CLI**: `allostery validate --scorers dccm --seeds 1` exits 0 and prints the table; `--json` emits a parseable report with per-scorer entries; an unknown scorer name exits 1.
- Determinism via fixed seeds; no network, no real MD, no checkpoint files.

## 8. Dependencies

None new. Generator, baselines, and metrics use numpy; model scorers use the existing torch pipeline. `--json` uses stdlib `json`.

## 9. Open Questions (resolved)

- Goal: validation backbone first (measuring stick before accuracy changes). ✓
- Ground truth: synthetic planted-coupling systems (exact, tunable, tiny, reproducible). ✓
- Baselines: included (DCCM, MI, contact frequency) + shuffled null — comparison is the heart of rigor. ✓
- Surface: first-class `allostery validate` subcommand, `Result`-based, `--json`/`--quiet` aware. ✓
- Generator: Approach A, damped coupled-oscillator / elastic network (real temporal dynamics). Directed signal-propagation variant deferred. ✓
