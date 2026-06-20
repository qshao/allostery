# GROMACS Chain ID and Threshold Fix

## Goal

Fix two issues exposed by the first real KRAS WT trajectory run:
1. Residue labels show `None` as chain ID when loading GROMACS `.trr`/`.gro` files via MDTraj.
2. `detect_threshold` returns rank 1 when the score CSV contains all N×(N-1)/2 pairs (most near-zero), compressing the meaningful signal into the first 0.2% of the Kneedle x-axis.

Both fixes are surgical: one line each, no new files, no new interfaces.

---

## Fix 1: Chain ID Fallback in MDTraj Loader

**File:** `src/allostery/io/trajectory.py`

**Root cause:** GROMACS topology files (`.gro`, `.tpr`) do not store PDB chain labels. MDTraj exposes this as `atom.residue.chain.chain_id = None`. The current code does `str(None)` → `"None"`.

**Fix:** In `_load_via_mdtraj`, replace the `chain_id=` expression with a conditional:

```python
chain_id=(
    str(atom.residue.chain.chain_id)
    if atom.residue.chain.chain_id is not None
    else chr(ord('A') + min(atom.residue.chain.index, 25))
),
```

- Chain index 0 → `"A"`, index 1 → `"B"`, …, index ≥ 25 → `"Z"`.
- Single-chain GROMACS proteins (most cases) get `"A"`, matching convention and the reference PDB.
- Multi-chain GROMACS systems with distinct MDTraj chain objects get alphabetic IDs.
- The MDAnalysis path uses `str(atom.segid).strip() or "_"`. Update it to `str(atom.segid).strip() or "A"` — MDAnalysis exposes `segid` per-atom rather than a chain index, so `"A"` is the correct single-character fallback for GROMACS single-chain systems.

**Error handling:** No new errors. The `min(..., 25)` clamp silently assigns `"Z"` to all chains beyond 26 — acceptable because GROMACS systems with more than 26 protein chains are rare and still produce valid (though non-unique) labels.

---

## Fix 2: Kneedle on Top-k Scores Only

**File:** `src/allostery/pipeline/analyze.py`

**Root cause:** `detect_threshold(all_scores)` receives all N×(N-1)/2 pair scores. For a 169-residue protein, that is 14,196 scores, of which ~14,166 are near zero. After Kneedle normalization, the meaningful top-30 scores occupy x ∈ [0, 0.002], making every interior point give `y + x - 1 < 0`. `np.argmax` returns index 0 (first occurrence of the maximum value 0.0) → rank 1.

**Fix:** Pass only the top-k scores to `detect_threshold`. The `rows` list from `read_scores_csv` is already sorted descending by score, so `all_scores[:top_k]` is the top-k slice.

```python
# Before
threshold_score, threshold_rank = detect_threshold(all_scores)

# After
threshold_score, threshold_rank = detect_threshold(all_scores[:top_k])
```

- The threshold line now reads: `"Suggested threshold: 0.0312 (top 4 of 30 scored pairs — largest gap at rank 4)"` — directly actionable.
- The full histogram (`format_score_histogram(all_scores, ...)`) is unaffected: users still see the complete score distribution including the near-zero tail.
- The `▶ threshold` marker in the histogram continues to work: `format_score_histogram` finds the correct bin using `threshold_score` (the actual score value), not the rank.

**No interface changes** to `detect_threshold` or `format_score_histogram`.

---

## Tests

Both fixes are in existing tested modules.

**Chain ID fix:**
- Add a test to `tests/test_trajectory_loading.py` (or equivalent) that constructs a mock MDTraj topology with `chain_id=None` and verifies the loader produces `chain_id="A"` for chain index 0 and `chain_id="B"` for chain index 1.

**Threshold fix:**
- Add a test to `tests/test_threshold.py`: given a score list where the first 5 scores are [0.9, 0.8, 0.7, 0.6, 0.5] and 10,000 near-zero scores follow, `detect_threshold(scores[:10])` returns a rank between 2 and 5, while `detect_threshold(scores)` returns rank 1. This documents the behavior difference and confirms the fix works.
- Add a test to `tests/test_cli_analyze_pml.py` or a new `tests/test_analyze_threshold.py` that verifies the threshold line in the report says `"of {top_k} scored pairs"` when top_k < total pairs.

---

## Global Constraints

- No new runtime dependencies.
- `from __future__ import annotations` at top of every modified module.
- One commit per fix.
