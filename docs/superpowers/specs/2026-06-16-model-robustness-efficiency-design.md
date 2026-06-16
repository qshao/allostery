# Model Robustness & Efficiency — Design

**Date:** 2026-06-16
**Status:** Approved (brainstorming) — pending spec review
**Scope target:** Large proteins / GPU, balanced robustness + efficiency
**Approach:** C — preserve the dense, no-cutoff `N×N` influence topology; make its
execution memory-safe and fast; bundle training-robustness and feature-correctness fixes.

## Goal

Make the `influence` model train and score faster and at larger scale (hundreds to
thousands of residues, on GPU) **without** abandoning its defining property: every
residue can influence every other with no spatial cutoff. Simultaneously improve
training stability, reproducibility, and the physical correctness of the learned
network. All changes are opt-in and back-compatible: existing configs and checkpoints
keep working unchanged.

## Non-goals

- Replacing dense attention with a sparse/cutoff model as the default (that would drop
  long-range allostery, the core premise). A spatial cutoff remains available only as an
  explicit opt-in for users who want it; it is never the default.
- Changes to the `cri` or `relational` families beyond shared runtime utilities.
- Distributed / multi-GPU training.

## Workstreams

The work splits into three independent workstreams touching distinct files, so each can
be implemented and tested on its own.

### 1. Scale & efficiency (execution path)

**1a. Memory-safe dense aggregation** — `models/influence.py`
The bottleneck is `aggregated = matmul(influence_matrix[b,1,N,N], V[b,t,N,h])`, with peak
memory `O(N²·time·hidden)`. Add `residue_chunk_size: int | None` (default `None` =
current behavior). When set, the receiver dimension `j` is processed in chunks so peak
memory is `O(chunk·N·hidden)`. Output is numerically identical to the unchunked path (a
tiled matmul). If `residue_chunk_size > N`, clamp to `N`.

**1b. Batched scoring** — `pipeline/influence_score.py`
Replace the one-window-at-a-time (`batch=1`) loop with the same `iter_batches` +
`stack_influence_batch` path that training uses, so scoring runs in batches on GPU.
Accumulate `influence_matrix.sum(dim=0)` per batch.

**1c. Vectorized pair extraction** — `pipeline/influence_score.py`
Replace the pure-Python `O(N²)` double loop with a vectorized `torch.triu_indices` gather
to build the `i_on_j` / `j_on_i` arrays, then assemble the pair dicts from those arrays.
Final ordering is unchanged (same score, same sort).

**1d. Batched Kabsch alignment** — `features/alignment.py`
Replace the per-frame Python loop with a single batched covariance + `np.linalg.svd` over
the frame axis, applying the determinant-sign correction vectorized. Output matches the
per-frame implementation within floating-point tolerance.

**1e. AMP for GPU** — `training/runtime.py` + `pipeline/influence_train.py`
Add `training.mixed_precision: bool` (default `False`). When `True` and device is CUDA,
wrap forward/loss in `torch.autocast` and use `GradScaler`. No-op (with a warning) on CPU.

### 2. Training robustness & stability

**2a. Input normalization + translation invariance** — `features/dynamics.py`
Today the model consumes absolute positions, so features depend on the box origin. Add
`normalize: bool` (default `True` for new runs). When enabled, recenter each frame's
positions by removing that frame's centroid, making position features translation-
invariant displacements. Velocities/accelerations are finite differences and already
origin-independent. This changes learned values (a quality change, not a no-op) and gets
a dedicated before/after test.

**2b. Gradient clipping** — `pipeline/influence_train.py`
Add `training.grad_clip_norm: float | None` (default `1.0`). Apply
`torch.nn.utils.clip_grad_norm_` between `backward()` and `optimizer.step()`. With AMP,
unscale before clipping.

**2c. LR scheduling** — `pipeline/influence_train.py`
Add `training.lr_scheduler: "none" | "plateau"`. Default `"plateau"` when validation is
enabled, else `"none"`. `ReduceLROnPlateau` driven by validation loss, complementing the
existing early-stop.

**2d. Finite-loss guard** — `pipeline/influence_train.py`
If a batch loss is non-finite, raise a clear error naming the epoch and batch index
instead of letting NaN propagate into the saved checkpoint.

**2e. Full reproducible seeding** — `training/runtime.py`
Extend `seed_everything` to also seed NumPy (`np.random.seed`) and, on CUDA,
`torch.cuda.manual_seed_all`. Add `training.deterministic: bool` (default `False`) to
toggle cuDNN deterministic/benchmark flags (they can slow GPU runs).

**2f. Degenerate (`N < 2`) guard** — `models/influence.py`
When `num_residues < 2`, the masked softmax is all `-inf` → NaN. Skip the attention path
and return baseline-only acceleration with a zero influence contribution, yielding finite
output.

### 3. Config, back-compat & error handling

**New config keys** (all optional, all with defaults):

| Key | Default | Notes |
|---|---|---|
| `model.residue_chunk_size` | `None` | `int` > 0; clamped to N if larger |
| `training.mixed_precision` | `False` | CUDA only; warn + no-op on CPU |
| `training.grad_clip_norm` | `1.0` | `None` or float > 0 |
| `training.lr_scheduler` | `"plateau"` | enum: `none` \| `plateau` |
| `training.deterministic` | `False` | toggles cuDNN flags |
| `data.normalize` | `True` | translation-invariant positions |

**Validation** (`config.py` + `allostery check`): each new key is range/enum-checked with
row-accurate errors, matching the existing validation-hardening style already in the tree.
Unknown/misspelled keys continue to be flagged.

**Checkpoint back-compat:** the config snapshot saved in each checkpoint already records
training settings. Scoring reads `normalize` and `residue_chunk_size` from the
checkpoint's snapshot when present, falling back to legacy behavior (`normalize=False`, no
chunking) when absent — so **old checkpoints reproduce their original scores exactly**.
New metadata fields are added to the saved snapshot.

**Error handling:** non-finite loss → explicit error (2d); `mixed_precision: True` on CPU →
warn and no-op; `residue_chunk_size > N` → clamp to N.

## Testing strategy

**Equivalence tests**
- Chunked aggregation == unchunked (1a)
- Batched scoring == loop scoring on the fixture (1b/1c)
- Batched Kabsch == per-frame Kabsch within tolerance (1d)

**Behavior tests**
- Normalization is translation-invariant: shifting all coordinates leaves state_features
  unchanged (2a)
- Grad-clip caps the gradient norm (2b)
- Finite-loss guard raises on an injected NaN (2d)
- Seeding reproduces identical runs including the NumPy path (2e)
- `N=1` returns finite output (2f)

**Config / regression tests**
- Each new key validated and rejected on bad input
- Old checkpoint scores unchanged (legacy-default regression)

All additions go in the existing `tests/` files alongside their siblings.

## Risks & mitigations

- **2a changes results** — gated by `normalize`, defaulted off for legacy checkpoints via
  the snapshot fallback; covered by an explicit before/after test.
- **Chunking correctness** — guarded by a bit-for-bit equivalence test against the dense
  path.
- **AMP numerical drift** — opt-in, CUDA-only; CPU and default runs are unaffected.

## Implementation order (by impact, low-risk first)

1. Batched + vectorized scoring (1b, 1c) — large speedup, no semantic change
2. Batched Kabsch (1d) — speedup, no semantic change
3. Robustness bundle (2b, 2d, 2e, 2f) — stability, low risk
4. Memory-safe chunking (1a) — the scale lever
5. AMP (1e) — GPU throughput
6. Input normalization (2a) — quality change, gated
7. LR scheduling (2c)
8. Config keys + validation + checkpoint snapshot wiring (3) — threaded through as each
   feature lands
