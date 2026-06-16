# Model Robustness & Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `influence` model train and score faster and at larger scale (GPU, hundreds–thousands of residues) while preserving its dense no-cutoff `N×N` network, and harden training stability, reproducibility, and feature correctness.

**Architecture:** Approach C — keep the dense attention topology but make its value-aggregation memory-safe via residue chunking; vectorize and batch the scoring path; batch the Kabsch alignment; and add a robustness bundle (input normalization, gradient clipping, LR scheduling, finite-loss guard, full seeding, degenerate-`N` guard). Every change is opt-in and back-compatible; old configs and checkpoints behave exactly as before.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML, pytest.

## Global Constraints

- Python `>=3.11`; use `from __future__ import annotations` at the top of every module (matches existing files).
- Dependencies limited to `numpy`, `torch`, `PyYAML` (runtime) and `pytest` (dev). Do **not** add new dependencies.
- All new config keys are optional with defaults; existing configs must load and behave unchanged.
- Old checkpoints must reproduce their original scores exactly: when a checkpoint config snapshot lacks `data.normalize` / `model.residue_chunk_size`, fall back to legacy behavior (`normalize=False`, `residue_chunk_size=None`).
- Follow existing code style: 4-space indent, single quotes in most modules, frozen `@dataclass(slots=True)` for value types, validation errors collected into lists and raised together (see `src/allostery/config.py`).
- Tests live beside their siblings in `tests/` and use the `fixture_path` fixture from `tests/conftest.py` where a trajectory is needed.
- Run the full suite with `pytest -q` before declaring any task done.

---

### Task 1: Batched + vectorized influence scoring

Replaces the one-window-at-a-time `batch=1` loop and the pure-Python `O(N²)` pair double-loop in `score_influence_trajectory` with a batched forward pass plus a vectorized `triu` gather. Output (values + ordering) is unchanged.

**Files:**
- Modify: `src/allostery/pipeline/influence_score.py`
- Test: `tests/test_influence_scoring.py`

**Interfaces:**
- Consumes: `iter_batches`, `stack_influence_batch` from `allostery.training.runtime`; `resolve_device` (for an optional `device` param).
- Produces: `score_influence_trajectory(model, pdb_path, window_size, stride, time_step=1.0, preprocess='none', topology_path=None, normalize=False, batch_size=8, device='cpu') -> list[InfluencePairScore]` — same return shape and ordering as today. (The `normalize` param is added here now and wired in Task 8; default `False` keeps legacy behavior.)

- [ ] **Step 1: Write the failing test** — add to `tests/test_influence_scoring.py`:

```python
def test_batched_scoring_matches_unbatched(fixture_path: Path) -> None:
    result = train_influence_model(
        pdb_path=fixture_path / 'tiny_trajectory.pdb',
        window_size=3, stride=1, time_step=1.0,
        hidden_dim=8, num_encoder_layers=1, dropout=0.0,
        epochs=1, learning_rate=1e-3, sparsity_weight=0.0,
        validation_fraction=0.0, patience=0, seed=0, device='cpu', batch_size=1,
    )
    common = dict(pdb_path=fixture_path / 'tiny_trajectory.pdb',
                  window_size=3, stride=1, time_step=1.0)
    one = score_influence_trajectory(model=result.model, batch_size=1, **common)
    many = score_influence_trajectory(model=result.model, batch_size=8, **common)
    assert len(one) == len(many)
    for a, b in zip(one, many, strict=True):
        assert a['residue_i']['index'] == b['residue_i']['index']
        assert a['residue_j']['index'] == b['residue_j']['index']
        assert abs(a['score'] - b['score']) < 1e-6
        assert abs(a['influence_i_on_j'] - b['influence_i_on_j']) < 1e-6
        assert abs(a['influence_j_on_i'] - b['influence_j_on_i']) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_influence_scoring.py::test_batched_scoring_matches_unbatched -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'batch_size'`

- [ ] **Step 3: Rewrite the scoring body**

Replace the loop and pair-building section of `score_influence_trajectory` (the function signature line and everything from `num_residues = ...` to `return scores`) with:

```python
def score_influence_trajectory(
    model: AllostericInfluenceModel,
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float = 1.0,
    preprocess: str = 'none',
    topology_path: str | Path | None = None,
    normalize: bool = False,
    batch_size: int = 8,
    device: str = 'cpu',
) -> list[InfluencePairScore]:
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
    samples = build_influence_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        preprocess=preprocess,
        normalize=normalize,
    )
    if not samples:
        raise ValueError('trajectory did not yield any influence scoring windows')

    torch_device = resolve_device(device)
    num_residues = trajectory.coordinates.shape[1]
    accumulated = torch.zeros(num_residues, num_residues, device=torch_device)
    count = 0

    model = model.to(torch_device)
    model.eval()
    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size):
            batch = stack_influence_batch(batch_samples, torch_device)
            output = model(batch.state_features)
            accumulated += output['influence_matrix'].sum(dim=0)
            count += len(batch_samples)

    mean_influence = (accumulated / max(count, 1)).cpu()  # [N, N]

    rows, cols = torch.triu_indices(num_residues, num_residues, offset=1)
    i_on_j = mean_influence[cols, rows]   # influence of i on j  (A[j, i])
    j_on_i = mean_influence[rows, cols]   # influence of j on i  (A[i, j])
    pair_score = (i_on_j + j_on_i) / 2.0

    scores: list[InfluencePairScore] = [
        {
            'residue_i': _residue_identifier(trajectory.residues[int(i)]),
            'residue_j': _residue_identifier(trajectory.residues[int(j)]),
            'score': float(pair_score[k].item()),
            'influence_i_on_j': float(i_on_j[k].item()),
            'influence_j_on_i': float(j_on_i[k].item()),
            'support_count': count,
        }
        for k, (i, j) in enumerate(zip(rows.tolist(), cols.tolist()))
    ]
    scores.sort(key=lambda item: item['score'], reverse=True)
    return scores
```

Add the import at the top of the file (the others are already present):

```python
from allostery.training.runtime import iter_batches, resolve_device, stack_influence_batch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_influence_scoring.py -v`
Expected: PASS (all three tests, including the existing `covers_all_pairs` and `ranked_pairs`).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/influence_score.py tests/test_influence_scoring.py
git commit -m "perf: batch and vectorize influence scoring"
```

---

### Task 2: Batched Kabsch alignment

Replace the per-frame Python loop in `align_trajectory_coordinates` with a single batched covariance + `np.linalg.svd` over the frame axis.

**Files:**
- Modify: `src/allostery/features/alignment.py`
- Test: `tests/test_alignment_features.py`

**Interfaces:**
- Produces: `align_trajectory_coordinates(window_coordinates, reference_frame_index=0) -> np.ndarray` — same signature and output (within float tolerance) as today.

- [ ] **Step 1: Write the failing test** — add to `tests/test_alignment_features.py`:

```python
import numpy as np
from allostery.features.alignment import align_trajectory_coordinates, _kabsch_align


def test_batched_align_matches_per_frame_loop() -> None:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((5, 7, 3)).astype(np.float32)
    reference = coords[0]
    expected = np.stack([_kabsch_align(frame, reference) for frame in coords], axis=0)
    result = align_trajectory_coordinates(coords, reference_frame_index=0)
    assert result.shape == coords.shape
    np.testing.assert_allclose(result, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails or passes against current loop**

Run: `pytest tests/test_alignment_features.py::test_batched_align_matches_per_frame_loop -v`
Expected: PASS against the current loop (this test pins behavior). Keep it — it guards the rewrite.

- [ ] **Step 3: Replace the loop with a batched implementation**

Replace the body of `align_trajectory_coordinates` (everything after the `reference_frame_index` bounds check) with:

```python
    reference = coordinates[reference_frame_index]
    reference_centroid = reference.mean(axis=0, keepdims=True)
    reference_centered = reference - reference_centroid

    frame_centroids = coordinates.mean(axis=1, keepdims=True)        # [T, 1, 3]
    mobile_centered = coordinates - frame_centroids                  # [T, N, 3]

    # Per-frame covariance: [T, 3, 3] = mobileᵀ @ reference
    covariance = np.einsum('tni,nj->tij', mobile_centered, reference_centered)
    left, _, right_t = np.linalg.svd(covariance)                    # [T,3,3] each
    det = np.linalg.det(np.einsum('tij,tjk->tik', right_t.transpose(0, 2, 1), left.transpose(0, 2, 1)))
    sign = np.where(det < 0.0, -1.0, 1.0)                           # [T]
    right_t = right_t.copy()
    right_t[:, -1, :] *= sign[:, None]
    rotation = np.einsum('tij,tjk->tik', right_t.transpose(0, 2, 1), left.transpose(0, 2, 1))

    aligned = np.einsum('tni,tij->tnj', mobile_centered, rotation) + reference_centroid
    return aligned.astype(np.float32, copy=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_alignment_features.py -v`
Expected: PASS (the new equivalence test plus all existing alignment tests).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/features/alignment.py tests/test_alignment_features.py
git commit -m "perf: vectorize Kabsch alignment over frames"
```

---

### Task 3: Full reproducible seeding

Extend `seed_everything` to seed NumPy and CUDA, with an opt-in `deterministic` flag for cuDNN.

**Files:**
- Modify: `src/allostery/training/runtime.py`
- Test: `tests/test_training_runtime.py`

**Interfaces:**
- Produces: `seed_everything(seed: int, deterministic: bool = False) -> None` — now also seeds `numpy` and `torch.cuda`; when `deterministic` is `True`, sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_training_runtime.py`:

```python
import numpy as np
from allostery.training.runtime import seed_everything


def test_seed_everything_seeds_numpy() -> None:
    seed_everything(123)
    first = np.random.rand(5)
    seed_everything(123)
    second = np.random.rand(5)
    np.testing.assert_array_equal(first, second)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_runtime.py::test_seed_everything_seeds_numpy -v`
Expected: FAIL (NumPy stream is not reseeded, arrays differ).

- [ ] **Step 3: Implement**

In `src/allostery/training/runtime.py`, add `import numpy as np` near the top imports and replace `seed_everything`:

```python
def seed_everything(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/training/runtime.py tests/test_training_runtime.py
git commit -m "fix: seed numpy and cuda in seed_everything for reproducibility"
```

---

### Task 4: Gradient clipping + finite-loss guard

Add gradient-norm clipping and a non-finite-loss guard to the influence training loop.

**Files:**
- Modify: `src/allostery/pipeline/influence_train.py`
- Test: `tests/test_influence_training.py`

**Interfaces:**
- Produces: `train_influence_model(..., grad_clip_norm: float | None = 1.0, ...)` — new keyword arg added after `batch_size`. Non-finite batch loss raises `ValueError` naming the epoch and batch index.

- [ ] **Step 1: Write the failing test** — add to `tests/test_influence_training.py`:

```python
import pytest
from allostery.pipeline.influence_train import train_influence_model


def test_non_finite_loss_raises(fixture_path, monkeypatch) -> None:
    import allostery.pipeline.influence_train as mod

    def fake_loss(prediction, target_acceleration, sparsity_weight):
        bad = prediction['acceleration'].sum() * float('inf')
        return mod.InfluenceLossBreakdown(reconstruction=bad, sparsity=bad * 0.0)

    monkeypatch.setattr(mod, 'influence_loss', fake_loss)
    with pytest.raises(ValueError, match='non-finite'):
        train_influence_model(
            pdb_path=fixture_path / 'tiny_trajectory.pdb',
            window_size=3, stride=1, time_step=1.0,
            hidden_dim=8, num_encoder_layers=1, dropout=0.0,
            epochs=1, learning_rate=1e-3, sparsity_weight=0.0,
            validation_fraction=0.0, patience=0, seed=0, device='cpu', batch_size=1,
        )
```

Note: `InfluenceLossBreakdown` must be importable from `influence_train`; add `InfluenceLossBreakdown` to the existing `from allostery.training.influence_objectives import influence_loss` import line so it reads `from allostery.training.influence_objectives import InfluenceLossBreakdown, influence_loss`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_influence_training.py::test_non_finite_loss_raises -v`
Expected: FAIL (no guard; training completes without raising).

- [ ] **Step 3: Implement** — in `train_influence_model`, add `grad_clip_norm: float | None = 1.0` to the signature (after `batch_size: int = 4,`). Replace the inner training batch block:

```python
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = stack_influence_batch(batch_samples, torch_device)
            output = model(batch.state_features)
            losses = influence_loss(output, batch.acceleration_targets, sparsity_weight=sparsity_weight)
            if not torch.isfinite(losses.total):
                raise ValueError(
                    f'non-finite training loss at epoch {epoch + 1}, batch {epoch_batch_count + 1}'
                )
            optimizer.zero_grad()
            losses.total.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            last_loss = float(losses.total.detach().item())
            epoch_loss_sum += last_loss
            epoch_batch_count += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_influence_training.py -v`
Expected: PASS (new guard test plus existing training tests).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/influence_train.py tests/test_influence_training.py
git commit -m "feat: add gradient clipping and finite-loss guard to influence training"
```

---

### Task 5: Degenerate `N<2` guard in the influence model

When fewer than two residues are present, the masked softmax is all `-inf` → NaN. Return baseline-only output with a zero influence contribution.

**Files:**
- Modify: `src/allostery/models/influence.py`
- Test: `tests/test_influence_model.py`

**Interfaces:**
- Produces: `AllostericInfluenceModel.forward` returns finite `acceleration` and an `influence_matrix` of shape `[batch, N, N]` (zeros when `N < 2`).

- [ ] **Step 1: Write the failing test** — add to `tests/test_influence_model.py`:

```python
import torch
from allostery.models.influence import AllostericInfluenceModel


def test_forward_single_residue_is_finite() -> None:
    model = AllostericInfluenceModel(state_dim=6, hidden_dim=8, num_encoder_layers=1)
    state = torch.randn(2, 4, 1, 6)  # batch=2, time=4, N=1, state_dim=6
    out = model(state)
    assert out['acceleration'].shape == (2, 4, 1, 3)
    assert out['influence_matrix'].shape == (2, 1, 1)
    assert torch.isfinite(out['acceleration']).all()
    assert torch.isfinite(out['influence_matrix']).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_influence_model.py::test_forward_single_residue_is_finite -v`
Expected: FAIL (NaN from all-`-inf` softmax row).

- [ ] **Step 3: Implement** — in `forward`, immediately after computing `batch_size, num_steps, num_residues, _ = state_features.shape`, insert:

```python
        if num_residues < 2:
            baseline = self.baseline_net(state_features)  # [batch, time, N, 3]
            influence_matrix = torch.zeros(
                batch_size, num_residues, num_residues, device=state_features.device
            )
            return {'acceleration': baseline, 'influence_matrix': influence_matrix}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_influence_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/models/influence.py tests/test_influence_model.py
git commit -m "fix: guard influence model against degenerate N<2 input"
```

---

### Task 6: Memory-safe chunked aggregation

Add an optional `residue_chunk_size` to the influence model that tiles the value-aggregation matmul over the receiver dimension, bounding peak memory. Output is numerically identical to the unchunked path.

**Files:**
- Modify: `src/allostery/models/influence.py`
- Test: `tests/test_influence_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AllostericInfluenceModel(state_dim, hidden_dim, num_encoder_layers=2, dropout=0.0, residue_chunk_size: int | None = None)` — stores `self.residue_chunk_size`. When set, aggregation is computed in chunks of receivers; when `None`, behavior is unchanged. `residue_chunk_size > N` is treated as `N`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_influence_model.py`:

```python
def test_chunked_aggregation_matches_dense() -> None:
    torch.manual_seed(0)
    dense = AllostericInfluenceModel(state_dim=6, hidden_dim=8, num_encoder_layers=2)
    chunked = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=2, residue_chunk_size=2
    )
    chunked.load_state_dict(dense.state_dict())
    dense.eval()
    chunked.eval()
    state = torch.randn(2, 4, 5, 6)  # N=5, chunk=2 -> chunks of 2,2,1
    with torch.no_grad():
        a = dense(state)
        b = chunked(state)
    torch.testing.assert_close(a['acceleration'], b['acceleration'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a['influence_matrix'], b['influence_matrix'], atol=1e-6, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_influence_model.py::test_chunked_aggregation_matches_dense -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'residue_chunk_size'`

- [ ] **Step 3: Implement** — add the constructor param and store it. In `__init__`, after `self.hidden_dim = hidden_dim`, add:

```python
        if residue_chunk_size is not None and residue_chunk_size <= 0:
            raise ValueError('residue_chunk_size must be greater than zero')
        self.residue_chunk_size = residue_chunk_size
```

and update the signature to:

```python
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_encoder_layers: int = 2,
        dropout: float = 0.0,
        residue_chunk_size: int | None = None,
    ) -> None:
```

In `forward`, replace the single-line aggregation:

```python
        aggregated = torch.matmul(influence_matrix.unsqueeze(1), V)  # [batch, time, N, hidden]
```

with a chunked version:

```python
        chunk = self.residue_chunk_size
        if chunk is None or chunk >= num_residues:
            aggregated = torch.matmul(influence_matrix.unsqueeze(1), V)
        else:
            parts = []
            for start in range(0, num_residues, chunk):
                stop = min(start + chunk, num_residues)
                # influence rows for receivers [start:stop] over all senders
                rows = influence_matrix[:, start:stop, :].unsqueeze(1)  # [b,1,c,N]
                parts.append(torch.matmul(rows, V))                    # [b,t,c,hidden]
            aggregated = torch.cat(parts, dim=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_influence_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/models/influence.py tests/test_influence_model.py
git commit -m "perf: add memory-safe chunked aggregation to influence model"
```

---

### Task 7: AMP (mixed precision) for GPU training

Add opt-in CUDA mixed precision to influence training; no-op with a warning on CPU.

**Files:**
- Modify: `src/allostery/pipeline/influence_train.py`
- Test: `tests/test_influence_training.py`

**Interfaces:**
- Produces: `train_influence_model(..., mixed_precision: bool = False, ...)` — new keyword after `grad_clip_norm`. On CPU it is forced off (with a `warnings.warn`); the training result is unchanged versus `mixed_precision=False`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_influence_training.py`:

```python
import warnings


def test_mixed_precision_on_cpu_warns_and_runs(fixture_path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = train_influence_model(
            pdb_path=fixture_path / 'tiny_trajectory.pdb',
            window_size=3, stride=1, time_step=1.0,
            hidden_dim=8, num_encoder_layers=1, dropout=0.0,
            epochs=1, learning_rate=1e-3, sparsity_weight=0.0,
            validation_fraction=0.0, patience=0, seed=0, device='cpu',
            batch_size=1, mixed_precision=True,
        )
    assert result.num_samples > 0
    assert any('mixed_precision' in str(w.message) for w in caught)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_influence_training.py::test_mixed_precision_on_cpu_warns_and_runs -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'mixed_precision'`

- [ ] **Step 3: Implement** — add `import warnings` at the top. Add `mixed_precision: bool = False` to the signature (after `grad_clip_norm`). After `torch_device = resolve_device(device)`, add:

```python
    use_amp = mixed_precision and torch_device.type == 'cuda'
    if mixed_precision and not use_amp:
        warnings.warn('mixed_precision requested but device is not CUDA; running in full precision')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
```

Replace the training batch block's forward/backward with an AMP-aware version:

```python
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = stack_influence_batch(batch_samples, torch_device)
            optimizer.zero_grad()
            with torch.autocast(device_type=torch_device.type, enabled=use_amp):
                output = model(batch.state_features)
                losses = influence_loss(output, batch.acceleration_targets, sparsity_weight=sparsity_weight)
            if not torch.isfinite(losses.total):
                raise ValueError(
                    f'non-finite training loss at epoch {epoch + 1}, batch {epoch_batch_count + 1}'
                )
            scaler.scale(losses.total).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            last_loss = float(losses.total.detach().item())
            epoch_loss_sum += last_loss
            epoch_batch_count += 1
```

(This block supersedes the one from Task 4; the finite-loss guard and clipping are preserved.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_influence_training.py -v`
Expected: PASS (CPU path runs full precision, warning emitted).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/influence_train.py tests/test_influence_training.py
git commit -m "feat: add opt-in CUDA mixed precision to influence training"
```

---

### Task 8: Translation-invariant input normalization

Add a `normalize` flag that removes each frame's centroid from the **position** features only, making them translation-invariant. Threaded through dynamics → influence samples → train/score. Defaults `False` at the low level (so `cri` and existing callers are unaffected) and is set `True` by the influence pipeline defaults.

**Files:**
- Modify: `src/allostery/features/dynamics.py`
- Modify: `src/allostery/influence/data.py`
- Modify: `src/allostery/pipeline/influence_train.py`
- Modify: `src/allostery/pipeline/influence_score.py` (param already added in Task 1)
- Test: `tests/test_dynamics_features.py`

**Interfaces:**
- Produces:
  - `build_residue_dynamics(window_coordinates, time_step=1.0, preprocess='none', reference_frame_index=0, normalize=False) -> ResidueDynamics`
  - `build_influence_samples(coordinates, window_size, stride, time_step=1.0, preprocess='none', normalize=False) -> list[InfluenceSample]`
  - `train_influence_model(..., normalize: bool = True, ...)` (new keyword after `mixed_precision`)
  - `score_influence_trajectory(..., normalize: bool = False, ...)` (already added in Task 1; the CLI passes the value read from the checkpoint snapshot — see Task 10)

- [ ] **Step 1: Write the failing test** — add to `tests/test_dynamics_features.py`:

```python
import numpy as np
from allostery.features.dynamics import build_residue_dynamics


def test_normalize_makes_positions_translation_invariant() -> None:
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((4, 6, 3)).astype(np.float32)
    shifted = coords + np.array([10.0, -5.0, 3.0], dtype=np.float32)
    base = build_residue_dynamics(coords, normalize=True)
    moved = build_residue_dynamics(shifted, normalize=True)
    np.testing.assert_allclose(base.positions, moved.positions, atol=1e-4)


def test_normalize_false_keeps_absolute_positions() -> None:
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((4, 6, 3)).astype(np.float32)
    shifted = coords + 10.0
    base = build_residue_dynamics(coords, normalize=False)
    moved = build_residue_dynamics(shifted, normalize=False)
    assert not np.allclose(base.positions, moved.positions, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamics_features.py -k normalize -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'normalize'`

- [ ] **Step 3: Implement in `dynamics.py`** — add `normalize: bool = False` to `build_residue_dynamics` (after `reference_frame_index`). After `positions = coordinates[1:-1]`, insert:

```python
    if normalize:
        positions = positions - positions.mean(axis=1, keepdims=True)
```

- [ ] **Step 4: Thread through `influence/data.py`** — add `normalize: bool = False` to `build_influence_samples` (after `preprocess`) and pass it into the `build_residue_dynamics(...)` call:

```python
        dynamics = build_residue_dynamics(
            window, time_step=time_step, preprocess=preprocess, normalize=normalize
        )
```

- [ ] **Step 5: Thread through `influence_train.py`** — add `normalize: bool = True` to `train_influence_model` (after `mixed_precision`) and pass `normalize=normalize` into the `build_influence_samples(...)` call.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_dynamics_features.py tests/test_influence_training.py tests/test_influence_scoring.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/allostery/features/dynamics.py src/allostery/influence/data.py \
        src/allostery/pipeline/influence_train.py tests/test_dynamics_features.py
git commit -m "feat: add translation-invariant input normalization for influence model"
```

---

### Task 9: LR scheduling (ReduceLROnPlateau)

Add an opt-in `ReduceLROnPlateau` scheduler driven by validation loss.

**Files:**
- Modify: `src/allostery/pipeline/influence_train.py`
- Test: `tests/test_influence_training.py`

**Interfaces:**
- Produces: `train_influence_model(..., lr_scheduler: str = 'plateau', ...)` (new keyword after `normalize`). Accepted values: `'none'`, `'plateau'`. With validation enabled and `'plateau'`, an `Adam` LR is reduced on validation-loss plateaus; with `'none'` or no validation, LR is constant.

- [ ] **Step 1: Write the failing test** — add to `tests/test_influence_training.py`:

```python
import pytest


def test_invalid_lr_scheduler_rejected(fixture_path) -> None:
    with pytest.raises(ValueError, match='lr_scheduler'):
        train_influence_model(
            pdb_path=fixture_path / 'tiny_trajectory.pdb',
            window_size=3, stride=1, time_step=1.0,
            hidden_dim=8, num_encoder_layers=1, dropout=0.0,
            epochs=1, learning_rate=1e-3, sparsity_weight=0.0,
            validation_fraction=0.0, patience=0, seed=0, device='cpu',
            batch_size=1, lr_scheduler='bogus',
        )


def test_plateau_scheduler_runs_with_validation(fixture_path) -> None:
    result = train_influence_model(
        pdb_path=fixture_path / 'tiny_trajectory.pdb',
        window_size=3, stride=1, time_step=1.0,
        hidden_dim=8, num_encoder_layers=1, dropout=0.0,
        epochs=2, learning_rate=1e-3, sparsity_weight=0.0,
        validation_fraction=0.5, patience=0, seed=0, device='cpu',
        batch_size=1, lr_scheduler='plateau',
    )
    assert result.num_samples > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_influence_training.py -k lr_scheduler -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'lr_scheduler'`

- [ ] **Step 3: Implement** — add `lr_scheduler: str = 'plateau'` to the signature (after `normalize`). After the `optimizer = torch.optim.Adam(...)` line, insert:

```python
    if lr_scheduler not in {'none', 'plateau'}:
        raise ValueError(f"lr_scheduler must be one of none, plateau (got {lr_scheduler!r})")
    scheduler = None
    if lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
```

Inside the `if validation_samples:` block, right after `validation_loss = _evaluate_epoch(...)`, add:

```python
            if scheduler is not None:
                scheduler.step(validation_loss)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_influence_training.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/influence_train.py tests/test_influence_training.py
git commit -m "feat: add ReduceLROnPlateau scheduler to influence training"
```

---

### Task 10: Config keys, validation & checkpoint wiring

Surface all new behavior through config: add the six new keys to `config.py` (parsing + validation + known-key sets), thread them through the CLI, and persist `normalize`/`residue_chunk_size` in the checkpoint snapshot so old checkpoints reproduce legacy behavior.

**Files:**
- Modify: `src/allostery/config.py`
- Modify: `src/allostery/cli.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `train_influence_model` / `score_influence_trajectory` keyword args from Tasks 1–9.
- Produces: new fields on the config dataclasses —
  - `DataConfig.normalize: bool = True`
  - `ModelConfig.residue_chunk_size: int | None = None`
  - `TrainingConfig.mixed_precision: bool = False`
  - `TrainingConfig.grad_clip_norm: float | None = 1.0`
  - `TrainingConfig.lr_scheduler: str = 'plateau'`
  - `TrainingConfig.deterministic: bool = False`

- [ ] **Step 1: Write the failing tests** — add to `tests/test_config.py` (follow the file's existing helper for writing a temp YAML config; if it uses a `write_config`/`tmp_path` pattern, mirror it):

```python
import pytest
from allostery.config import ConfigError, load_config


def _base_config_text(extra_model: str = '', extra_training: str = '', extra_data: str = '') -> str:
    return f"""
mode: run
data:
  pdb_path: tiny_trajectory.pdb
  window_size: 3
  horizon_size: 1
  stride: 1
{extra_data}
model:
  family: influence
  hidden_dim: 8
  residue_layers: 1
  pair_layers: 1
  dropout: 0.0
{extra_model}
training:
  epochs: 1
  learning_rate: 0.001
  consistency_weight: 0.0
{extra_training}
scoring:
  top_k: 5
output:
  model_path: out/model.pt
  score_csv_path: out/scores.csv
"""


def test_new_keys_default(tmp_path) -> None:
    (tmp_path / 'tiny_trajectory.pdb').write_text('MODEL\nENDMDL\n')
    cfg_path = tmp_path / 'c.yaml'
    cfg_path.write_text(_base_config_text())
    cfg = load_config(cfg_path)
    assert cfg.data.normalize is True
    assert cfg.model.residue_chunk_size is None
    assert cfg.training.mixed_precision is False
    assert cfg.training.grad_clip_norm == 1.0
    assert cfg.training.lr_scheduler == 'plateau'
    assert cfg.training.deterministic is False


def test_bad_lr_scheduler_rejected(tmp_path) -> None:
    (tmp_path / 'tiny_trajectory.pdb').write_text('MODEL\nENDMDL\n')
    cfg_path = tmp_path / 'c.yaml'
    cfg_path.write_text(_base_config_text(extra_training='  lr_scheduler: bogus\n'))
    with pytest.raises(ConfigError, match='lr_scheduler'):
        load_config(cfg_path)


def test_bad_residue_chunk_size_rejected(tmp_path) -> None:
    (tmp_path / 'tiny_trajectory.pdb').write_text('MODEL\nENDMDL\n')
    cfg_path = tmp_path / 'c.yaml'
    cfg_path.write_text(_base_config_text(extra_model='  residue_chunk_size: 0\n'))
    with pytest.raises(ConfigError, match='residue_chunk_size'):
        load_config(cfg_path)
```

(If `tests/test_config.py` already has a config-writing helper, use it instead of `_base_config_text` to stay DRY.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -k "new_keys or lr_scheduler or residue_chunk" -v`
Expected: FAIL (`AttributeError`/no validation for the new keys).

- [ ] **Step 3: Add keys to the known-key sets** — in `config.py`:

```python
_DATA_KEYS: frozenset[str] = frozenset({
    'pdb_path', 'window_size', 'horizon_size', 'stride', 'time_step',
    'distance_cutoff', 'max_neighbors', 'min_sequence_separation', 'preprocess', 'topology_path',
    'normalize',
})
_MODEL_KEYS: frozenset[str] = frozenset({
    'family', 'hidden_dim', 'residue_layers', 'pair_layers', 'dropout', 'edge_types',
    'residue_chunk_size',
})
_TRAINING_KEYS: frozenset[str] = frozenset({
    'epochs', 'learning_rate', 'consistency_weight', 'entropy_weight', 'no_edge_weight',
    'sparsity_weight', 'validation_fraction', 'patience', 'seed', 'device', 'batch_size', 'verbose',
    'mixed_precision', 'grad_clip_norm', 'lr_scheduler', 'deterministic',
})
```

- [ ] **Step 4: Add dataclass fields** — append to each dataclass (after the last existing field, preserving order):

```python
# DataConfig:
    normalize: bool = True

# ModelConfig:
    residue_chunk_size: int | None = None

# TrainingConfig:
    mixed_precision: bool = False
    grad_clip_norm: float | None = 1.0
    lr_scheduler: str = 'plateau'
    deterministic: bool = False
```

- [ ] **Step 5: Parse the new keys** — in `load_config`, extend the constructors:

```python
# DataConfig(...):
            normalize=bool(data_raw.get('normalize', True)),

# ModelConfig(...):
            residue_chunk_size=(
                int(model_raw['residue_chunk_size'])
                if model_raw.get('residue_chunk_size') is not None else None
            ),

# TrainingConfig(...):  (inside the `if mode in {'train', 'run'}` block)
            mixed_precision=bool(training_raw.get('mixed_precision', False)),
            grad_clip_norm=(
                float(training_raw['grad_clip_norm'])
                if training_raw.get('grad_clip_norm') is not None else None
            ),
            lr_scheduler=str(training_raw.get('lr_scheduler', 'plateau')),
            deterministic=bool(training_raw.get('deterministic', False)),
```

- [ ] **Step 6: Validate the new keys** — in `validate_config`, add to the model block:

```python
    if config.model.residue_chunk_size is not None and config.model.residue_chunk_size <= 0:
        errors.append(
            f"model.residue_chunk_size must be > 0 (got {config.model.residue_chunk_size})"
        )
```

and inside the `if config.mode in {'train', 'run'}:` → `else:` training block:

```python
            if config.training.lr_scheduler not in {'none', 'plateau'}:
                errors.append(
                    f"training.lr_scheduler must be one of none, plateau "
                    f"(got {config.training.lr_scheduler!r})"
                )
            if config.training.grad_clip_norm is not None and config.training.grad_clip_norm <= 0:
                errors.append(
                    f"training.grad_clip_norm must be > 0 (got {config.training.grad_clip_norm})"
                )
```

- [ ] **Step 7: Run config tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 8: Wire the CLI train path** — in `src/allostery/cli.py`, in `_run_train`'s `train_influence_model(...)` call, add:

```python
            normalize=config.data.normalize,
            grad_clip_norm=config.training.grad_clip_norm,
            mixed_precision=config.training.mixed_precision,
            lr_scheduler=config.training.lr_scheduler,
```

In `train_influence_model`, also thread `residue_chunk_size` into model construction: add `residue_chunk_size: int | None = None` to its signature and pass `residue_chunk_size=residue_chunk_size` into `AllostericInfluenceModel(...)`; then add `residue_chunk_size=config.model.residue_chunk_size` to the CLI call. Also pass `deterministic` to seeding: change `seed_everything(seed)` in `train_influence_model` to `seed_everything(seed, deterministic=deterministic)` and add `deterministic: bool = False` to the signature plus `deterministic=config.training.deterministic` to the CLI call.

- [ ] **Step 9: Persist and replay normalize/chunk via checkpoint** — in `train_influence_model`'s `save_checkpoint(...)` `metadata['training']` dict, add `'normalize': normalize` and `'residue_chunk_size': residue_chunk_size`. In `cli.py`'s `_run_score` influence branch, read these back from the checkpoint so old checkpoints replay legacy behavior:

```python
    if config.model.family == 'influence':
        from allostery.io.checkpoint import load_checkpoint
        snapshot = load_checkpoint(model_path).metadata.get('training', {})
        scores = score_influence_trajectory(
            model=model,  # type: ignore[arg-type]
            pdb_path=config.data.pdb_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            preprocess=config.data.preprocess,
            normalize=bool(snapshot.get('normalize', False)),
            device=config.training.device if config.training else 'cpu',
        )
```

(Legacy checkpoints lack `normalize` in metadata → defaults to `False`, reproducing original scores.)

- [ ] **Step 10: Run the full suite**

Run: `pytest -q`
Expected: PASS (all tests).

- [ ] **Step 11: Commit**

```bash
git add src/allostery/config.py src/allostery/cli.py src/allostery/pipeline/influence_train.py \
        tests/test_config.py
git commit -m "feat: surface robustness/efficiency options through config and CLI"
```

---

### Task 11: Documentation

Document the new config keys in the README config reference and tutorial.

**Files:**
- Modify: `README.md`
- Modify: `docs/tutorial.md`

- [ ] **Step 1: Update the README config reference** — in the "Config Reference" key list (around the `influence` parameters), add bullet lines:

```markdown
- `data.normalize` — remove each frame's centroid from position features for translation invariance (default `true`)
- `model.residue_chunk_size` — tile the influence aggregation over receivers to bound peak memory on large proteins (default: unset = dense)
- `training.mixed_precision` — enable CUDA autocast/GradScaler (default `false`; no-op on CPU)
- `training.grad_clip_norm` — max gradient norm (default `1.0`; set to null to disable)
- `training.lr_scheduler` — `none` or `plateau` (default `plateau`)
- `training.deterministic` — set cuDNN deterministic flags for reproducible GPU runs (default `false`)
```

- [ ] **Step 2: Mirror the keys in the tutorial** — add the same six keys with one-line descriptions to the config section of `docs/tutorial.md`, matching its existing formatting.

- [ ] **Step 3: Verify docs build/links** — confirm no broken internal links:

Run: `grep -n "residue_chunk_size\|mixed_precision\|lr_scheduler\|normalize" README.md docs/tutorial.md`
Expected: matches in both files.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/tutorial.md
git commit -m "docs: document new robustness and efficiency config keys"
```

---

## Self-Review Notes

- **Spec coverage:** 1a→Task 6, 1b/1c→Task 1, 1d→Task 2, 1e→Task 7, 2a→Task 8, 2b/2d→Task 4, 2c→Task 9, 2e→Task 3, 2f→Task 5, Section 3 (config/validation/checkpoint back-compat)→Task 10, docs→Task 11. All spec sections map to a task.
- **Back-compat:** legacy checkpoints lack `normalize`/`residue_chunk_size` in metadata; Task 10 Step 9 defaults `normalize` to `False` on replay, preserving original scores. `build_residue_dynamics` defaults `normalize=False`, so the `cri` and `relational` paths are untouched.
- **Type consistency:** `train_influence_model` keyword order across tasks is `... batch_size, grad_clip_norm, mixed_precision, normalize, lr_scheduler, deterministic, residue_chunk_size`; `score_influence_trajectory` adds `normalize, batch_size, device`. The Task 7 training-batch block supersedes Task 4's (both shown in full to avoid ambiguity).
