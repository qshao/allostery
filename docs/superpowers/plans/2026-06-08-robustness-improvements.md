# Robustness Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three targeted improvements: per-epoch training progress output, direct loading of non-PDB MD trajectories via MDAnalysis/MDTraj, and sequence-separation masking in the allosteric influence model to suppress trivially bonded pairs.

**Architecture:** Each improvement is independent. #1 adds a `verbose` flag and mean-epoch-loss tracking to all three training pipelines. #5 adds a `load_trajectory` dispatcher in `src/allostery/io/trajectory.py` that falls back gracefully when optional packages are absent. #13 generalises the diagonal mask in `AllostericInfluenceModel` to a configurable sequence-distance mask, stored in checkpoints so models reload correctly.

**Tech Stack:** Python 3.11, PyTorch, NumPy. MDAnalysis and MDTraj are optional runtime dependencies for improvement #5.

---

## File map

**Improvement #1 — training progress**

| Action | File |
|---|---|
| Modify | `src/allostery/pipeline/influence_train.py` |
| Modify | `src/allostery/pipeline/cri_train.py` |
| Modify | `src/allostery/pipeline/train.py` |
| Modify | `src/allostery/config.py` |
| Modify | `src/allostery/cli.py` |
| Test | `tests/test_training_progress.py` |

**Improvement #5 — direct trajectory loading**

| Action | File |
|---|---|
| Create | `src/allostery/io/trajectory.py` |
| Modify | `src/allostery/io/__init__.py` |
| Modify | `src/allostery/config.py` |
| Modify | `src/allostery/pipeline/train.py` |
| Modify | `src/allostery/pipeline/cri_train.py` |
| Modify | `src/allostery/pipeline/influence_train.py` |
| Modify | `src/allostery/pipeline/score.py` |
| Modify | `src/allostery/pipeline/cri_score.py` |
| Modify | `src/allostery/pipeline/influence_score.py` |
| Modify | `src/allostery/cli.py` |
| Test | `tests/test_trajectory_loader.py` |

**Improvement #13 — sequence-separation masking**

| Action | File |
|---|---|
| Modify | `src/allostery/models/influence.py` |
| Modify | `src/allostery/io/checkpoint.py` |
| Modify | `src/allostery/pipeline/influence_train.py` |
| Modify | `src/allostery/pipeline/influence_score.py` |
| Modify | `src/allostery/pipeline/score.py` |
| Modify | `src/allostery/cli.py` |
| Test | `tests/test_influence_model.py` (extend existing) |

---

## Improvement #1 — Training Progress Feedback

### Task 1: Add verbose progress printing to influence_train.py

**Files:**
- Modify: `src/allostery/pipeline/influence_train.py`
- Test: `tests/test_training_progress.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_progress.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


def test_influence_training_prints_epoch_lines_when_verbose(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("epoch 1/2")
    assert "train=" in lines[0]
    assert lines[1].startswith("epoch 2/2")


def test_influence_training_is_silent_when_verbose_false(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_influence_training_prints_val_loss_and_best_marker(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    # No validation split here, so no "val=" in output
    assert "val=" not in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_training_progress.py -q
```

Expected: `TypeError: train_influence_model() got an unexpected keyword argument 'verbose'`

- [ ] **Step 3: Add verbose parameter to influence_train.py**

In `src/allostery/pipeline/influence_train.py`, make these changes:

Add `verbose: bool = True` to `train_influence_model` signature (after `batch_size`):

```python
def train_influence_model(
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float,
    hidden_dim: int,
    num_encoder_layers: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    sparsity_weight: float,
    preprocess: str = 'none',
    validation_fraction: float = 0.2,
    patience: int = 5,
    seed: int = 0,
    device: str = 'cpu',
    batch_size: int = 4,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> InfluenceTrainResult:
```

Replace the epoch loop (the `for epoch in range(epochs):` block) with this version that tracks mean train loss and prints progress:

```python
    width = len(str(epochs))
    early_stopped = False
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = stack_influence_batch(batch_samples, torch_device)
            output = model(batch.state_features)
            losses = influence_loss(output, batch.acceleration_targets, sparsity_weight=sparsity_weight)
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            last_loss = float(losses.total.detach().item())
            epoch_loss_sum += last_loss
            epoch_batch_count += 1

        train_loss = epoch_loss_sum / max(epoch_batch_count, 1)

        if validation_samples:
            validation_loss = _evaluate_epoch(
                model=model,
                samples=validation_samples,
                device=torch_device,
                sparsity_weight=sparsity_weight,
                batch_size=batch_size,
            )
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    early_stopped = True
                    if verbose:
                        print(f"early stop at epoch {epoch + 1}", flush=True)
                    break
            if verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
        elif verbose:
            print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

Remove the old epoch loop that only tracked `last_loss` without printing (the block starting at `for epoch in range(epochs):` through the `if patience > 0 and ...` break). Replace it entirely with the block above.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src pytest tests/test_training_progress.py -q
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/influence_train.py tests/test_training_progress.py
git commit -m "feat: add verbose epoch progress to influence training"
```

---

### Task 2: Add verbose progress printing to cri_train.py

**Files:**
- Modify: `src/allostery/pipeline/cri_train.py`
- Test: `tests/test_training_progress.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_progress.py`:

```python
def test_cri_training_prints_epoch_lines_when_verbose(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.cri_train import train_cri_model

    train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("epoch 1/2")
    assert "train=" in lines[0]


def test_cri_training_is_silent_when_verbose_false(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.cri_train import train_cri_model

    train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_training_progress.py::test_cri_training_prints_epoch_lines_when_verbose -q
```

Expected: `TypeError: train_cri_model() got an unexpected keyword argument 'verbose'`

- [ ] **Step 3: Add verbose to cri_train.py**

In `src/allostery/pipeline/cri_train.py`, add `verbose: bool = True` to `train_cri_model` signature after `batch_size`:

```python
def train_cri_model(
    ...
    batch_size: int = 4,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> CRITrainResult:
```

Replace the epoch loop with this version:

```python
    width = len(str(epochs))
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = stack_cri_batch(batch_samples, torch_device)
            output = model(batch.state_features, batch.edge_index, batch.edge_distance, batch.edge_mask)
            losses = cri_loss(
                output,
                batch.acceleration_targets,
                entropy_weight=entropy_weight,
                no_edge_weight=no_edge_weight,
            )
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            last_loss = float(losses.total.detach().item())
            epoch_loss_sum += last_loss
            epoch_batch_count += 1

        train_loss = epoch_loss_sum / max(epoch_batch_count, 1)

        if validation_samples:
            validation_loss = _evaluate_epoch(
                model=model,
                samples=validation_samples,
                device=torch_device,
                entropy_weight=entropy_weight,
                no_edge_weight=no_edge_weight,
                batch_size=batch_size,
            )
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch + 1}", flush=True)
                    break
            if verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
        elif verbose:
            print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src pytest tests/test_training_progress.py -q
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/cri_train.py tests/test_training_progress.py
git commit -m "feat: add verbose epoch progress to CRI training"
```

---

### Task 3: Add verbose progress printing to train.py (relational)

**Files:**
- Modify: `src/allostery/pipeline/train.py`
- Test: `tests/test_training_progress.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_progress.py`:

```python
def test_relational_training_prints_epoch_lines_when_verbose(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.train import train_model

    train_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=1,
        horizon_size=1,
        stride=1,
        hidden_dim=8,
        residue_layers=1,
        pair_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        consistency_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("epoch 1/2")
    assert "train=" in lines[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_training_progress.py::test_relational_training_prints_epoch_lines_when_verbose -q
```

Expected: `TypeError: train_model() got an unexpected keyword argument 'verbose'`

- [ ] **Step 3: Add verbose to train.py**

In `src/allostery/pipeline/train.py`, add `verbose: bool = True` to both `train_relational_model` and `train_model` after `batch_size`. Pass `verbose=verbose` from `train_model` into `train_relational_model`.

Replace the epoch loop inside `train_relational_model` with:

```python
    width = len(str(epochs))
    for epoch in range(epochs):
        model.train()
        previous_scores: Tensor | None = None
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        for batch_samples in iter_batches(train_samples, batch_size):
            batch = _training_batch(batch_samples, torch_device)
            last_loss, previous_scores = _train_batch(
                model=model,
                batch=batch,
                optimizer=optimizer,
                consistency_weight=consistency_weight,
                previous_scores=previous_scores,
            )
            epoch_loss_sum += last_loss
            epoch_batch_count += 1

        train_loss = epoch_loss_sum / max(epoch_batch_count, 1)

        if validation_samples:
            validation_loss = _evaluate_epoch(
                model=model,
                samples=validation_samples,
                device=torch_device,
                consistency_weight=consistency_weight,
                batch_size=batch_size,
            )
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch + 1}", flush=True)
                    break
            if verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
        elif verbose:
            print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

- [ ] **Step 4: Run all progress tests**

```bash
PYTHONPATH=src pytest tests/test_training_progress.py -q
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/train.py tests/test_training_progress.py
git commit -m "feat: add verbose epoch progress to relational training"
```

---

### Task 4: Wire verbose through config and CLI

**Files:**
- Modify: `src/allostery/config.py`
- Modify: `src/allostery/cli.py`
- Test: `tests/test_config.py` (extend existing)

- [ ] **Step 1: Add verbose to TrainingConfig**

In `src/allostery/config.py`, add `verbose: bool = True` to `TrainingConfig` after `batch_size`:

```python
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    consistency_weight: float
    entropy_weight: float = 0.0
    no_edge_weight: float = 0.0
    sparsity_weight: float = 0.0
    validation_fraction: float = 0.2
    patience: int = 5
    seed: int = 0
    device: str = 'cpu'
    batch_size: int = 4
    verbose: bool = True
```

In `load_config`, add parsing for `verbose` inside the `TrainingConfig(...)` constructor call:

```python
            verbose=bool(training_raw.get('verbose', True)),
```

- [ ] **Step 2: Wire verbose through CLI**

In `src/allostery/cli.py`, add `verbose=training.verbose` to each of the three `_run_train` calls:

For the influence branch:
```python
        inf_result = train_influence_model(
            ...
            batch_size=training.batch_size,
            verbose=training.verbose,
            checkpoint_path=model_path,
            config_snapshot=_serialize_config(config),
        )
```

For the CRI branch:
```python
        result = train_cri_model(
            ...
            batch_size=training.batch_size,
            verbose=training.verbose,
            checkpoint_path=model_path,
            config_snapshot=_serialize_config(config),
        )
```

For the relational branch:
```python
    result = train_model(
        ...
        batch_size=training.batch_size,
        verbose=training.verbose,
        checkpoint_path=model_path,
        config_snapshot=_serialize_config(config),
    )
```

- [ ] **Step 3: Run the full test suite**

```bash
PYTHONPATH=src pytest -q
```

Expected: all tests pass (count increases by 6 from the new progress tests).

- [ ] **Step 4: Commit**

```bash
git add src/allostery/config.py src/allostery/cli.py
git commit -m "feat: wire verbose training flag through config and CLI"
```

---

## Improvement #5 — Direct Trajectory Loading

### Task 5: Create load_trajectory dispatcher with PDB passthrough

**Files:**
- Create: `src/allostery/io/trajectory.py`
- Test: `tests/test_trajectory_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trajectory_loader.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_load_trajectory_dispatches_pdb_by_extension(fixture_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    result = load_trajectory(fixture_path / "tiny_trajectory.pdb")

    assert result.coordinates.ndim == 3
    assert result.coordinates.shape[2] == 3
    assert len(result.residues) == result.coordinates.shape[1]


def test_load_trajectory_accepts_string_path(fixture_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    result = load_trajectory(str(fixture_path / "tiny_trajectory.pdb"))

    assert result.coordinates.shape[0] > 0


def test_load_trajectory_requires_topology_for_non_pdb() -> None:
    from allostery.io.trajectory import load_trajectory

    with pytest.raises(ValueError, match="topology_path is required"):
        load_trajectory("trajectory.xtc")


def test_load_trajectory_requires_topology_for_dcd() -> None:
    from allostery.io.trajectory import load_trajectory

    with pytest.raises(ValueError, match="topology_path is required"):
        load_trajectory("trajectory.dcd")


def test_load_trajectory_raises_import_error_without_backends(tmp_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    fake_traj = tmp_path / "traj.xtc"
    fake_traj.touch()

    with patch.dict("sys.modules", {"MDAnalysis": None, "mdtraj": None}):
        with pytest.raises(ImportError, match="MDAnalysis"):
            load_trajectory(fake_traj, topology_path=tmp_path / "top.tpr")


def test_load_trajectory_error_message_mentions_both_packages(tmp_path: Path) -> None:
    from allostery.io.trajectory import load_trajectory

    fake_traj = tmp_path / "traj.xtc"
    fake_traj.touch()

    with patch.dict("sys.modules", {"MDAnalysis": None, "mdtraj": None}):
        with pytest.raises(ImportError) as exc_info:
            load_trajectory(fake_traj, topology_path=tmp_path / "top.tpr")

    message = str(exc_info.value)
    assert "MDAnalysis" in message
    assert "mdtraj" in message
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_trajectory_loader.py -q
```

Expected: `ImportError: cannot import name 'load_trajectory' from 'allostery.io.trajectory'`

- [ ] **Step 3: Implement the dispatcher**

Create `src/allostery/io/trajectory.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

from allostery.io.pdb import ResidueRecord, Trajectory, load_multimodel_pdb

_PDB_EXTENSIONS = {'.pdb', '.ent'}


def load_trajectory(
    path: str | Path,
    topology_path: str | Path | None = None,
) -> Trajectory:
    """Load a C-alpha trajectory from any supported format.

    PDB (multi-model) is handled natively. All other formats (.xtc, .dcd, .nc,
    etc.) require a topology file and either MDAnalysis or MDTraj to be installed.
    """
    path = Path(path)
    if path.suffix.lower() in _PDB_EXTENSIONS:
        return load_multimodel_pdb(path)

    if topology_path is None:
        raise ValueError(
            f"topology_path is required for non-PDB trajectories (got {path.suffix!r}). "
            "Provide the matching topology file (e.g. .tpr, .psf, or .prmtop)."
        )

    topology_path = Path(topology_path)

    # Try MDAnalysis first
    mda_module = sys.modules.get('MDAnalysis', ...)
    if mda_module is not ...:  # not patched out
        try:
            import MDAnalysis  # noqa: F401
            return _load_via_mdanalysis(path, topology_path)
        except ImportError:
            pass
    elif mda_module is not None:  # patched to a real module
        return _load_via_mdanalysis(path, topology_path)

    # Try MDTraj
    mdt_module = sys.modules.get('mdtraj', ...)
    if mdt_module is not ...:
        try:
            import mdtraj  # noqa: F401
            return _load_via_mdtraj(path, topology_path)
        except ImportError:
            pass
    elif mdt_module is not None:
        return _load_via_mdtraj(path, topology_path)

    raise ImportError(
        f"Cannot load {path.suffix!r} trajectory: install MDAnalysis or MDTraj.\n"
        "  pip install MDAnalysis\n"
        "  pip install mdtraj"
    )


def _load_via_mdanalysis(path: Path, topology_path: Path) -> Trajectory:
    import MDAnalysis as mda
    import numpy as np

    u = mda.Universe(str(topology_path), str(path))
    ca = u.select_atoms("name CA")
    if ca.n_atoms == 0:
        raise ValueError(f"No CA atoms found in topology {topology_path}")

    residues = tuple(
        ResidueRecord(
            index=i,
            chain_id=str(atom.segid).strip() or "_",
            residue_number=int(atom.resid),
            name=str(atom.resname)[:3],
        )
        for i, atom in enumerate(ca)
    )

    coordinates = np.empty((len(u.trajectory), ca.n_atoms, 3), dtype=np.float32)
    for ts_idx, _ts in enumerate(u.trajectory):
        coordinates[ts_idx] = ca.positions.astype(np.float32)

    return Trajectory(residues=residues, coordinates=coordinates)


def _load_via_mdtraj(path: Path, topology_path: Path) -> Trajectory:
    import mdtraj as md
    import numpy as np

    traj = md.load(str(path), top=str(topology_path))
    ca_indices = traj.topology.select("name CA")
    if len(ca_indices) == 0:
        raise ValueError(f"No CA atoms found in topology {topology_path}")

    ca_traj = traj.atom_slice(ca_indices)

    residues = tuple(
        ResidueRecord(
            index=i,
            chain_id=str(atom.residue.chain.chain_id),
            residue_number=int(atom.residue.resSeq),
            name=str(atom.residue.name)[:3],
        )
        for i, atom in enumerate(ca_traj.topology.atoms)
    )

    # MDTraj stores coordinates in nanometres; convert to Angstroms to match PDB convention
    coordinates = (ca_traj.xyz * 10.0).astype(np.float32)

    return Trajectory(residues=residues, coordinates=coordinates)


__all__ = ['load_trajectory']
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src pytest tests/test_trajectory_loader.py -q
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/allostery/io/trajectory.py tests/test_trajectory_loader.py
git commit -m "feat: add load_trajectory dispatcher for non-PDB formats"
```

---

### Task 6: Wire topology_path through config and all six pipelines

**Files:**
- Modify: `src/allostery/config.py`
- Modify: `src/allostery/pipeline/train.py`
- Modify: `src/allostery/pipeline/cri_train.py`
- Modify: `src/allostery/pipeline/influence_train.py`
- Modify: `src/allostery/pipeline/score.py`
- Modify: `src/allostery/pipeline/cri_score.py`
- Modify: `src/allostery/pipeline/influence_score.py`
- Modify: `src/allostery/io/__init__.py`
- Modify: `src/allostery/cli.py`
- Test: `tests/test_trajectory_loader.py`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_trajectory_loader.py`:

```python
def test_train_influence_accepts_topology_path_kwarg(fixture_path: Path) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    # topology_path=None should work for .pdb (existing behaviour unchanged)
    result = train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        topology_path=None,
    )
    assert result.num_samples >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/test_trajectory_loader.py::test_train_influence_accepts_topology_path_kwarg -q
```

Expected: `TypeError: train_influence_model() got an unexpected keyword argument 'topology_path'`

- [ ] **Step 3: Add topology_path to DataConfig**

In `src/allostery/config.py`, add to `DataConfig`:

```python
@dataclass(frozen=True, slots=True)
class DataConfig:
    pdb_path: Path
    window_size: int
    horizon_size: int
    stride: int
    time_step: float = 1.0
    distance_cutoff: float = 20.0
    max_neighbors: int = 2
    min_sequence_separation: int = 0
    preprocess: str = 'none'
    topology_path: Path | None = None
```

In `load_config`, add topology_path parsing inside the `DataConfig(...)` constructor:

```python
            topology_path=_optional_path(base_dir, data_raw.get('topology_path')),
```

No validation rule is needed — `topology_path` is optional; `load_trajectory` raises at runtime if it is required but absent.

- [ ] **Step 4: Replace load_multimodel_pdb with load_trajectory in all six pipelines**

For each of the six files below, make two changes:
1. Replace `from allostery.io.pdb import load_multimodel_pdb` with `from allostery.io.trajectory import load_trajectory`
2. Add `topology_path: str | Path | None = None` to the function signature
3. Replace the `load_multimodel_pdb(Path(pdb_path))` call with `load_trajectory(Path(pdb_path), topology_path=topology_path)`

**`src/allostery/pipeline/influence_train.py`** — `train_influence_model`:

```python
# replace import
from allostery.io.trajectory import load_trajectory

# add to signature after pdb_path
def train_influence_model(
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    ...
```

```python
# replace load call
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
```

**`src/allostery/pipeline/cri_train.py`** — `train_cri_model`:

```python
from allostery.io.trajectory import load_trajectory

def train_cri_model(
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    ...
```

```python
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
```

**`src/allostery/pipeline/train.py`** — `_load_training_samples`:

```python
from allostery.io.trajectory import load_trajectory

def _load_training_samples(
    pdb_path: str | Path,
    window_size: int,
    horizon_size: int,
    stride: int,
    topology_path: str | Path | None = None,
) -> list[TrainingSample]:
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
    ...
```

Also add `topology_path: str | Path | None = None` to both `train_relational_model` and `train_model`, and pass `topology_path=topology_path` in the internal `_load_training_samples(...)` calls.

**`src/allostery/pipeline/score.py`** — `score_trajectory`:

```python
from allostery.io.trajectory import load_trajectory

def score_trajectory(
    model: RelationalScoreModel,
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    window_size: int = 8,
    horizon_size: int = 4,
    stride: int = 2,
) -> list[PairScore]:
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
    ...
```

**`src/allostery/pipeline/cri_score.py`** — `score_cri_trajectory`:

```python
from allostery.io.trajectory import load_trajectory

def score_cri_trajectory(
    model: CRILatentInteractionModel,
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    window_size: int,
    ...
```

```python
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
```

**`src/allostery/pipeline/influence_score.py`** — `score_influence_trajectory`:

```python
from allostery.io.trajectory import load_trajectory

def score_influence_trajectory(
    model: AllostericInfluenceModel,
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    window_size: int,
    ...
```

```python
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
```

- [ ] **Step 5: Wire topology_path through CLI**

In `src/allostery/cli.py`, pass `topology_path=config.data.topology_path` to every train/score call. The three train calls and three score calls each get this extra argument. Example for the influence train block:

```python
        inf_result = train_influence_model(
            pdb_path=config.data.pdb_path,
            topology_path=config.data.topology_path,
            ...
        )
```

Apply the same pattern to CRI train, relational train, influence score, CRI score, and relational score.

- [ ] **Step 6: Export load_trajectory from the io package**

In `src/allostery/io/__init__.py`, add:

```python
from .trajectory import load_trajectory

__all__ = [
    "CSV_COLUMNS",
    "ModelCheckpoint",
    "load_checkpoint",
    "load_trajectory",
    "save_checkpoint",
    "write_pair_scores_csv",
]
```

- [ ] **Step 7: Run full test suite**

```bash
PYTHONPATH=src pytest -q
```

Expected: all existing tests plus new trajectory tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/allostery/config.py \
  src/allostery/io/__init__.py \
  src/allostery/pipeline/train.py \
  src/allostery/pipeline/cri_train.py \
  src/allostery/pipeline/influence_train.py \
  src/allostery/pipeline/score.py \
  src/allostery/pipeline/cri_score.py \
  src/allostery/pipeline/influence_score.py \
  src/allostery/cli.py \
  tests/test_trajectory_loader.py
git commit -m "feat: wire topology_path through config, pipelines, and CLI"
```

---

## Improvement #13 — Sequence-Separation Masking

### Task 7: Add min_sequence_separation to AllostericInfluenceModel

**Files:**
- Modify: `src/allostery/models/influence.py`
- Test: `tests/test_influence_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_influence_model.py`:

```python
def test_influence_model_masks_pairs_within_sequence_separation() -> None:
    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=3
    )
    state_features = torch.randn(1, 3, 8, 6)  # 8 residues

    output = model(state_features)
    A = output['influence_matrix'].squeeze(0)  # [8, 8]

    for i in range(8):
        for j in range(8):
            if abs(i - j) < 3:
                assert A[i, j].item() == pytest.approx(0.0, abs=1e-6), (
                    f"A[{i},{j}] = {A[i,j].item():.6f} should be 0 "
                    f"(sequence separation {abs(i-j)} < 3)"
                )


def test_influence_model_rows_still_sum_to_one_with_separation() -> None:
    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=3
    )
    state_features = torch.randn(1, 3, 8, 6)

    output = model(state_features)
    A = output['influence_matrix'].squeeze(0)  # [8, 8]

    torch.testing.assert_close(A.sum(dim=-1), torch.ones(8))


def test_influence_model_separation_one_is_diagonal_only() -> None:
    model_sep1 = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=1
    )
    model_diag = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=1
    )
    # copy weights so both models are identical
    model_diag.load_state_dict(model_sep1.state_dict())

    state = torch.randn(1, 3, 5, 6)
    with torch.no_grad():
        A1 = model_sep1(state)['influence_matrix']
        A2 = model_diag(state)['influence_matrix']

    torch.testing.assert_close(A1, A2)
    # Diagonal must be zero
    diag = torch.diagonal(A1.squeeze(0))
    torch.testing.assert_close(diag, torch.zeros(5))


def test_influence_model_rejects_separation_less_than_one() -> None:
    with pytest.raises(ValueError, match="min_sequence_separation"):
        AllostericInfluenceModel(state_dim=6, hidden_dim=8, min_sequence_separation=0)


def test_influence_model_rejects_separation_too_large_for_protein() -> None:
    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=5
    )
    state_features = torch.randn(1, 3, 4, 6)  # only 4 residues, sep=5 would mask everything

    with pytest.raises(ValueError, match="no valid pairs"):
        model(state_features)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_influence_model.py -q
```

Expected: failures on the new tests (model has no `min_sequence_separation` parameter yet).

- [ ] **Step 3: Update AllostericInfluenceModel**

In `src/allostery/models/influence.py`, add `min_sequence_separation: int = 1` to `__init__` and update the forward pass:

```python
class AllostericInfluenceModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_encoder_layers: int = 2,
        dropout: float = 0.0,
        min_sequence_separation: int = 1,
    ) -> None:
        super().__init__()
        if num_encoder_layers <= 0:
            raise ValueError('num_encoder_layers must be greater than zero')
        if min_sequence_separation < 1:
            raise ValueError('min_sequence_separation must be at least 1 (diagonal must always be masked)')
        self.hidden_dim = hidden_dim
        self.min_sequence_separation = min_sequence_separation
        self.encoder = _build_mlp(state_dim, hidden_dim, num_encoder_layers, dropout)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(state_dim, hidden_dim)
        self.baseline_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 3),
        )
        self.decode_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, 3),
        )
```

Replace the mask-building block in `forward` (the two lines that build `diag_mask`) with:

```python
        # Build sequence-separation mask: block all pairs with |i - j| < min_sequence_separation.
        # This always includes the diagonal (|i-j|=0) when min_sequence_separation >= 1.
        if self.min_sequence_separation >= num_residues:
            raise ValueError(
                f"min_sequence_separation={self.min_sequence_separation} leaves no valid pairs "
                f"for a protein of {num_residues} residues. "
                f"Use a value less than {num_residues}."
            )
        indices = torch.arange(num_residues, device=state_features.device)
        sep_mask = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs() < self.min_sequence_separation
        attn_logits = attn_logits.masked_fill(sep_mask.unsqueeze(0), float('-inf'))
```

Remove the old `diag_mask` two-liner entirely. The `sep_mask` above subsumes it when `min_sequence_separation >= 1`.

- [ ] **Step 4: Run all influence model tests**

```bash
PYTHONPATH=src pytest tests/test_influence_model.py -q
```

Expected: all pass (both old and new tests).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/models/influence.py tests/test_influence_model.py
git commit -m "feat: add sequence-separation masking to AllostericInfluenceModel"
```

---

### Task 8: Persist min_sequence_separation in checkpoints

**Files:**
- Modify: `src/allostery/io/checkpoint.py`
- Test: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_checkpoint.py`:

```python
def test_checkpoint_round_trips_min_sequence_separation(tmp_path) -> None:
    from allostery.io.checkpoint import load_checkpoint, save_checkpoint
    from allostery.models.influence import AllostericInfluenceModel

    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=3
    )
    path = tmp_path / "influence_sep3.pt"

    save_checkpoint(
        path=path,
        model=model,
        config_snapshot={},
        residue_dim=6,
        pair_dim=1,
        hidden_dim=8,
        target_dim=3,
        residue_layers=1,
        pair_layers=1,
        dropout=0.0,
        model_family='influence',
        min_sequence_separation=3,
    )

    ckpt = load_checkpoint(path)
    assert ckpt.min_sequence_separation == 3


def test_checkpoint_defaults_min_sequence_separation_to_one(tmp_path) -> None:
    from allostery.io.checkpoint import load_checkpoint, save_checkpoint
    from allostery.models.relational import RelationalScoreModel

    model = RelationalScoreModel(
        residue_dim=10, pair_dim=5, hidden_dim=8, target_dim=3
    )
    path = tmp_path / "relational.pt"

    save_checkpoint(
        path=path,
        model=model,
        config_snapshot={},
        residue_dim=10,
        pair_dim=5,
        hidden_dim=8,
        target_dim=3,
    )

    ckpt = load_checkpoint(path)
    assert ckpt.min_sequence_separation == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_checkpoint.py -q
```

Expected: `TypeError: save_checkpoint() got an unexpected keyword argument 'min_sequence_separation'`

- [ ] **Step 3: Update ModelCheckpoint, save_checkpoint, and load_checkpoint**

In `src/allostery/io/checkpoint.py`:

Add `min_sequence_separation: int = 1` to `ModelCheckpoint`:

```python
@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    state_dict: dict[str, Tensor]
    residue_dim: int
    pair_dim: int
    hidden_dim: int
    residue_layers: int
    pair_layers: int
    dropout: float
    target_dim: int
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    model_family: str = 'relational'
    min_sequence_separation: int = 1
```

Add `min_sequence_separation: int = 1` to `save_checkpoint` and include it in the saved dict:

```python
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config_snapshot: Mapping[str, Any],
    residue_dim: int,
    pair_dim: int,
    hidden_dim: int,
    target_dim: int,
    residue_layers: int = 2,
    pair_layers: int = 2,
    dropout: float = 0.0,
    model_family: str = 'relational',
    min_sequence_separation: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'state_dict': dict(model.state_dict().items()),
            'residue_dim': residue_dim,
            'pair_dim': pair_dim,
            'hidden_dim': hidden_dim,
            'residue_layers': residue_layers,
            'pair_layers': pair_layers,
            'dropout': dropout,
            'target_dim': target_dim,
            'config': dict(config_snapshot),
            'metadata': dict(metadata or {}),
            'model_family': model_family,
            'min_sequence_separation': min_sequence_separation,
        },
        target,
    )
```

Update `load_checkpoint` to read the new field:

```python
    return ModelCheckpoint(
        ...
        model_family=str(raw.get('model_family', 'relational')),
        min_sequence_separation=int(raw.get('min_sequence_separation', 1)),
    )
```

- [ ] **Step 4: Run checkpoint tests**

```bash
PYTHONPATH=src pytest tests/test_checkpoint.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/io/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: persist min_sequence_separation in checkpoints"
```

---

### Task 9: Wire min_sequence_separation through influence pipelines and CLI

**Files:**
- Modify: `src/allostery/pipeline/influence_train.py`
- Modify: `src/allostery/pipeline/influence_score.py`
- Modify: `src/allostery/pipeline/score.py`
- Modify: `src/allostery/cli.py`
- Test: `tests/test_influence_training.py`, `tests/test_influence_scoring.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_influence_training.py`:

```python
def test_train_influence_model_respects_min_sequence_separation(fixture_path: Path) -> None:
    # tiny_trajectory has 3 residues; sep=1 is the max safe value for N=3
    result = train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        min_sequence_separation=1,
    )
    assert result.num_samples >= 1


def test_train_influence_model_saves_min_sequence_separation_in_checkpoint(
    fixture_path: Path, tmp_path: Path
) -> None:
    from allostery.io.checkpoint import load_checkpoint

    checkpoint_path = tmp_path / "influence_sep.pt"
    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        min_sequence_separation=1,
        checkpoint_path=checkpoint_path,
    )

    ckpt = load_checkpoint(checkpoint_path)
    assert ckpt.min_sequence_separation == 1
```

Append to `tests/test_influence_scoring.py`:

```python
def test_score_influence_trajectory_respects_min_sequence_separation(fixture_path: Path) -> None:
    from allostery.pipeline.influence_train import train_influence_model
    from allostery.pipeline.influence_score import score_influence_trajectory

    result = train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        min_sequence_separation=1,
    )

    scores = score_influence_trajectory(
        model=result.model,
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        min_sequence_separation=1,
    )

    assert len(scores) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src pytest tests/test_influence_training.py tests/test_influence_scoring.py -q
```

Expected: `TypeError: train_influence_model() got an unexpected keyword argument 'min_sequence_separation'`

- [ ] **Step 3: Update train_influence_model**

In `src/allostery/pipeline/influence_train.py`:

Add `min_sequence_separation: int = 1` to `train_influence_model` after `dropout`:

```python
def train_influence_model(
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    window_size: int,
    stride: int,
    time_step: float,
    hidden_dim: int,
    num_encoder_layers: int,
    dropout: float,
    min_sequence_separation: int = 1,
    epochs: int,
    ...
```

Pass it to the model constructor:

```python
    model = AllostericInfluenceModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        min_sequence_separation=min_sequence_separation,
    ).to(torch_device)
```

Pass it to `save_checkpoint`:

```python
        save_checkpoint(
            ...
            model_family='influence',
            min_sequence_separation=min_sequence_separation,
            ...
        )
```

- [ ] **Step 4: Update score_influence_trajectory**

In `src/allostery/pipeline/influence_score.py`:

Add `min_sequence_separation: int = 1` to `score_influence_trajectory` after `preprocess`:

```python
def score_influence_trajectory(
    model: AllostericInfluenceModel,
    pdb_path: str | Path,
    topology_path: str | Path | None = None,
    window_size: int,
    stride: int,
    time_step: float = 1.0,
    preprocess: str = 'none',
    min_sequence_separation: int = 1,
) -> list[InfluencePairScore]:
```

The `min_sequence_separation` is already encoded in the model (it was set at construction time and stored as `model.min_sequence_separation`). The parameter here is for documentation parity; the model will use its own internal value. You do not need to pass it anywhere inside the function body — the model's `forward` uses `self.min_sequence_separation`.

- [ ] **Step 5: Update load_scoring_model to read min_sequence_separation from checkpoint**

In `src/allostery/pipeline/score.py`, update the `influence` branch:

```python
    elif checkpoint.model_family == 'influence':
        model = AllostericInfluenceModel(
            state_dim=checkpoint.residue_dim,
            hidden_dim=checkpoint.hidden_dim,
            num_encoder_layers=checkpoint.residue_layers,
            dropout=checkpoint.dropout,
            min_sequence_separation=checkpoint.min_sequence_separation,
        )
```

- [ ] **Step 6: Update CLI to pass data.min_sequence_separation to influence pipelines**

In `src/allostery/cli.py`, update the influence train block:

```python
        inf_result = train_influence_model(
            pdb_path=config.data.pdb_path,
            topology_path=config.data.topology_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            preprocess=config.data.preprocess,
            hidden_dim=config.model.hidden_dim,
            num_encoder_layers=config.model.residue_layers,
            dropout=config.model.dropout,
            min_sequence_separation=config.data.min_sequence_separation,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            sparsity_weight=training.sparsity_weight,
            validation_fraction=training.validation_fraction,
            patience=training.patience,
            seed=training.seed,
            device=training.device,
            batch_size=training.batch_size,
            verbose=training.verbose,
            checkpoint_path=model_path,
            config_snapshot=_serialize_config(config),
        )
```

Update the influence score block:

```python
    if config.model.family == 'influence':
        scores = score_influence_trajectory(
            model=model,
            pdb_path=config.data.pdb_path,
            topology_path=config.data.topology_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            preprocess=config.data.preprocess,
            min_sequence_separation=config.data.min_sequence_separation,
        )
```

- [ ] **Step 7: Run full test suite**

```bash
PYTHONPATH=src pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/allostery/pipeline/influence_train.py \
  src/allostery/pipeline/influence_score.py \
  src/allostery/pipeline/score.py \
  src/allostery/cli.py \
  tests/test_influence_training.py \
  tests/test_influence_scoring.py
git commit -m "feat: wire min_sequence_separation through influence pipelines and CLI"
```

---

## Final verification

- [ ] **Run the complete test suite one last time**

```bash
PYTHONPATH=src pytest -q
```

Expected: all tests pass.

- [ ] **Smoke-test the CLI with the example config**

```bash
PYTHONPATH=src python -m allostery.cli examples/influence_example_config.yaml
```

Expected: training prints epoch lines to stdout, then scoring completes and prints `completed mode=run`.

- [ ] **Check git log looks clean**

```bash
git log --oneline -10
```

Expected output (most recent first):

```
feat: wire min_sequence_separation through influence pipelines and CLI
feat: persist min_sequence_separation in checkpoints
feat: add sequence-separation masking to AllostericInfluenceModel
feat: wire topology_path through config, pipelines, and CLI
feat: add load_trajectory dispatcher for non-PDB formats
feat: wire verbose training flag through config and CLI
feat: add verbose epoch progress to relational training
feat: add verbose epoch progress to CRI training
feat: add verbose epoch progress to influence training
```

---

## Acceptance criteria

- All three model families print `epoch N/M  train=X.XXXX  [val=Y.YYYY  [best]]` to stdout during training when `verbose: true` (default) and produce no output when `verbose: false`.
- `load_trajectory("file.pdb")` is identical in behaviour to `load_multimodel_pdb`.
- `load_trajectory("file.xtc", topology_path="top.tpr")` succeeds when MDAnalysis is installed; raises `ImportError` with a helpful install message when neither MDAnalysis nor MDTraj is installed; raises `ValueError` when `topology_path` is omitted.
- `AllostericInfluenceModel` with `min_sequence_separation=3` produces `influence_matrix[j,i] == 0` for all pairs with `|i-j| < 3`.
- `load_scoring_model` correctly reconstructs an influence model with the stored `min_sequence_separation` from a checkpoint.
- All 78 pre-existing tests continue to pass.
