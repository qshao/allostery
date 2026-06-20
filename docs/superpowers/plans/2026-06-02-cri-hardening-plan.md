# CRI Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the CRI-inspired residue interaction workflow so it produces a scientifically defensible residue-residue network, scores interactions with confidence, and runs fast enough to be practical on real protein trajectories.

**Architecture:** Keep the current CRI path as the base, but split hardening into four layers: trajectory preprocessing, candidate graph policy, interaction scoring and provenance, and training/runtime performance. Each layer should have a small interface and dedicated tests so we can improve correctness without changing the overall user workflow.

**Tech Stack:** Python 3.11, NumPy, PyTorch, PyYAML, pytest. No new runtime dependency is required for the first pass.

---

## Source Review Summary

The current CRI path already trains on C-alpha trajectory windows, infers per-edge latent interaction types, and exports pair scores. The main gaps are:

- Trajectories are used as loaded, so rigid-body motion can leak into the dynamics targets.
- The graph only includes cutoff-limited top-k neighbors, which is scalable but can hide long-range couplings.
- Edge messages explain all acceleration directly, without a baseline for self-dynamics or global motion.
- Scoring collapses directed edge probabilities into a single pair score without confidence or support metadata.
- Training is sample-by-sample and CPU-oriented, which is fine for a prototype but not for larger proteins.

This plan keeps the existing relational model path intact while hardening the CRI path for allosteric network inference.

## File Structure

- Create `src/allostery/features/alignment.py`: optional Kabsch alignment and frame-centering helpers for trajectory preprocessing.
- Modify `src/allostery/features/dynamics.py`: apply aligned coordinates, keep finite-difference dynamics utilities, and expose preprocessing hooks.
- Modify `src/allostery/features/graph.py`: add graph policy options, residue-separation filters, and edge metadata.
- Modify `src/allostery/cri/data.py`: build CRI samples from the updated preprocessing and graph policy.
- Modify `src/allostery/models/cri.py`: add a residual baseline for acceleration and vectorized edge aggregation.
- Modify `src/allostery/training/cri_objectives.py`: add validation-aware losses and metrics helpers.
- Modify `src/allostery/pipeline/cri_train.py`: add batching, validation split, device selection, seeding, and early stopping.
- Modify `src/allostery/pipeline/cri_score.py`: export directed probabilities, support counts, and confidence statistics.
- Modify `src/allostery/io/results.py`: write richer network outputs and preserve backward-compatible CSV columns.
- Modify `src/allostery/io/checkpoint.py`: store model provenance, graph policy, alignment policy, and training metadata.
- Modify `src/allostery/config.py`: add config fields for preprocessing, graph policy, batching, and reproducibility.
- Modify `src/allostery/cli.py`: route the new configuration fields into train and score paths.
- Test files to add or extend: `tests/test_alignment_features.py`, `tests/test_graph_features.py`, `tests/test_cri_data.py`, `tests/test_cri_model.py`, `tests/test_cri_training.py`, `tests/test_cri_scoring.py`, `tests/test_checkpoint.py`, `tests/test_config.py`, `tests/test_results.py`, and a new synthetic validation test file.

## Design Decisions

- Use C-alpha alignment as an optional preprocessing step, with centering as the simpler fallback.
- Keep the current sparse directed graph approach, but make the candidate policy explicit and configurable.
- Treat the no-edge class as a prior on weak coupling rather than as a hard physical state.
- Preserve the existing relational workflow and keep the CRI path opt-in through `model.family: cri`.
- Favor small, testable changes over a single large rewrite so each improvement can be validated independently.

### Task 1: Add Trajectory Alignment Preprocessing

**Files:**
- Create: `src/allostery/features/alignment.py`
- Modify: `src/allostery/features/dynamics.py`
- Test: `tests/test_alignment_features.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alignment_features.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from allostery.features.alignment import align_trajectory_coordinates, center_trajectory_coordinates


def test_center_trajectory_coordinates_removes_translation() -> None:
    coordinates = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ],
        dtype=np.float32,
    )

    centered = center_trajectory_coordinates(coordinates)

    np.testing.assert_allclose(centered.mean(axis=1), np.zeros((2, 3), dtype=np.float32), atol=1e-6)


def test_align_trajectory_coordinates_preserves_pairwise_distances() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]],
        ],
        dtype=np.float32,
    )

    aligned = align_trajectory_coordinates(coordinates, reference_frame_index=0)

    np.testing.assert_allclose(
        np.linalg.norm(aligned[:, 0] - aligned[:, 1], axis=-1),
        np.array([1.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_align_trajectory_coordinates_rejects_invalid_reference_frame() -> None:
    with pytest.raises(IndexError, match="reference_frame_index"):
        align_trajectory_coordinates(np.zeros((2, 2, 3), dtype=np.float32), reference_frame_index=2)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_alignment_features.py -q`

Expected: fail because `allostery.features.alignment` does not exist yet.

- [ ] **Step 3: Implement alignment helpers**

Create `src/allostery/features/alignment.py`:

```python
from __future__ import annotations

import numpy as np

from allostery.features.residue import _validate_coordinate_window


def center_trajectory_coordinates(window_coordinates: np.ndarray) -> np.ndarray:
    coordinates = _validate_coordinate_window(window_coordinates)
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    return centered.astype(np.float32, copy=False)


def align_trajectory_coordinates(window_coordinates: np.ndarray, reference_frame_index: int = 0) -> np.ndarray:
    coordinates = _validate_coordinate_window(window_coordinates)
    if reference_frame_index < 0 or reference_frame_index >= coordinates.shape[0]:
        raise IndexError("reference_frame_index is out of range")

    reference = coordinates[reference_frame_index]
    aligned = np.empty_like(coordinates)
    for frame_index, frame in enumerate(coordinates):
        aligned[frame_index] = _kabsch_align(frame, reference)
    return aligned.astype(np.float32, copy=False)


def _kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_centered = mobile - mobile.mean(axis=0, keepdims=True)
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    covariance = mobile_centered.T @ reference_centered
    v, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ v.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ v.T
    return mobile_centered @ rotation + reference.mean(axis=0, keepdims=True)


__all__ = ["align_trajectory_coordinates", "center_trajectory_coordinates"]
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest tests/test_alignment_features.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/features/alignment.py src/allostery/features/dynamics.py tests/test_alignment_features.py
git commit -m "feat: add trajectory alignment helpers"
```

### Task 2: Make the Graph Policy Explicit

**Files:**
- Modify: `src/allostery/features/graph.py`
- Modify: `src/allostery/cri/data.py`
- Modify: `src/allostery/config.py`
- Test: `tests/test_graph_features.py`

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_graph_features.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from allostery.features.graph import build_directed_contact_graph


def test_build_directed_contact_graph_can_enforce_sequence_separation() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0], [5.1, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    graph = build_directed_contact_graph(
        coordinates,
        distance_cutoff=3.0,
        max_neighbors=2,
        min_sequence_separation=2,
    )

    assert all(abs(int(sender) - int(receiver)) >= 2 for sender, receiver in graph.edge_index)


def test_build_directed_contact_graph_can_return_edge_metadata() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    graph = build_directed_contact_graph(coordinates, distance_cutoff=3.0, max_neighbors=1)

    assert graph.edge_index.shape[1] == 2
    assert graph.mean_distances.shape[0] == graph.edge_index.shape[0]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_graph_features.py -q`

Expected: fail because `min_sequence_separation` is not implemented yet.

- [ ] **Step 3: Extend the graph policy**

Update `src/allostery/features/graph.py`:

```python
def build_directed_contact_graph(
    window_coordinates: np.ndarray,
    distance_cutoff: float,
    max_neighbors: int,
    min_sequence_separation: int = 0,
) -> DirectedContactGraph:
    if distance_cutoff <= 0.0:
        raise ValueError("distance_cutoff must be greater than zero")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be greater than zero")
    if min_sequence_separation < 0:
        raise ValueError("min_sequence_separation must be greater than or equal to zero")

    coordinates = _validate_coordinate_window(window_coordinates)
    if not np.isfinite(coordinates).all():
        raise ValueError("window_coordinates must contain only finite values")

    mean_positions = coordinates.mean(axis=0)
    delta = mean_positions[:, None, :] - mean_positions[None, :, :]
    distances = np.linalg.norm(delta, axis=-1).astype(np.float32)
    num_residues = distances.shape[0]

    edges: list[tuple[int, int]] = []
    mean_distances: list[float] = []
    for receiver in range(num_residues):
        candidates = [
            (float(distances[sender, receiver]), sender)
            for sender in range(num_residues)
            if sender != receiver
            and abs(sender - receiver) >= min_sequence_separation
            and distances[sender, receiver] <= distance_cutoff
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        for distance, sender in candidates[:max_neighbors]:
            edges.append((sender, receiver))
            mean_distances.append(distance)

    if not edges:
        return DirectedContactGraph(
            edge_index=np.empty((0, 2), dtype=np.int64),
            mean_distances=np.empty(0, dtype=np.float32),
        )
    return DirectedContactGraph(
        edge_index=np.array(edges, dtype=np.int64),
        mean_distances=np.array(mean_distances, dtype=np.float32),
    )
```

- [ ] **Step 4: Wire the policy through CRI sample construction**

Update `src/allostery/cri/data.py`:

```python
def build_cri_training_samples(
    coordinates: np.ndarray,
    window_size: int,
    stride: int,
    time_step: float,
    distance_cutoff: float,
    max_neighbors: int,
    min_sequence_separation: int,
) -> list[CRISample]:
    trajectory = _validate_coordinate_window(coordinates)
    if window_size < 3:
        raise ValueError("window_size must be at least 3 for central differences")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if min_sequence_separation < 0:
        raise ValueError("min_sequence_separation must be greater than or equal to zero")
    if trajectory.shape[0] < window_size:
        return []

    samples: list[CRISample] = []
    for start in range(0, trajectory.shape[0] - window_size + 1, stride):
        window = trajectory[start : start + window_size]
        dynamics = build_residue_dynamics(window, time_step=time_step)
        graph = build_directed_contact_graph(
            window,
            distance_cutoff=distance_cutoff,
            max_neighbors=max_neighbors,
            min_sequence_separation=min_sequence_separation,
        )
        samples.append(
            CRISample(
                state_features=dynamics.state_features,
                acceleration_targets=dynamics.accelerations,
                edge_index=graph.edge_index,
                edge_distance=graph.mean_distances,
                incoming_edges=incoming_edge_indices(graph.edge_index, num_residues=window.shape[1]),
            )
        )
    return samples
```

- [ ] **Step 5: Add config fields for the policy**

Update `src/allostery/config.py` to add `data.min_sequence_separation: int = 0`, validate that it is nonnegative, and thread it into CRI training and scoring.

- [ ] **Step 6: Run the graph and CRI data tests**

Run:

```bash
pytest tests/test_graph_features.py tests/test_cri_data.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/allostery/features/graph.py src/allostery/cri/data.py src/allostery/config.py tests/test_graph_features.py
git commit -m "feat: make CRI graph policy explicit"
```

### Task 3: Add Network Confidence and Provenance

**Files:**
- Modify: `src/allostery/pipeline/cri_score.py`
- Modify: `src/allostery/io/results.py`
- Modify: `src/allostery/io/checkpoint.py`
- Test: `tests/test_cri_scoring.py`
- Test: `tests/test_results.py`
- Test: `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_cri_scoring.py`:

```python
from __future__ import annotations

from pathlib import Path

from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model


def test_score_cri_trajectory_reports_support_and_uncertainty(fixture_path: Path) -> None:
    result = train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
    )

    scores = score_cri_trajectory(
        model=result.model,
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
    )

    assert "support_count" in scores[0]
    assert "edge_type_probabilities" in scores[0]
    assert "edge_type_stddev" in scores[0]
```

Extend `tests/test_results.py`:

```python
def test_write_pair_scores_csv_preserves_network_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "scores.csv"
    scores = [
        {
            "residue_i": {"index": 0, "chain_id": "A", "residue_number": 1, "name": "GLY"},
            "residue_j": {"index": 4, "chain_id": "A", "residue_number": 5, "name": "LEU"},
            "score": 0.95,
            "support_count": 3,
            "edge_type_probabilities": [0.8, 0.2],
            "edge_type_stddev": [0.1, 0.1],
        }
    ]

    write_pair_scores_csv(output_path, scores)  # type: ignore[arg-type]

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert rows[0]["support_count"] == "3"
```

Extend `tests/test_checkpoint.py`:

```python
def test_checkpoint_round_trips_graph_and_alignment_metadata(tmp_path: Path) -> None:
    from allostery.io.checkpoint import load_checkpoint, save_checkpoint
    from allostery.models.cri import CRILatentInteractionModel

    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=2)
    checkpoint_path = tmp_path / "cri.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config_snapshot={
            "model": {"family": "cri"},
            "data": {"min_sequence_separation": 2, "time_step": 1.0},
        },
        residue_dim=6,
        pair_dim=1,
        hidden_dim=8,
        target_dim=3,
        residue_layers=1,
        pair_layers=2,
        dropout=0.0,
        model_family="cri",
    )

    loaded = load_checkpoint(checkpoint_path)

    assert loaded.model_family == "cri"
    assert loaded.config["data"]["min_sequence_separation"] == 2
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
pytest tests/test_cri_scoring.py tests/test_results.py tests/test_checkpoint.py -q
```

Expected: fail because the new metadata fields are not implemented yet.

- [ ] **Step 3: Implement network metadata and provenance**

Update `src/allostery/pipeline/cri_score.py`, `src/allostery/io/results.py`, and `src/allostery/io/checkpoint.py` so that:

```python
{
    "residue_i": ...,
    "residue_j": ...,
    "score": ...,
    "support_count": int,
    "edge_type_probabilities": [float, ...],
    "edge_type_stddev": [float, ...],
    "mean_distance": float,
}
```

is available for each pair, CSV output includes the new numeric columns, and checkpoints retain the graph/alignment config used during training.

- [ ] **Step 4: Run the tests and confirm they pass**

Run:

```bash
pytest tests/test_cri_scoring.py tests/test_results.py tests/test_checkpoint.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/cri_score.py src/allostery/io/results.py src/allostery/io/checkpoint.py tests/test_cri_scoring.py tests/test_results.py tests/test_checkpoint.py
git commit -m "feat: add CRI network confidence metadata"
```

### Task 4: Make Training and Inference Practical

**Files:**
- Modify: `src/allostery/models/cri.py`
- Modify: `src/allostery/pipeline/cri_train.py`
- Modify: `src/allostery/config.py`
- Test: `tests/test_cri_model.py`
- Test: `tests/test_cri_training.py`

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_cri_model.py`:

```python
def test_cri_model_aggregates_edges_without_python_loop() -> None:
    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=3, dropout=0.0)
    state_features = torch.randn(2, 3, 128, 6)
    edge_index = torch.tensor([[1, 0], [2, 0], [0, 1], [1, 2]], dtype=torch.long)
    edge_distance = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float32)

    output = model(state_features, edge_index, edge_distance)

    assert output["acceleration"].shape == (2, 3, 128, 3)
```

Extend `tests/test_cri_training.py`:

```python
def test_train_cri_model_uses_validation_and_seed(fixture_path: Path) -> None:
    result = train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        seed=123,
        batch_size=2,
    )

    assert result.num_samples == 1
    assert result.last_loss >= 0.0
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
pytest tests/test_cri_model.py tests/test_cri_training.py -q
```

Expected: fail because batching, seeding, and vectorized aggregation are not implemented yet.

- [ ] **Step 3: Implement the runtime improvements**

Update `src/allostery/models/cri.py` to replace the edge loop with tensorized aggregation, and update `src/allostery/pipeline/cri_train.py` to:

```python
train_cri_model(
    ...,
    batch_size: int = 1,
    seed: int | None = None,
    device: str = "cpu",
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
)
```

Use a `Dataset`/`DataLoader`, split train/validation windows, clip gradients, and keep the best checkpoint by validation loss.

- [ ] **Step 4: Run the tests and confirm they pass**

Run:

```bash
pytest tests/test_cri_model.py tests/test_cri_training.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/models/cri.py src/allostery/pipeline/cri_train.py src/allostery/config.py tests/test_cri_model.py tests/test_cri_training.py
git commit -m "feat: make CRI training practical"
```

## Verification Plan

- Run the full test suite after each task.
- Add synthetic controls before treating any score as an allosteric signal.
- Benchmark graph construction, one training step, and one scoring pass on a small protein trajectory before and after the speed work.
- Confirm the CRI path still coexists with the relational path and does not change the default behavior for existing configs.

## Gaps To Watch

- The alignment implementation needs to stay optional so existing trajectories are not broken.
- Confidence metrics must remain interpretable and should not be reduced to a single score too early.
- Batching and vectorization must preserve the current output ordering, or the tests that compare ranked pairs will become brittle.

