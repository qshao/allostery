# CRI-Inspired Allostery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scalable CRI-inspired latent interaction model that infers residue-residue interaction types from MD trajectories and ranks residue pairs by learned dynamic coupling.

**Architecture:** Implement a differentiable CRI-inspired model rather than exact CRI EM enumeration. The model builds sparse directed residue neighborhoods, computes position/velocity/acceleration targets from aligned C-alpha trajectories, infers per-edge latent interaction type probabilities, predicts residue accelerations from type-specific edge messages, and scores residue pairs from the learned posterior probabilities.

**Tech Stack:** Python 3.11, NumPy, PyTorch, PyYAML, pytest. No new runtime dependency is required for the first version.

---

## Source Algorithm Summary

The referenced CRI code in `/tmp/cri` implements Collective Relational Inference for particle systems. The fixed-topology implementation builds directed receiver/sender edge matrices, enumerates all edge-type realizations for each node's incoming subgraph, computes predicted forces under each realization, infers posterior probabilities from acceleration likelihood, then optimizes a type-specific force decoder. Key reference points:

- `/tmp/cri/codes_for_fixed_topology/modules.py`: `MLP_PIGNPI_Decoder` predicts type-specific pairwise force vectors.
- `/tmp/cri/codes_for_fixed_topology/train_CRI.py`: `return_latent_space`, `cmpt_categorical_distribution`, and `train` implement exact posterior inference plus M-step optimization.
- `/tmp/cri/codes_for_evolving_topology/train_EvolvingCRI.py`: adapts CRI to dynamic neighborhoods by using per-frame incoming-edge sets and complete-edge posteriors.

For proteins, exact incoming-edge enumeration is not the first implementation target because realistic residue neighborhoods make `edge_types ** incoming_edges` expensive. This plan adapts the CRI principle, "infer latent pair types by explaining observed residue dynamics", using a differentiable edge posterior instead of exact EM.

## File Structure

- Create `src/allostery/features/dynamics.py`: trajectory finite-difference utilities for velocities and accelerations.
- Create `src/allostery/features/graph.py`: sparse directed residue graph construction and incoming-edge indexing.
- Create `src/allostery/cri/__init__.py`: CRI subpackage exports.
- Create `src/allostery/cri/data.py`: CRI sample construction from trajectory windows.
- Create `src/allostery/models/cri.py`: CRI-inspired latent interaction model.
- Create `src/allostery/training/cri_objectives.py`: acceleration reconstruction, entropy, and no-edge sparsity losses.
- Create `src/allostery/pipeline/cri_train.py`: train CRI-inspired model from a PDB trajectory.
- Create `src/allostery/pipeline/cri_score.py`: score residue pairs from a trained CRI-inspired model.
- Modify `src/allostery/config.py`: add model family and CRI-specific config fields.
- Modify `src/allostery/cli.py`: route `model.family: cri` to CRI train/score paths.
- Modify `src/allostery/io/checkpoint.py`: store checkpoint `model_family` and CRI-specific dimensions.
- Modify `src/allostery/io/results.py`: support optional edge-type probability columns.
- Test files: add `tests/test_dynamics_features.py`, `tests/test_graph_features.py`, `tests/test_cri_data.py`, `tests/test_cri_model.py`, `tests/test_cri_training.py`, `tests/test_cri_scoring.py`; modify `tests/test_config.py`, `tests/test_checkpoint.py`, `tests/test_cli.py`, and `tests/test_results.py`.

## Design Decisions

- Use C-alpha coordinates already loaded by `load_multimodel_pdb`.
- Treat each residue as a particle with unit mass. This avoids inventing residue masses and matches the current C-alpha-only representation.
- Use central differences for dynamics targets: velocity at frame `t` is `(x[t + 1] - x[t - 1]) / (2 * dt)`, acceleration at frame `t` is `(x[t + 1] - 2*x[t] + x[t - 1]) / dt**2`.
- Model edge types as `K` categorical latent classes. Type `0` is "no/weak dynamic coupling"; types `1..K-1` are learned interaction modes.
- Build sparse directed edges from contact distance and top-k nearest neighbors. This preserves CRI's directed incoming-edge structure while keeping the graph scalable.
- Score an unordered residue pair by averaging the two directed posterior distributions if both directions exist, then using `1 - P(type=0)` as the primary score.
- Keep the existing `RelationalScoreModel` path working. Existing configs without `model.family` continue using the current relational model.

---

### Task 1: Add Dynamics Feature Utilities

**Files:**
- Create: `src/allostery/features/dynamics.py`
- Test: `tests/test_dynamics_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dynamics_features.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from allostery.features.dynamics import build_residue_dynamics


def test_build_residue_dynamics_uses_central_differences() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
            [[9.0, 0.0, 0.0], [0.0, 12.0, 0.0]],
        ],
        dtype=np.float32,
    )

    dynamics = build_residue_dynamics(coordinates, time_step=2.0)

    assert dynamics.positions.shape == (2, 2, 3)
    np.testing.assert_allclose(dynamics.positions, coordinates[1:-1], atol=1e-6)
    np.testing.assert_allclose(
        dynamics.velocities,
        np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.5, 0.0]],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dynamics.accelerations,
        np.array(
            [
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_build_residue_dynamics_rejects_short_windows() -> None:
    with pytest.raises(ValueError, match="at least 3 frames"):
        build_residue_dynamics(np.zeros((2, 3, 3), dtype=np.float32), time_step=1.0)


def test_build_residue_dynamics_rejects_nonfinite_coordinates() -> None:
    coordinates = np.zeros((3, 2, 3), dtype=np.float32)
    coordinates[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_residue_dynamics(coordinates, time_step=1.0)


def test_build_residue_dynamics_rejects_nonpositive_time_step() -> None:
    with pytest.raises(ValueError, match="time_step"):
        build_residue_dynamics(np.zeros((3, 2, 3), dtype=np.float32), time_step=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamics_features.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'allostery.features.dynamics'`.

- [ ] **Step 3: Implement dynamics utilities**

Create `src/allostery/features/dynamics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class ResidueDynamics:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    @property
    def state_features(self) -> np.ndarray:
        return np.concatenate((self.positions, self.velocities), axis=-1).astype(np.float32, copy=False)


def build_residue_dynamics(window_coordinates: np.ndarray, time_step: float = 1.0) -> ResidueDynamics:
    if time_step <= 0.0:
        raise ValueError("time_step must be greater than zero")

    coordinates = _validate_coordinate_window(window_coordinates)
    if coordinates.shape[0] < 3:
        raise ValueError("window_coordinates must contain at least 3 frames")
    if not np.isfinite(coordinates).all():
        raise ValueError("window_coordinates must contain only finite values")

    positions = coordinates[1:-1]
    velocities = (coordinates[2:] - coordinates[:-2]) / (2.0 * time_step)
    accelerations = (coordinates[2:] - (2.0 * coordinates[1:-1]) + coordinates[:-2]) / (time_step * time_step)
    return ResidueDynamics(
        positions=positions.astype(np.float32, copy=False),
        velocities=velocities.astype(np.float32, copy=False),
        accelerations=accelerations.astype(np.float32, copy=False),
    )


__all__ = ["ResidueDynamics", "build_residue_dynamics"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dynamics_features.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/features/dynamics.py tests/test_dynamics_features.py
git commit -m "feat: add residue dynamics features"
```

---

### Task 2: Add Sparse Directed Graph Utilities

**Files:**
- Create: `src/allostery/features/graph.py`
- Test: `tests/test_graph_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_graph_features.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from allostery.features.graph import build_directed_contact_graph, incoming_edge_indices


def test_build_directed_contact_graph_uses_cutoff_and_top_k() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    graph = build_directed_contact_graph(coordinates, distance_cutoff=3.0, max_neighbors=1)

    np.testing.assert_array_equal(
        graph.edge_index,
        np.array([[1, 0], [0, 1], [1, 2], [2, 1]], dtype=np.int64),
    )
    np.testing.assert_allclose(graph.mean_distances, np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float32), atol=1e-6)


def test_incoming_edge_indices_groups_edges_by_receiver() -> None:
    edge_index = np.array([[1, 0], [2, 0], [0, 1], [1, 2]], dtype=np.int64)

    incoming = incoming_edge_indices(edge_index, num_residues=3)

    assert [group.tolist() for group in incoming] == [[0, 1], [2], [3]]


def test_build_directed_contact_graph_rejects_invalid_parameters() -> None:
    coordinates = np.zeros((3, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="distance_cutoff"):
        build_directed_contact_graph(coordinates, distance_cutoff=0.0, max_neighbors=1)
    with pytest.raises(ValueError, match="max_neighbors"):
        build_directed_contact_graph(coordinates, distance_cutoff=1.0, max_neighbors=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_features.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'allostery.features.graph'`.

- [ ] **Step 3: Implement graph utilities**

Create `src/allostery/features/graph.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class DirectedContactGraph:
    edge_index: np.ndarray
    mean_distances: np.ndarray


def build_directed_contact_graph(
    window_coordinates: np.ndarray,
    distance_cutoff: float,
    max_neighbors: int,
) -> DirectedContactGraph:
    if distance_cutoff <= 0.0:
        raise ValueError("distance_cutoff must be greater than zero")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be greater than zero")

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
            if sender != receiver and distances[sender, receiver] <= distance_cutoff
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


def incoming_edge_indices(edge_index: np.ndarray, num_residues: int) -> tuple[np.ndarray, ...]:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
        raise ValueError("edge_index must have shape (num_edges, 2)")
    return tuple(np.flatnonzero(edge_index[:, 1] == receiver).astype(np.int64) for receiver in range(num_residues))


__all__ = ["DirectedContactGraph", "build_directed_contact_graph", "incoming_edge_indices"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_features.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/features/graph.py tests/test_graph_features.py
git commit -m "feat: add sparse residue graph utilities"
```

---

### Task 3: Add CRI Sample Construction

**Files:**
- Create: `src/allostery/cri/__init__.py`
- Create: `src/allostery/cri/data.py`
- Test: `tests/test_cri_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cri_data.py`:

```python
from __future__ import annotations

import numpy as np

from allostery.cri.data import CRISample, build_cri_training_samples


def test_build_cri_training_samples_constructs_dynamics_and_graph_windows() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 1.0, 0.0]],
            [[4.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 2.0, 0.0]],
            [[9.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 3.0, 0.0]],
        ],
        dtype=np.float32,
    )

    samples = build_cri_training_samples(
        coordinates,
        window_size=4,
        stride=1,
        time_step=1.0,
        distance_cutoff=4.0,
        max_neighbors=1,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, CRISample)
    assert sample.state_features.shape == (2, 3, 6)
    assert sample.acceleration_targets.shape == (2, 3, 3)
    np.testing.assert_array_equal(sample.edge_index, np.array([[1, 0], [0, 1], [1, 2]], dtype=np.int64))
    assert len(sample.incoming_edges) == 3


def test_build_cri_training_samples_returns_empty_for_short_trajectory() -> None:
    samples = build_cri_training_samples(
        np.zeros((2, 3, 3), dtype=np.float32),
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=4.0,
        max_neighbors=1,
    )

    assert samples == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cri_data.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'allostery.cri'`.

- [ ] **Step 3: Implement CRI data construction**

Create `src/allostery/cri/__init__.py`:

```python
from allostery.cri.data import CRISample, build_cri_training_samples

__all__ = ["CRISample", "build_cri_training_samples"]
```

Create `src/allostery/cri/data.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.features.dynamics import build_residue_dynamics
from allostery.features.graph import build_directed_contact_graph, incoming_edge_indices
from allostery.features.residue import _validate_coordinate_window


@dataclass(frozen=True, slots=True)
class CRISample:
    state_features: np.ndarray
    acceleration_targets: np.ndarray
    edge_index: np.ndarray
    edge_distance: np.ndarray
    incoming_edges: tuple[np.ndarray, ...]


def build_cri_training_samples(
    coordinates: np.ndarray,
    window_size: int,
    stride: int,
    time_step: float,
    distance_cutoff: float,
    max_neighbors: int,
) -> list[CRISample]:
    trajectory = _validate_coordinate_window(coordinates)
    if window_size < 3:
        raise ValueError("window_size must be at least 3 for central differences")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
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


__all__ = ["CRISample", "build_cri_training_samples"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cri_data.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/cri tests/test_cri_data.py
git commit -m "feat: build CRI training samples"
```

---

### Task 4: Add CRI-Inspired Latent Interaction Model

**Files:**
- Create: `src/allostery/models/cri.py`
- Test: `tests/test_cri_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cri_model.py`:

```python
from __future__ import annotations

import torch

from allostery.models.cri import CRILatentInteractionModel


def test_cri_model_predicts_accelerations_and_edge_probabilities() -> None:
    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=3, dropout=0.0)
    state_features = torch.randn(2, 3, 4, 6)
    edge_index = torch.tensor([[1, 0], [2, 0], [0, 1], [1, 2]], dtype=torch.long)
    edge_distance = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float32)

    output = model(state_features, edge_index, edge_distance)

    assert output["acceleration"].shape == (2, 3, 4, 3)
    assert output["edge_type_prob"].shape == (2, 4, 3)
    torch.testing.assert_close(output["edge_type_prob"].sum(dim=-1), torch.ones(2, 4))
    assert output["edge_score"].shape == (2, 4)


def test_cri_model_handles_empty_edges() -> None:
    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=2, dropout=0.0)
    state_features = torch.randn(1, 2, 3, 6)
    edge_index = torch.empty((0, 2), dtype=torch.long)
    edge_distance = torch.empty(0, dtype=torch.float32)

    output = model(state_features, edge_index, edge_distance)

    assert output["acceleration"].shape == (1, 2, 3, 3)
    assert output["edge_type_prob"].shape == (1, 0, 2)
    assert output["edge_score"].shape == (1, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cri_model.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'allostery.models.cri'`.

- [ ] **Step 3: Implement CRI model**

Create `src/allostery/models/cri.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor, nn


class CRILatentInteractionModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, edge_types: int, dropout: float = 0.0) -> None:
        super().__init__()
        if edge_types < 2:
            raise ValueError("edge_types must be at least 2")
        self.edge_types = edge_types
        self.message_dim = 3
        edge_input_dim = (2 * state_dim) + 1
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, edge_types),
        )
        self.edge_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * state_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, self.message_dim),
                )
                for _ in range(edge_types)
            ]
        )

    def forward(self, state_features: Tensor, edge_index: Tensor, edge_distance: Tensor) -> dict[str, Tensor]:
        if state_features.ndim != 4:
            raise ValueError("state_features must have shape (batch, time, residues, state_dim)")
        batch_size, num_steps, num_residues, state_dim = state_features.shape
        if edge_index.numel() == 0:
            empty_prob = state_features.new_empty((batch_size, 0, self.edge_types))
            empty_score = state_features.new_empty((batch_size, 0))
            return {
                "acceleration": state_features.new_zeros((batch_size, num_steps, num_residues, self.message_dim)),
                "edge_type_prob": empty_prob,
                "edge_score": empty_score,
            }

        senders = edge_index[:, 0]
        receivers = edge_index[:, 1]
        sender_state = state_features[:, :, senders, :]
        receiver_state = state_features[:, :, receivers, :]
        pair_state = torch.cat((sender_state, receiver_state), dim=-1)
        pair_summary = pair_state.mean(dim=1)
        distance_feature = edge_distance.to(state_features.device, dtype=state_features.dtype)[None, :, None].expand(batch_size, -1, -1)
        classifier_input = torch.cat((pair_summary, distance_feature), dim=-1)
        edge_type_prob = torch.softmax(self.edge_classifier(classifier_input), dim=-1)

        type_messages = torch.stack([decoder(pair_state) for decoder in self.edge_decoders], dim=-2)
        weighted_messages = torch.sum(type_messages * edge_type_prob[:, None, :, :, None], dim=-2)
        acceleration = state_features.new_zeros((batch_size, num_steps, num_residues, self.message_dim))
        for edge_id, receiver in enumerate(receivers.tolist()):
            acceleration[:, :, receiver, :] = acceleration[:, :, receiver, :] + weighted_messages[:, :, edge_id, :]

        edge_score = 1.0 - edge_type_prob[:, :, 0]
        return {
            "acceleration": acceleration,
            "edge_type_prob": edge_type_prob,
            "edge_score": edge_score,
        }


__all__ = ["CRILatentInteractionModel"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cri_model.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/models/cri.py tests/test_cri_model.py
git commit -m "feat: add CRI latent interaction model"
```

---

### Task 5: Add CRI Training Objectives

**Files:**
- Create: `src/allostery/training/cri_objectives.py`
- Test: `tests/test_cri_training.py`

- [ ] **Step 1: Write failing tests**

Create the objective-focused part of `tests/test_cri_training.py`:

```python
from __future__ import annotations

import torch

from allostery.training.cri_objectives import cri_loss


def test_cri_loss_combines_reconstruction_entropy_and_sparsity() -> None:
    prediction = {
        "acceleration": torch.zeros(1, 2, 3, 3),
        "edge_type_prob": torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]], dtype=torch.float32),
        "edge_score": torch.tensor([[0.3, 0.9]], dtype=torch.float32),
    }
    target = torch.ones(1, 2, 3, 3)

    losses = cri_loss(
        prediction,
        target,
        entropy_weight=0.01,
        no_edge_weight=0.02,
    )

    assert losses.reconstruction.item() > 0.0
    assert losses.entropy.item() > 0.0
    assert losses.no_edge.item() > 0.0
    torch.testing.assert_close(losses.total, losses.reconstruction + losses.entropy + losses.no_edge)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cri_training.py::test_cri_loss_combines_reconstruction_entropy_and_sparsity -q`

Expected: fail with `ModuleNotFoundError: No module named 'allostery.training.cri_objectives'`.

- [ ] **Step 3: Implement CRI objectives**

Create `src/allostery/training/cri_objectives.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CRILossBreakdown:
    reconstruction: Tensor
    entropy: Tensor
    no_edge: Tensor

    @property
    def total(self) -> Tensor:
        return self.reconstruction + self.entropy + self.no_edge


def cri_loss(
    prediction: dict[str, Tensor],
    target_acceleration: Tensor,
    entropy_weight: float,
    no_edge_weight: float,
) -> CRILossBreakdown:
    reconstruction = F.mse_loss(prediction["acceleration"], target_acceleration)
    edge_type_prob = prediction["edge_type_prob"].clamp_min(1e-8)
    entropy = entropy_weight * torch.mean(torch.sum(edge_type_prob * torch.log(edge_type_prob), dim=-1))
    no_edge = no_edge_weight * torch.mean(prediction["edge_score"])
    return CRILossBreakdown(reconstruction=reconstruction, entropy=entropy, no_edge=no_edge)


__all__ = ["CRILossBreakdown", "cri_loss"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cri_training.py::test_cri_loss_combines_reconstruction_entropy_and_sparsity -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/training/cri_objectives.py tests/test_cri_training.py
git commit -m "feat: add CRI training objectives"
```

---

### Task 6: Add CRI Training Pipeline

**Files:**
- Create: `src/allostery/pipeline/cri_train.py`
- Modify: `tests/test_cri_training.py`

- [ ] **Step 1: Add failing pipeline test**

Append to `tests/test_cri_training.py`:

```python
from pathlib import Path

from allostery.pipeline.cri_train import train_cri_model


def test_train_cri_model_runs_on_tiny_trajectory(fixture_path: Path) -> None:
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

    assert result.num_samples == 1
    assert result.last_loss >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cri_training.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'allostery.pipeline.cri_train'`.

- [ ] **Step 3: Implement CRI training pipeline**

Create `src/allostery/pipeline/cri_train.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from allostery.cri.data import CRISample, build_cri_training_samples
from allostery.io.checkpoint import save_checkpoint
from allostery.io.pdb import load_multimodel_pdb
from allostery.models.cri import CRILatentInteractionModel
from allostery.training.cri_objectives import cri_loss


@dataclass(frozen=True, slots=True)
class CRITrainResult:
    model: CRILatentInteractionModel
    num_samples: int
    last_loss: float


def _tensorize_sample(sample: CRISample) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(sample.state_features[None, ...], dtype=torch.float32),
        torch.as_tensor(sample.acceleration_targets[None, ...], dtype=torch.float32),
        torch.as_tensor(sample.edge_index, dtype=torch.long),
        torch.as_tensor(sample.edge_distance, dtype=torch.float32),
    )


def train_cri_model(
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float,
    distance_cutoff: float,
    max_neighbors: int,
    edge_types: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    entropy_weight: float,
    no_edge_weight: float,
    checkpoint_path: str | Path | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> CRITrainResult:
    trajectory = load_multimodel_pdb(Path(pdb_path))
    samples = build_cri_training_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        distance_cutoff=distance_cutoff,
        max_neighbors=max_neighbors,
    )
    if not samples:
        raise ValueError("trajectory did not yield any CRI training windows")

    state_dim = int(samples[0].state_features.shape[-1])
    model = CRILatentInteractionModel(state_dim=state_dim, hidden_dim=hidden_dim, edge_types=edge_types, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        for sample in samples:
            state_features, targets, edge_index, edge_distance = _tensorize_sample(sample)
            output = model(state_features, edge_index, edge_distance)
            losses = cri_loss(
                output,
                targets,
                entropy_weight=entropy_weight,
                no_edge_weight=no_edge_weight,
            )
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            last_loss = float(losses.total.detach().item())

    if checkpoint_path is not None:
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            config_snapshot=config_snapshot or {},
            residue_dim=state_dim,
            pair_dim=1,
            hidden_dim=hidden_dim,
            target_dim=3,
            residue_layers=1,
            pair_layers=edge_types,
            dropout=dropout,
            model_family="cri",
        )

    return CRITrainResult(model=model, num_samples=len(samples), last_loss=last_loss)


__all__ = ["CRITrainResult", "train_cri_model"]
```

- [ ] **Step 4: Run test to verify it exposes checkpoint signature gap**

Run: `pytest tests/test_cri_training.py -q`

Expected: objective and no-checkpoint training tests pass. If the checkpoint path is exercised later, `save_checkpoint` still needs `model_family`; that change happens in Task 8.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/cri_train.py tests/test_cri_training.py
git commit -m "feat: train CRI latent interaction model"
```

---

### Task 7: Add CRI Scoring Pipeline

**Files:**
- Create: `src/allostery/pipeline/cri_score.py`
- Test: `tests/test_cri_scoring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cri_scoring.py`:

```python
from __future__ import annotations

from pathlib import Path

from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model


def test_score_cri_trajectory_returns_ranked_residue_pairs(fixture_path: Path) -> None:
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

    assert scores
    assert scores[0]["score"] >= scores[-1]["score"]
    assert "edge_type_probabilities" in scores[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cri_scoring.py -q`

Expected: fail with `ModuleNotFoundError: No module named 'allostery.pipeline.cri_score'`.

- [ ] **Step 3: Implement CRI scoring**

Create `src/allostery/pipeline/cri_score.py`:

```python
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import torch

from allostery.cri.data import build_cri_training_samples
from allostery.io.pdb import ResidueRecord, load_multimodel_pdb
from allostery.models.cri import CRILatentInteractionModel
from allostery.pipeline.cri_train import _tensorize_sample
from allostery.pipeline.score import ResidueIdentifier


class CRIPairScore(TypedDict):
    residue_i: ResidueIdentifier
    residue_j: ResidueIdentifier
    score: float
    edge_type_probabilities: list[float]


def _residue_identifier(residue: ResidueRecord) -> ResidueIdentifier:
    return {
        "index": residue.index,
        "chain_id": residue.chain_id,
        "residue_number": residue.residue_number,
        "name": residue.name,
    }


def score_cri_trajectory(
    model: CRILatentInteractionModel,
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float,
    distance_cutoff: float,
    max_neighbors: int,
) -> list[CRIPairScore]:
    trajectory = load_multimodel_pdb(Path(pdb_path))
    samples = build_cri_training_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        distance_cutoff=distance_cutoff,
        max_neighbors=max_neighbors,
    )
    if not samples:
        raise ValueError("trajectory did not yield any CRI scoring windows")

    accumulator: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            state_features, _, edge_index, edge_distance = _tensorize_sample(sample)
            output = model(state_features, edge_index, edge_distance)
            probabilities = output["edge_type_prob"].squeeze(0)
            for edge_id, (sender, receiver) in enumerate(sample.edge_index.tolist()):
                unordered = tuple(sorted((int(sender), int(receiver))))
                accumulator[unordered].append(probabilities[edge_id].cpu())

    ranked_scores: list[CRIPairScore] = []
    for (left_index, right_index), probability_values in accumulator.items():
        mean_prob = torch.stack(probability_values, dim=0).mean(dim=0)
        ranked_scores.append(
            {
                "residue_i": _residue_identifier(trajectory.residues[left_index]),
                "residue_j": _residue_identifier(trajectory.residues[right_index]),
                "score": float((1.0 - mean_prob[0]).item()),
                "edge_type_probabilities": [float(value) for value in mean_prob.tolist()],
            }
        )
    ranked_scores.sort(key=lambda item: item["score"], reverse=True)
    return ranked_scores


__all__ = ["CRIPairScore", "score_cri_trajectory"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cri_scoring.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/cri_score.py tests/test_cri_scoring.py
git commit -m "feat: score CRI residue interactions"
```

---

### Task 8: Extend Checkpoint Metadata for Model Family

**Files:**
- Modify: `src/allostery/io/checkpoint.py`
- Modify: `tests/test_checkpoint.py`

- [ ] **Step 1: Add failing checkpoint tests**

In `tests/test_checkpoint.py`, add a test that saves and loads a CRI checkpoint:

```python
def test_checkpoint_round_trips_model_family(tmp_path):
    import torch
    from allostery.io.checkpoint import load_checkpoint, save_checkpoint
    from allostery.models.cri import CRILatentInteractionModel

    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=2)
    checkpoint_path = tmp_path / "cri.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config_snapshot={"model": {"family": "cri"}},
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
    assert loaded.residue_dim == 6
    assert isinstance(loaded.state_dict, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_checkpoint.py::test_checkpoint_round_trips_model_family -q`

Expected: fail with `TypeError: save_checkpoint() got an unexpected keyword argument 'model_family'`.

- [ ] **Step 3: Modify checkpoint schema**

Modify `src/allostery/io/checkpoint.py`:

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
    model_family: str = "relational"
```

Update `save_checkpoint` signature:

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
    model_family: str = "relational",
) -> None:
```

Add `"model_family": model_family` to the saved dictionary.

Update `load_checkpoint` return:

```python
model_family=str(raw.get("model_family", "relational")),
```

- [ ] **Step 4: Run checkpoint tests**

Run: `pytest tests/test_checkpoint.py -q`

Expected: all checkpoint tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/io/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: persist model family in checkpoints"
```

---

### Task 9: Extend Config for CRI Model Family

**Files:**
- Modify: `src/allostery/config.py`
- Modify: `tests/test_config.py`
- Modify: `examples/example_config.yaml`

- [ ] **Step 1: Add failing config test**

Add to `tests/test_config.py`:

```python
def test_load_config_parses_cri_model_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "cri.yaml"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 3",
            "  horizon_size: 1",
            "  stride: 1",
            "  time_step: 1.0",
            "  distance_cutoff: 20.0",
            "  max_neighbors: 2",
            "model:",
            "  family: cri",
            "  hidden_dim: 8",
            "  residue_layers: 1",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "  edge_types: 2",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.0",
            "  entropy_weight: 0.0",
            "  no_edge_weight: 0.0",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/cri.pt",
            "  score_csv_path: outputs/cri_scores.csv",
        ],
    )

    config = load_config(config_path)

    assert config.model.family == "cri"
    assert config.model.edge_types == 2
    assert config.data.time_step == 1.0
    assert config.data.distance_cutoff == 20.0
    assert config.data.max_neighbors == 2
    assert config.training is not None
    assert config.training.entropy_weight == 0.0
    assert config.training.no_edge_weight == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_load_config_parses_cri_model_fields -q`

Expected: fail because `ModelConfig` and `DataConfig` do not expose CRI fields.

- [ ] **Step 3: Modify config dataclasses and parser**

Update `src/allostery/config.py` dataclasses:

```python
@dataclass(frozen=True, slots=True)
class DataConfig:
    pdb_path: Path
    window_size: int
    horizon_size: int
    stride: int
    time_step: float = 1.0
    distance_cutoff: float = 8.0
    max_neighbors: int = 8


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int
    residue_layers: int
    pair_layers: int
    dropout: float
    family: str = "relational"
    edge_types: int = 2


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    consistency_weight: float
    entropy_weight: float = 0.0
    no_edge_weight: float = 0.0
```

When constructing `DataConfig`, use:

```python
time_step=float(data_raw.get("time_step", 1.0)),
distance_cutoff=float(data_raw.get("distance_cutoff", 8.0)),
max_neighbors=int(data_raw.get("max_neighbors", 8)),
```

When constructing `ModelConfig`, use:

```python
family=str(model_raw.get("family", "relational")),
edge_types=int(model_raw.get("edge_types", 2)),
```

When constructing `TrainingConfig`, use:

```python
entropy_weight=float(training_raw.get("entropy_weight", 0.0)),
no_edge_weight=float(training_raw.get("no_edge_weight", 0.0)),
```

Add validation:

```python
if config.model.family not in {"relational", "cri"}:
    raise ValueError("model.family must be relational or cri")
if config.data.time_step <= 0.0:
    raise ValueError("time_step must be greater than zero")
if config.data.distance_cutoff <= 0.0:
    raise ValueError("distance_cutoff must be greater than zero")
if config.data.max_neighbors <= 0:
    raise ValueError("max_neighbors must be greater than zero")
if config.model.edge_types < 2:
    raise ValueError("edge_types must be at least 2")
```

- [ ] **Step 4: Add CRI example config**

Create `examples/cri_example_config.yaml`:

```yaml
mode: run

data:
  pdb_path: ../tests/fixtures/tiny_trajectory.pdb
  window_size: 3
  horizon_size: 1
  stride: 1
  time_step: 1.0
  distance_cutoff: 20.0
  max_neighbors: 2

model:
  family: cri
  hidden_dim: 8
  residue_layers: 1
  pair_layers: 2
  dropout: 0.0
  edge_types: 2

training:
  epochs: 1
  learning_rate: 0.001
  consistency_weight: 0.0
  entropy_weight: 0.0
  no_edge_weight: 0.0

scoring:
  top_k: 5

output:
  model_path: ../outputs/cri_example_model.pt
  score_csv_path: ../outputs/cri_example_scores.csv
```

- [ ] **Step 5: Run config tests**

Run: `pytest tests/test_config.py -q`

Expected: all config tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/allostery/config.py tests/test_config.py examples/cri_example_config.yaml
git commit -m "feat: configure CRI model family"
```

---

### Task 10: Route CLI to CRI Pipelines

**Files:**
- Modify: `src/allostery/cli.py`
- Modify: `src/allostery/pipeline/score.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI run test**

In `tests/test_cli.py`, add a CRI run-mode test using the existing test helpers:

```python
def test_cli_runs_cri_model_family(tmp_path: Path, fixture_path: Path) -> None:
    config_path = tmp_path / "cri.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: run",
                "data:",
                f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
                "  window_size: 3",
                "  horizon_size: 1",
                "  stride: 1",
                "  time_step: 1.0",
                "  distance_cutoff: 20.0",
                "  max_neighbors: 2",
                "model:",
                "  family: cri",
                "  hidden_dim: 8",
                "  residue_layers: 1",
                "  pair_layers: 2",
                "  dropout: 0.0",
                "  edge_types: 2",
                "training:",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  consistency_weight: 0.0",
                "  entropy_weight: 0.0",
                "  no_edge_weight: 0.0",
                "scoring:",
                "  top_k: 5",
                "output:",
                "  model_path: model.pt",
                "  score_csv_path: scores.csv",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main([str(config_path)])

    assert exit_code == 0
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "scores.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_cli_runs_cri_model_family -q`

Expected: fail because CLI still routes all training/scoring to the relational pipeline.

- [ ] **Step 3: Modify CLI routing**

Update imports in `src/allostery/cli.py`:

```python
from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model
```

Update `_run_train`:

```python
if config.model.family == "cri":
    result = train_cri_model(
        pdb_path=config.data.pdb_path,
        window_size=config.data.window_size,
        stride=config.data.stride,
        time_step=config.data.time_step,
        distance_cutoff=config.data.distance_cutoff,
        max_neighbors=config.data.max_neighbors,
        edge_types=config.model.edge_types,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        epochs=training.epochs,
        learning_rate=training.learning_rate,
        entropy_weight=training.entropy_weight,
        no_edge_weight=training.no_edge_weight,
        checkpoint_path=model_path,
        config_snapshot=_serialize_config(config),
    )
    print(f"trained CRI samples={result.num_samples} checkpoint={model_path}")
    return result
```

Leave the existing relational branch after the CRI branch.

Update `_run_score`:

```python
model = load_scoring_model(model_path)
if config.model.family == "cri":
    scores = score_cri_trajectory(
        model=model,
        pdb_path=config.data.pdb_path,
        window_size=config.data.window_size,
        stride=config.data.stride,
        time_step=config.data.time_step,
        distance_cutoff=config.data.distance_cutoff,
        max_neighbors=config.data.max_neighbors,
    )
else:
    scores = score_trajectory(
        model=model,
        pdb_path=config.data.pdb_path,
        window_size=config.data.window_size,
        horizon_size=config.data.horizon_size,
        stride=config.data.stride,
    )
```

Modify `load_scoring_model` or add a new loader so CRI checkpoints instantiate `CRILatentInteractionModel`:

```python
if checkpoint.model_family == "cri":
    model = CRILatentInteractionModel(
        state_dim=checkpoint.residue_dim,
        hidden_dim=checkpoint.hidden_dim,
        edge_types=checkpoint.pair_layers,
        dropout=checkpoint.dropout,
    )
else:
    model = RelationalScoreModel(...)
```

- [ ] **Step 4: Run CLI test**

Run: `pytest tests/test_cli.py::test_cli_runs_cri_model_family -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/cli.py src/allostery/pipeline/score.py tests/test_cli.py
git commit -m "feat: route CLI to CRI pipelines"
```

---

### Task 11: Extend Score CSV for Edge-Type Probabilities

**Files:**
- Modify: `src/allostery/io/results.py`
- Modify: `tests/test_results.py`

- [ ] **Step 1: Add failing results test**

Add to `tests/test_results.py`:

```python
def test_write_pair_scores_csv_includes_edge_type_probabilities(tmp_path: Path) -> None:
    from allostery.io.results import write_pair_scores_csv

    output_path = tmp_path / "cri_scores.csv"
    write_pair_scores_csv(
        output_path,
        [
            {
                "residue_i": {"index": 0, "chain_id": "A", "residue_number": 1, "name": "GLY"},
                "residue_j": {"index": 1, "chain_id": "A", "residue_number": 2, "name": "ALA"},
                "score": 0.75,
                "edge_type_probabilities": [0.25, 0.75],
            }
        ],
    )

    text = output_path.read_text(encoding="utf-8")

    assert "edge_type_probabilities" in text
    assert "0.25;0.75" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_results.py::test_write_pair_scores_csv_includes_edge_type_probabilities -q`

Expected: fail because the CSV writer does not include the optional field.

- [ ] **Step 3: Modify result writer**

In `src/allostery/io/results.py`, add `"edge_type_probabilities"` to fieldnames and serialize optional probabilities:

```python
probabilities = pair_score.get("edge_type_probabilities", [])
row["edge_type_probabilities"] = ";".join(f"{float(value):.6g}" for value in probabilities)
```

For existing relational scores, this column is present and empty.

- [ ] **Step 4: Run result tests**

Run: `pytest tests/test_results.py -q`

Expected: all result tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/io/results.py tests/test_results.py
git commit -m "feat: write CRI edge type probabilities"
```

---

### Task 12: Full Verification and Documentation

**Files:**
- Modify: `README.md`
- Test: full suite

- [ ] **Step 1: Update README**

Add a concise section after "Run From YAML":

```markdown
## CRI-Inspired Latent Interaction Model

Set `model.family: cri` to train a CRI-inspired model that infers latent residue-pair interaction types by reconstructing residue accelerations from sparse directed contact neighborhoods. Type `0` is interpreted as no/weak coupling; score CSVs rank unordered residue pairs by `1 - P(type=0)` and include the mean edge-type probability vector.

The first implementation uses C-alpha coordinates, unit residue masses, central-difference velocities and accelerations, and sparse neighborhoods from `distance_cutoff` plus `max_neighbors`. Trajectories should be aligned and imaged before conversion to multi-model PDB, because the package does not yet perform MD alignment or periodic-boundary correction.
```

- [ ] **Step 2: Run focused CRI tests**

Run:

```bash
pytest tests/test_dynamics_features.py tests/test_graph_features.py tests/test_cri_data.py tests/test_cri_model.py tests/test_cri_training.py tests/test_cri_scoring.py -q
```

Expected: all focused CRI tests pass.

- [ ] **Step 3: Run full test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Run example CRI config**

Run:

```bash
PYTHONPATH=src python -m allostery.cli examples/cri_example_config.yaml
```

Expected: command exits with code 0, creates `outputs/cri_example_model.pt`, creates `outputs/cri_example_scores.csv`, and prints `completed mode=run`.

- [ ] **Step 5: Inspect git status**

Run: `git status --short`

Expected: only intentional files are modified.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document CRI-inspired model"
```

---

## Acceptance Criteria

- Existing relational model configs continue to train and score without adding CRI fields.
- `model.family: cri` trains a CRI-inspired model on the tiny fixture and produces ranked pair scores.
- CRI score CSV rows include `edge_type_probabilities`.
- CRI training loss directly supervises the learned interaction posteriors through acceleration reconstruction.
- The implementation avoids exact incoming-edge enumeration, so runtime scales with number of sparse directed edges and edge types, not with `edge_types ** incoming_neighbors`.
- Full test suite passes with `pytest -q`.

## Known Limitations

- The first version assumes aligned C-alpha trajectories and does not perform structural alignment, PBC unwrapping, or imaging.
- Unit residue masses are used.
- Edge-type identities are latent clusters; only type `0` receives a fixed interpretation as no/weak coupling.
- The model is dynamics-reconstruction based. It does not use external allosteric labels unless later tasks add supervised endpoints or perturbation labels.

## Follow-Up Work

- Add trajectory alignment and PBC preprocessing or integrate MDAnalysis/mdtraj.
- Add time-varying edge neighborhoods for an Evolving-CRI variant.
- Add exact or block mean-field CRI posterior inference for small neighborhoods.
- Add supervised allostery labels or perturbation-response targets when available.
