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
