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
