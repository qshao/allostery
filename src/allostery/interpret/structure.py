from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.io.pdb import Trajectory


@dataclass
class ResidueStructuralFeatures:
    rmsf: float
    contact_number: int


@dataclass
class StructuralContext:
    per_residue: dict[int, ResidueStructuralFeatures]
    mean_coords: np.ndarray
    label_to_index: dict[str, int]
    contact_cutoff: float

    def geometry(self, labels: list[str]) -> dict[str, float]:
        indices = [self.label_to_index[label] for label in labels if label in self.label_to_index]
        if not indices:
            return {"radius_of_gyration": 0.0, "n_resolved": 0}
        points = self.mean_coords[indices]
        centroid = points.mean(axis=0)
        rg = float(np.sqrt(((points - centroid) ** 2).sum(axis=1).mean()))
        return {"radius_of_gyration": rg, "n_resolved": len(indices)}


def compute_structural_context(trajectory: Trajectory, contact_cutoff: float = 8.0) -> StructuralContext:
    coords = np.asarray(trajectory.coordinates, dtype=np.float64)  # [F, N, 3]
    mean = coords.mean(axis=0)  # [N, 3]
    displacement = coords - mean[None, :, :]
    rmsf = np.sqrt((displacement ** 2).sum(axis=2).mean(axis=0))  # [N]

    diff = mean[:, None, :] - mean[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=2))
    contacts = (distance < contact_cutoff) & (distance > 0.0)
    contact_number = contacts.sum(axis=1)

    per_residue = {
        i: ResidueStructuralFeatures(rmsf=float(rmsf[i]), contact_number=int(contact_number[i]))
        for i in range(mean.shape[0])
    }
    label_to_index = {
        f"{r.chain_id}:{r.residue_number} {r.name}": i
        for i, r in enumerate(trajectory.residues)
    }
    return StructuralContext(
        per_residue=per_residue,
        mean_coords=mean,
        label_to_index=label_to_index,
        contact_cutoff=contact_cutoff,
    )


__all__ = [
    "ResidueStructuralFeatures", "StructuralContext", "compute_structural_context",
]
