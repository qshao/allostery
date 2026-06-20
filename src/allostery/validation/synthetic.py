from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RESIDUE_NAMES = ['GLY', 'ALA', 'SER', 'THR', 'LEU', 'VAL', 'ASP', 'ASN', 'GLU', 'GLN']
_BACKBONE_SPACING = 3.8     # Angstrom CA-CA spacing along a straight backbone
_SELF_STIFFNESS = 1.0       # k0: tether of each residue to its backbone position
_INTEGRATION_DT = 0.1       # stable step for overdamped Langevin


@dataclass(frozen=True, slots=True)
class PlantedSystem:
    pdb_path: Path
    coupling_matrix: np.ndarray
    n_residues: int
    n_couplings: int


def _write_pdb(path: Path, coords: np.ndarray, n_residues: int) -> None:
    lines: list[str] = []
    serial = 1
    for frame_index in range(coords.shape[0]):
        lines.append(f'MODEL{frame_index + 1:>9}')
        for residue_index in range(n_residues):
            name = _RESIDUE_NAMES[residue_index % len(_RESIDUE_NAMES)]
            x, y, z = coords[frame_index, residue_index]
            lines.append(
                f'ATOM  {serial:5d}  CA  {name:>3s} A{residue_index + 1:4d}'
                f'{x:11.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C'
            )
            serial += 1
        lines.append('ENDMDL')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def generate_planted_system(
    out_path: str | Path,
    *,
    n_residues: int = 24,
    n_couplings: int = 8,
    coupling_strength: float = 1.0,
    noise: float = 0.05,
    frames: int = 128,
    seed: int = 0,
    min_sequence_separation: int = 2,
) -> PlantedSystem:
    if n_residues < 4:
        raise ValueError(f'n_residues must be >= 4 (got {n_residues})')
    if frames < 2:
        raise ValueError(f'frames must be >= 2 (got {frames})')
    rng = np.random.default_rng(seed)

    candidates = [
        (i, j)
        for i in range(n_residues)
        for j in range(i + min_sequence_separation, n_residues)
    ]
    if n_couplings > len(candidates):
        raise ValueError(
            f'n_couplings={n_couplings} exceeds available non-local pairs ({len(candidates)}) '
            f'for n_residues={n_residues}, min_sequence_separation={min_sequence_separation}'
        )
    chosen = rng.choice(len(candidates), size=n_couplings, replace=False)
    couplings = [candidates[int(k)] for k in chosen]

    coupling_matrix = np.zeros((n_residues, n_residues), dtype=bool)
    for i, j in couplings:
        coupling_matrix[i, j] = True
        coupling_matrix[j, i] = True

    base = np.zeros((n_residues, 3), dtype=np.float64)
    base[:, 0] = np.arange(n_residues) * _BACKBONE_SPACING
    base[:, 1] = 10.0
    base[:, 2] = 10.0

    displacement = np.zeros((n_residues, 3), dtype=np.float64)
    coords = np.empty((frames, n_residues, 3), dtype=np.float64)
    k_c = float(coupling_strength)
    for frame_index in range(frames):
        force = -_SELF_STIFFNESS * displacement
        for i, j in couplings:
            diff = displacement[i] - displacement[j]
            force[i] -= k_c * diff
            force[j] += k_c * diff
        displacement = (
            displacement
            + _INTEGRATION_DT * force
            + noise * rng.standard_normal((n_residues, 3))
        )
        coords[frame_index] = base + displacement

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdb(out_path, coords, n_residues)
    return PlantedSystem(
        pdb_path=out_path,
        coupling_matrix=coupling_matrix,
        n_residues=n_residues,
        n_couplings=n_couplings,
    )


__all__ = ['PlantedSystem', 'generate_planted_system']
