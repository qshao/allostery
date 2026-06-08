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

    # Try MDAnalysis first (already imported or available to import)
    mda_in_modules = 'MDAnalysis' in sys.modules
    if mda_in_modules:
        mda = sys.modules['MDAnalysis']
        if mda is not None:
            return _load_via_mdanalysis(path, topology_path)
    else:
        try:
            import MDAnalysis  # noqa: F401
            return _load_via_mdanalysis(path, topology_path)
        except ImportError:
            pass

    # Try MDTraj
    mdt_in_modules = 'mdtraj' in sys.modules
    if mdt_in_modules:
        mdt = sys.modules['mdtraj']
        if mdt is not None:
            return _load_via_mdtraj(path, topology_path)
    else:
        try:
            import mdtraj  # noqa: F401
            return _load_via_mdtraj(path, topology_path)
        except ImportError:
            pass

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
