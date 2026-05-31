from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CA_ATOM_NAME = "CA"
_CHAIN_FALLBACK = "_"


@dataclass(frozen=True, slots=True)
class ResidueRecord:
    index: int
    chain_id: str
    residue_number: int
    name: str


@dataclass(frozen=True, slots=True)
class Trajectory:
    residues: tuple[ResidueRecord, ...]
    coordinates: np.ndarray


def load_multimodel_pdb(path: str | Path) -> Trajectory:
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    frames: list[list[tuple[tuple[str, int, str], np.ndarray]]] = []
    current_frame: list[tuple[tuple[str, int, str], np.ndarray]] | None = None
    saw_model = False

    for line in lines:
        record_name = line[0:6].strip()
        if record_name == "MODEL":
            if current_frame is not None:
                raise ValueError("Nested MODEL blocks are not supported")
            saw_model = True
            current_frame = []
            continue

        if record_name == "ENDMDL":
            if current_frame is None:
                raise ValueError("ENDMDL encountered before MODEL")
            if not current_frame:
                raise ValueError("MODEL block did not contain any CA atoms")
            frames.append(current_frame)
            current_frame = None
            continue

        if current_frame is None or record_name != "ATOM":
            continue

        atom_name = line[12:16].strip()
        if atom_name != _CA_ATOM_NAME:
            continue

        residue_name = line[17:20].strip()
        chain_id = line[21:22].strip() or _CHAIN_FALLBACK
        residue_number = int(line[22:26])
        insertion_code = line[26:27].strip()
        if insertion_code:
            raise ValueError("PDB insertion codes are not supported in V1")
        key = (chain_id, residue_number, residue_name)
        xyz = np.array(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ),
            dtype=np.float32,
        )
        if any(existing_key == key for existing_key, _ in current_frame):
            raise ValueError("Duplicate residue key within MODEL block")
        current_frame.append((key, xyz))

    if not saw_model:
        raise ValueError("PDB must contain MODEL records")
    if current_frame is not None:
        raise ValueError("Unterminated MODEL block")
    if not frames:
        raise ValueError("PDB did not contain any completed MODEL blocks")

    reference_keys = [key for key, _ in frames[0]]
    if not reference_keys:
        raise ValueError("First model did not contain any CA atoms")

    residues = tuple(
        ResidueRecord(index=index, chain_id=key[0], residue_number=key[1], name=key[2])
        for index, key in enumerate(reference_keys)
    )

    coordinates = np.empty((len(frames), len(reference_keys), 3), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        keys = [key for key, _ in frame]
        if keys != reference_keys:
            raise ValueError("Residue identity or ordering changed across models")
        coordinates[frame_index] = np.stack([xyz for _, xyz in frame], axis=0)

    return Trajectory(residues=residues, coordinates=coordinates)


__all__ = ["ResidueRecord", "Trajectory", "load_multimodel_pdb"]
