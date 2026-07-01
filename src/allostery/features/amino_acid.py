from __future__ import annotations

_AA_TO_IDX: dict[str, int] = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
}
_UNK_IDX = 20
NUM_AMINO_ACID_TYPES = 21  # 20 standard + 1 unknown


def aa_name_to_idx(name: str) -> int:
    """Map a 3-letter amino acid code to a 0-based index (unknown → 20)."""
    return _AA_TO_IDX.get(name.upper(), _UNK_IDX)


__all__ = ['NUM_AMINO_ACID_TYPES', 'aa_name_to_idx']
