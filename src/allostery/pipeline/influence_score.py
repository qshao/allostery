from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch

from allostery.features.amino_acid import aa_name_to_idx
from allostery.influence.data import build_influence_samples
from allostery.io.pdb import ResidueRecord
from allostery.io.trajectory import load_trajectory
from allostery.models.influence import AllostericInfluenceModel
from allostery.pipeline.score import ResidueIdentifier
from allostery.training.runtime import iter_batches, resolve_device, stack_influence_batch


class InfluencePairScore(TypedDict, total=False):
    residue_i: ResidueIdentifier
    residue_j: ResidueIdentifier
    score: float
    influence_i_on_j: float   # mean A[j, i] — how strongly i drives j
    influence_j_on_i: float   # mean A[i, j] — how strongly j drives i
    support_count: int


def _residue_identifier(residue: ResidueRecord) -> ResidueIdentifier:
    return {
        'index': residue.index,
        'chain_id': residue.chain_id,
        'residue_number': residue.residue_number,
        'name': residue.name,
    }


def score_influence_trajectory(
    model: AllostericInfluenceModel,
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float = 1.0,
    preprocess: str = 'none',
    topology_path: str | Path | None = None,
    normalize: bool = True,
    batch_size: int = 8,
    device: str = 'cpu',
    min_sequence_separation: int = 1,
    window_sizes: list[int] | None = None,
) -> list[InfluencePairScore]:
    """Score a trajectory with the influence model.

    window_sizes: if provided, averages influence matrices across all listed
                  window sizes (multi-scale scoring).
    """
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)

    # Residue identity indices — matches training-time injection.
    residue_type_indices = torch.tensor(
        [aa_name_to_idx(r.name) for r in trajectory.residues],
        dtype=torch.long,
    )

    effective_window_sizes = window_sizes if window_sizes else [window_size]

    torch_device = resolve_device(device)
    num_residues = trajectory.coordinates.shape[1]
    accumulated = torch.zeros(num_residues, num_residues, device=torch_device)
    total_count = 0

    rti = residue_type_indices.to(torch_device)
    model = model.to(torch_device)
    model.eval()

    for ws in effective_window_sizes:
        samples = build_influence_samples(
            trajectory.coordinates,
            window_size=ws,
            stride=stride,
            time_step=time_step,
            preprocess=preprocess,
            normalize=normalize,
        )
        if not samples:
            continue
        with torch.no_grad():
            for batch_samples in iter_batches(samples, batch_size):
                batch = stack_influence_batch(batch_samples, torch_device)
                output = model(batch.state_features, residue_type_indices=rti)
                accumulated += output['influence_matrix'].sum(dim=0)
                total_count += len(batch_samples)

    if total_count == 0:
        raise ValueError('trajectory did not yield any influence scoring windows')

    mean_influence = (accumulated / total_count).cpu()  # [N, N]

    sep = max(min_sequence_separation, model.min_sequence_separation)
    rows, cols = torch.triu_indices(num_residues, num_residues, offset=sep)
    i_on_j = mean_influence[cols, rows]   # influence of i on j  (A[j, i])
    j_on_i = mean_influence[rows, cols]   # influence of j on i  (A[i, j])
    pair_score = (i_on_j + j_on_i) / 2.0

    scores: list[InfluencePairScore] = [
        {
            'residue_i': _residue_identifier(trajectory.residues[int(i)]),
            'residue_j': _residue_identifier(trajectory.residues[int(j)]),
            'score': float(pair_score[k].item()),
            'influence_i_on_j': float(i_on_j[k].item()),
            'influence_j_on_i': float(j_on_i[k].item()),
            'support_count': total_count,
        }
        for k, (i, j) in enumerate(zip(rows.tolist(), cols.tolist()))
    ]
    scores.sort(key=lambda item: item['score'], reverse=True)
    return scores


__all__ = ['InfluencePairScore', 'score_influence_trajectory']
