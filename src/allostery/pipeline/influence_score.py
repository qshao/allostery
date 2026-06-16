from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch

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
    normalize: bool = False,
    batch_size: int = 8,
    device: str = 'cpu',
) -> list[InfluencePairScore]:
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
    samples = build_influence_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        preprocess=preprocess,
    )
    if not samples:
        raise ValueError('trajectory did not yield any influence scoring windows')

    torch_device = resolve_device(device)
    num_residues = trajectory.coordinates.shape[1]
    accumulated = torch.zeros(num_residues, num_residues, device=torch_device)
    count = 0

    model = model.to(torch_device)
    model.eval()
    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size):
            batch = stack_influence_batch(batch_samples, torch_device)
            output = model(batch.state_features)
            accumulated += output['influence_matrix'].sum(dim=0)
            count += len(batch_samples)

    mean_influence = (accumulated / max(count, 1)).cpu()  # [N, N]

    rows, cols = torch.triu_indices(num_residues, num_residues, offset=1)
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
            'support_count': count,
        }
        for k, (i, j) in enumerate(zip(rows.tolist(), cols.tolist()))
    ]
    scores.sort(key=lambda item: item['score'], reverse=True)
    return scores


__all__ = ['InfluencePairScore', 'score_influence_trajectory']
