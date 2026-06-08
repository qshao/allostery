from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch

from allostery.influence.data import build_influence_samples
from allostery.io.pdb import ResidueRecord, load_multimodel_pdb
from allostery.models.influence import AllostericInfluenceModel
from allostery.pipeline.score import ResidueIdentifier


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
) -> list[InfluencePairScore]:
    trajectory = load_multimodel_pdb(Path(pdb_path))
    samples = build_influence_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        preprocess=preprocess,
    )
    if not samples:
        raise ValueError('trajectory did not yield any influence scoring windows')

    num_residues = trajectory.coordinates.shape[1]
    # Accumulate influence matrices across windows
    accumulated = torch.zeros(num_residues, num_residues)
    count = 0

    model.eval()
    with torch.no_grad():
        for sample in samples:
            state_features = torch.as_tensor(
                sample.state_features[None, ...], dtype=torch.float32
            )
            output = model(state_features)
            # influence_matrix[0, j, i] = influence of i on j
            accumulated += output['influence_matrix'].squeeze(0).cpu()
            count += 1

    mean_influence = accumulated / max(count, 1)  # [N, N]

    # Score each unordered pair (i, j) as mean of both directed influences
    scores: list[InfluencePairScore] = []
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            i_on_j = float(mean_influence[j, i].item())  # i → j
            j_on_i = float(mean_influence[i, j].item())  # j → i
            scores.append(
                {
                    'residue_i': _residue_identifier(trajectory.residues[i]),
                    'residue_j': _residue_identifier(trajectory.residues[j]),
                    'score': (i_on_j + j_on_i) / 2.0,
                    'influence_i_on_j': i_on_j,
                    'influence_j_on_i': j_on_i,
                    'support_count': count,
                }
            )

    scores.sort(key=lambda item: item['score'], reverse=True)
    return scores


__all__ = ['InfluencePairScore', 'score_influence_trajectory']
