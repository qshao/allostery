from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import torch

from allostery.cri.data import build_cri_training_samples
from allostery.io.pdb import ResidueRecord
from allostery.io.trajectory import load_trajectory
from allostery.models.cri import CRILatentInteractionModel
from allostery.pipeline.cri_train import _tensorize_sample
from allostery.pipeline.score import ResidueIdentifier


class CRIPairScore(TypedDict, total=False):
    residue_i: ResidueIdentifier
    residue_j: ResidueIdentifier
    score: float
    support_count: int
    mean_distance: float
    edge_type_probabilities: list[float]
    edge_type_stddev: list[float]



def _residue_identifier(residue: ResidueRecord) -> ResidueIdentifier:
    return {
        "index": residue.index,
        "chain_id": residue.chain_id,
        "residue_number": residue.residue_number,
        "name": residue.name,
    }


def score_cri_trajectory(
    model: CRILatentInteractionModel,
    pdb_path: str | Path,
    window_size: int,
    stride: int,
    time_step: float,
    distance_cutoff: float,
    max_neighbors: int,
    min_sequence_separation: int = 0,
    preprocess: str = "none",
    topology_path: str | Path | None = None,
) -> list[CRIPairScore]:
    trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
    samples = build_cri_training_samples(
        trajectory.coordinates,
        window_size=window_size,
        stride=stride,
        time_step=time_step,
        distance_cutoff=distance_cutoff,
        max_neighbors=max_neighbors,
        min_sequence_separation=min_sequence_separation,
        preprocess=preprocess,
    )
    if not samples:
        raise ValueError("trajectory did not yield any CRI scoring windows")

    accumulator: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    distance_accumulator: dict[tuple[int, int], list[float]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            state_features, _, edge_index, edge_distance = _tensorize_sample(sample, torch.device("cpu"))
            output = model(state_features, edge_index, edge_distance)
            probabilities = output["edge_type_prob"].squeeze(0)
            for edge_id, (sender, receiver) in enumerate(sample.edge_index.tolist()):
                unordered = tuple(sorted((int(sender), int(receiver))))
                accumulator[unordered].append(probabilities[edge_id].cpu())
                distance_accumulator[unordered].append(float(sample.edge_distance[edge_id]))

    ranked_scores: list[CRIPairScore] = []
    for (left_index, right_index), probability_values in accumulator.items():
        stacked = torch.stack(probability_values, dim=0)
        mean_prob = stacked.mean(dim=0)
        std_prob = stacked.std(dim=0, unbiased=False)
        ranked_scores.append(
            {
                "residue_i": _residue_identifier(trajectory.residues[left_index]),
                "residue_j": _residue_identifier(trajectory.residues[right_index]),
                "score": float((1.0 - mean_prob[0]).item()),
                "support_count": len(probability_values),
                "mean_distance": float(sum(distance_accumulator[(left_index, right_index)]) / len(distance_accumulator[(left_index, right_index)])),
                "edge_type_probabilities": [float(value) for value in mean_prob.tolist()],
                "edge_type_stddev": [float(value) for value in std_prob.tolist()],
            }
        )
    ranked_scores.sort(key=lambda item: item["score"], reverse=True)
    return ranked_scores


__all__ = ["CRIPairScore", "score_cri_trajectory"]
