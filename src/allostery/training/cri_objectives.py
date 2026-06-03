from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CRILossBreakdown:
    reconstruction: Tensor
    entropy: Tensor
    no_edge: Tensor

    @property
    def total(self) -> Tensor:
        return self.reconstruction + self.entropy + self.no_edge


def cri_loss(
    prediction: dict[str, Tensor],
    target_acceleration: Tensor,
    entropy_weight: float,
    no_edge_weight: float,
) -> CRILossBreakdown:
    reconstruction = F.mse_loss(prediction['acceleration'], target_acceleration)
    edge_type_prob = prediction['edge_type_prob'].clamp_min(1e-8)
    edge_mask = prediction.get('edge_mask')
    if edge_type_prob.numel() == 0:
        entropy = reconstruction.new_zeros(())
    else:
        per_edge_entropy = -torch.sum(edge_type_prob * torch.log(edge_type_prob), dim=-1)
        if edge_mask is None:
            entropy = entropy_weight * per_edge_entropy.mean()
        else:
            edge_mask_float = edge_mask.to(dtype=edge_type_prob.dtype)
            normalizer = edge_mask_float.sum().clamp_min(1.0)
            entropy = entropy_weight * torch.sum(per_edge_entropy * edge_mask_float) / normalizer
    edge_score = prediction['edge_score']
    if edge_score.numel() == 0:
        no_edge = reconstruction.new_zeros(())
    else:
        if edge_mask is None:
            no_edge = no_edge_weight * torch.mean(edge_score)
        else:
            edge_mask_float = edge_mask.to(dtype=edge_score.dtype)
            normalizer = edge_mask_float.sum().clamp_min(1.0)
            no_edge = no_edge_weight * torch.sum(edge_score * edge_mask_float) / normalizer
    return CRILossBreakdown(reconstruction=reconstruction, entropy=entropy, no_edge=no_edge)


__all__ = ['CRILossBreakdown', 'cri_loss']
