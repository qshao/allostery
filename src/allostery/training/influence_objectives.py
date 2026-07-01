from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class InfluenceLossBreakdown:
    reconstruction: Tensor
    sparsity: Tensor

    @property
    def total(self) -> Tensor:
        return self.reconstruction + self.sparsity


def influence_loss(
    prediction: dict[str, Tensor],
    target_acceleration: Tensor,
    sparsity_weight: float,
) -> InfluenceLossBreakdown:
    """
    reconstruction: MSE between predicted and target accelerations.
    sparsity: L1 mean of the influence matrix — penalises large values globally,
              encouraging a sparse network without the row-competition artefact
              of entropy regularisation (which conflicted with multi-pathway allostery).
    """
    reconstruction = F.mse_loss(prediction['acceleration'], target_acceleration)
    sparsity = sparsity_weight * prediction['influence_matrix'].mean()
    return InfluenceLossBreakdown(reconstruction=reconstruction, sparsity=sparsity)


__all__ = ['InfluenceLossBreakdown', 'influence_loss']
