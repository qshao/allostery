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
    sparsity: row entropy of the influence matrix — minimizing it encourages
              each residue to be influenced by few others (sparse network).
    """
    reconstruction = F.mse_loss(prediction['acceleration'], target_acceleration)
    influence_matrix = prediction['influence_matrix'].clamp_min(1e-8)
    # H(row j) = -sum_i A[j,i] * log(A[j,i]); minimize to encourage peaked rows
    per_row_entropy = -torch.sum(influence_matrix * torch.log(influence_matrix), dim=-1)
    sparsity = sparsity_weight * per_row_entropy.mean()
    return InfluenceLossBreakdown(reconstruction=reconstruction, sparsity=sparsity)


__all__ = ['InfluenceLossBreakdown', 'influence_loss']
