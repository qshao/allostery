from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TrainingLossBreakdown:
    future_summary: Tensor
    consistency: Tensor

    @property
    def total(self) -> Tensor:
        return self.future_summary + self.consistency


def future_summary_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.huber_loss(prediction, target)


def consistency_loss(current_scores: Tensor, next_scores: Tensor) -> Tensor:
    return F.mse_loss(current_scores, next_scores)


__all__ = [
    "TrainingLossBreakdown",
    "consistency_loss",
    "future_summary_loss",
]
