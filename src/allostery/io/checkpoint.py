from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    state_dict: dict[str, Tensor]
    residue_dim: int
    pair_dim: int
    hidden_dim: int
    residue_layers: int
    pair_layers: int
    dropout: float
    target_dim: int
    config: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config_snapshot: Mapping[str, Any],
    residue_dim: int,
    pair_dim: int,
    hidden_dim: int,
    target_dim: int,
    residue_layers: int = 2,
    pair_layers: int = 2,
    dropout: float = 0.0,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": dict(model.state_dict().items()),
            "residue_dim": residue_dim,
            "pair_dim": pair_dim,
            "hidden_dim": hidden_dim,
            "residue_layers": residue_layers,
            "pair_layers": pair_layers,
            "dropout": dropout,
            "target_dim": target_dim,
            "config": dict(config_snapshot),
        },
        target,
    )


def load_checkpoint(path: str | Path) -> ModelCheckpoint:
    raw = torch.load(Path(path), map_location="cpu")
    return ModelCheckpoint(
        state_dict=dict(_require_mapping(raw, "state_dict")),
        residue_dim=int(raw["residue_dim"]),
        pair_dim=int(raw["pair_dim"]),
        hidden_dim=int(raw["hidden_dim"]),
        residue_layers=int(raw.get("residue_layers", 2)),
        pair_layers=int(raw.get("pair_layers", 2)),
        dropout=float(raw.get("dropout", 0.0)),
        target_dim=int(raw["target_dim"]),
        config=dict(_require_mapping(raw, "config")),
    )


def _require_mapping(raw: object, key: str) -> Mapping[str, Any]:
    if not isinstance(raw, dict) or key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"checkpoint is missing {key}")
    return raw[key]


__all__ = ["ModelCheckpoint", "load_checkpoint", "save_checkpoint"]
