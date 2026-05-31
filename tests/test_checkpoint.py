from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from allostery.io import load_checkpoint, save_checkpoint
from allostery.models.relational import RelationalScoreModel


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = RelationalScoreModel(
        residue_dim=10,
        pair_dim=5,
        hidden_dim=8,
        target_dim=3,
        residue_layers=3,
        pair_layers=4,
        dropout=0.15,
    )
    checkpoint_path = tmp_path / "nested" / "checkpoints" / "model.pt"
    config_snapshot = {
        "mode": "run",
        "model": {
            "hidden_dim": 8,
            "residue_layers": 3,
            "pair_layers": 4,
            "dropout": 0.15,
        },
        "output": {"model_path": "outputs/model.pt"},
    }

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config_snapshot=config_snapshot,
        residue_dim=10,
        pair_dim=5,
        hidden_dim=8,
        target_dim=3,
        residue_layers=3,
        pair_layers=4,
        dropout=0.15,
    )

    loaded = load_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    assert loaded.residue_dim == 10
    assert loaded.pair_dim == 5
    assert loaded.hidden_dim == 8
    assert loaded.residue_layers == 3
    assert loaded.pair_layers == 4
    assert loaded.dropout == 0.15
    assert loaded.target_dim == 3
    assert loaded.config == config_snapshot
    assert loaded.state_dict.keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict[name], tensor)
