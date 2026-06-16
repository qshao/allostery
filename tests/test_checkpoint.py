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
    assert loaded.metadata == {}
    assert loaded.state_dict.keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict[name], tensor)


def test_checkpoint_round_trips_model_family(tmp_path: Path) -> None:
    from allostery.io.checkpoint import load_checkpoint, save_checkpoint
    from allostery.models.cri import CRILatentInteractionModel

    model = CRILatentInteractionModel(state_dim=6, hidden_dim=8, edge_types=2)
    checkpoint_path = tmp_path / "cri.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config_snapshot={"model": {"family": "cri"}},
        residue_dim=6,
        pair_dim=1,
        hidden_dim=8,
        target_dim=3,
        residue_layers=1,
        pair_layers=2,
        dropout=0.0,
        model_family="cri",
    )

    loaded = load_checkpoint(checkpoint_path)

    assert loaded.model_family == "cri"
    assert loaded.residue_dim == 6
    assert isinstance(loaded.state_dict, dict)



def test_checkpoint_round_trips_min_sequence_separation(tmp_path: Path) -> None:
    from allostery.io.checkpoint import load_checkpoint, save_checkpoint
    from allostery.models.influence import AllostericInfluenceModel

    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=3
    )
    path = tmp_path / 'influence_sep3.pt'

    save_checkpoint(
        path=path,
        model=model,
        config_snapshot={},
        residue_dim=6,
        pair_dim=1,
        hidden_dim=8,
        target_dim=3,
        residue_layers=1,
        pair_layers=1,
        dropout=0.0,
        model_family='influence',
        min_sequence_separation=3,
    )

    ckpt = load_checkpoint(path)
    assert ckpt.min_sequence_separation == 3


def test_checkpoint_defaults_min_sequence_separation_to_one(tmp_path: Path) -> None:
    model = RelationalScoreModel(
        residue_dim=10, pair_dim=5, hidden_dim=8, target_dim=3
    )
    path = tmp_path / 'relational.pt'

    save_checkpoint(
        path=path,
        model=model,
        config_snapshot={},
        residue_dim=10,
        pair_dim=5,
        hidden_dim=8,
        target_dim=3,
    )

    ckpt = load_checkpoint(path)
    assert ckpt.min_sequence_separation == 1


def test_checkpoint_round_trips_metadata(tmp_path: Path) -> None:
    model = RelationalScoreModel(
        residue_dim=10,
        pair_dim=5,
        hidden_dim=8,
        target_dim=3,
        residue_layers=3,
        pair_layers=4,
        dropout=0.15,
    )
    checkpoint_path = tmp_path / "metadata.pt"
    metadata = {"training": {"seed": 7, "validation_fraction": 0.25}}

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config_snapshot={"mode": "train"},
        residue_dim=10,
        pair_dim=5,
        hidden_dim=8,
        target_dim=3,
        residue_layers=3,
        pair_layers=4,
        dropout=0.15,
        metadata=metadata,
    )

    loaded = load_checkpoint(checkpoint_path)

    assert loaded.metadata == metadata
