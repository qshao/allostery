from __future__ import annotations

import torch
import pytest

from allostery.models.influence import AllostericInfluenceModel


def test_influence_model_output_shapes() -> None:
    model = AllostericInfluenceModel(state_dim=6, hidden_dim=8, num_encoder_layers=2, dropout=0.0)
    state_features = torch.randn(2, 3, 4, 6)  # [batch=2, time=3, N=4, state_dim=6]

    output = model(state_features)

    assert output['acceleration'].shape == (2, 3, 4, 3)
    assert output['influence_matrix'].shape == (2, 4, 4)


def test_influence_matrix_values_in_unit_interval() -> None:
    model = AllostericInfluenceModel(state_dim=6, hidden_dim=8)
    state_features = torch.randn(1, 3, 5, 6)

    output = model(state_features)
    influence = output['influence_matrix']  # [1, 5, 5]

    # Sigmoid attention → independent per-pair scores in [0, 1]; no row-sum constraint.
    assert (influence >= 0.0).all(), 'influence values must be non-negative'
    assert (influence <= 1.0).all(), 'influence values must be at most 1.0'


def test_influence_matrix_diagonal_is_zero() -> None:
    model = AllostericInfluenceModel(state_dim=6, hidden_dim=8)
    state_features = torch.randn(1, 4, 6, 6)

    output = model(state_features)
    influence = output['influence_matrix'].squeeze(0)  # [6, 6]

    diag = torch.diagonal(influence)
    torch.testing.assert_close(diag, torch.zeros(6))


def test_influence_model_rejects_wrong_input_shape() -> None:
    model = AllostericInfluenceModel(state_dim=6, hidden_dim=8)

    with pytest.raises(ValueError, match='batch, time, N, state_dim'):
        model(torch.randn(2, 4, 6))  # 3D — missing batch


def test_influence_model_single_residue_pair() -> None:
    model = AllostericInfluenceModel(state_dim=6, hidden_dim=8, num_encoder_layers=1)
    state_features = torch.randn(1, 3, 2, 6)  # two residues

    output = model(state_features)

    assert output['acceleration'].shape == (1, 3, 2, 3)
    # With 2 residues and diagonal masked, off-diagonal values are sigmoid scores in (0, 1).
    influence = output['influence_matrix'].squeeze(0)  # [2, 2]
    assert 0.0 <= influence[0, 1].item() <= 1.0
    assert 0.0 <= influence[1, 0].item() <= 1.0


def test_influence_model_rejects_nonpositive_encoder_layers() -> None:
    with pytest.raises(ValueError, match='num_encoder_layers'):
        AllostericInfluenceModel(state_dim=6, hidden_dim=8, num_encoder_layers=0)


def test_forward_single_residue_is_finite() -> None:
    model = AllostericInfluenceModel(state_dim=6, hidden_dim=8, num_encoder_layers=1)
    state = torch.randn(2, 4, 1, 6)  # batch=2, time=4, N=1, state_dim=6
    out = model(state)
    assert out['acceleration'].shape == (2, 4, 1, 3)
    assert out['influence_matrix'].shape == (2, 1, 1)
    assert torch.isfinite(out['acceleration']).all()
    assert torch.isfinite(out['influence_matrix']).all()


def test_chunked_aggregation_matches_dense() -> None:
    torch.manual_seed(0)
    dense = AllostericInfluenceModel(state_dim=6, hidden_dim=8, num_encoder_layers=2)
    chunked = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=2, residue_chunk_size=2
    )
    chunked.load_state_dict(dense.state_dict())
    dense.eval()
    chunked.eval()
    state = torch.randn(2, 4, 5, 6)  # N=5, chunk=2 -> chunks of 2,2,1
    with torch.no_grad():
        a = dense(state)
        b = chunked(state)
    torch.testing.assert_close(a['acceleration'], b['acceleration'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a['influence_matrix'], b['influence_matrix'], atol=1e-6, rtol=1e-5)


def test_influence_model_masks_pairs_within_sequence_separation() -> None:
    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=3
    )
    state_features = torch.randn(1, 3, 8, 6)  # 8 residues

    output = model(state_features)
    A = output['influence_matrix'].squeeze(0)  # [8, 8]

    for i in range(8):
        for j in range(8):
            if abs(i - j) < 3:
                assert A[i, j].item() == pytest.approx(0.0, abs=1e-6), (
                    f'A[{i},{j}] = {A[i,j].item():.6f} should be 0 '
                    f'(sequence separation {abs(i-j)} < 3)'
                )


def test_influence_model_valid_values_with_separation() -> None:
    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=3
    )
    state_features = torch.randn(1, 3, 8, 6)

    output = model(state_features)
    A = output['influence_matrix'].squeeze(0)  # [8, 8]

    # Masked entries must be exactly 0 (sigmoid(-inf) = 0).
    for i in range(8):
        for j in range(8):
            if abs(i - j) < 3:
                assert A[i, j].item() == pytest.approx(0.0, abs=1e-6)
    # Unmasked entries must be sigmoid scores in [0, 1].
    assert (A >= 0.0).all()
    assert (A <= 1.0).all()


def test_influence_model_separation_one_is_diagonal_only() -> None:
    model_sep1 = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=1
    )
    model_diag = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=1
    )
    model_diag.load_state_dict(model_sep1.state_dict())

    state = torch.randn(1, 3, 5, 6)
    with torch.no_grad():
        A1 = model_sep1(state)['influence_matrix']
        A2 = model_diag(state)['influence_matrix']

    torch.testing.assert_close(A1, A2)
    diag = torch.diagonal(A1.squeeze(0))
    torch.testing.assert_close(diag, torch.zeros(5))


def test_influence_model_rejects_separation_less_than_one() -> None:
    with pytest.raises(ValueError, match='min_sequence_separation'):
        AllostericInfluenceModel(state_dim=6, hidden_dim=8, min_sequence_separation=0)


def test_influence_model_rejects_separation_too_large_for_protein() -> None:
    model = AllostericInfluenceModel(
        state_dim=6, hidden_dim=8, num_encoder_layers=1, min_sequence_separation=5
    )
    state_features = torch.randn(1, 3, 4, 6)  # only 4 residues, sep=5 would mask everything

    with pytest.raises(ValueError, match='no valid pairs'):
        model(state_features)
