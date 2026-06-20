from __future__ import annotations

import numpy as np

from allostery.validation.metrics import evaluate_scores


def _pair(i: int, j: int, score: float) -> dict:
    return {
        "residue_i": {"index": i, "chain_id": "A", "residue_number": i + 1, "name": "GLY"},
        "residue_j": {"index": j, "chain_id": "A", "residue_number": j + 1, "name": "GLY"},
        "score": score,
    }


def _truth(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        matrix[i, j] = True
        matrix[j, i] = True
    return matrix


def test_perfect_ranking_scores_one() -> None:
    # n=5, sep>=2 candidate pairs: (0,2)(0,3)(0,4)(1,3)(1,4)(2,4); truths (0,2)(1,3)
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [
        _pair(0, 2, 9.0), _pair(1, 3, 8.0),    # true pairs ranked highest
        _pair(0, 3, 1.0), _pair(0, 4, 0.5), _pair(1, 4, 0.4), _pair(2, 4, 0.3),
    ]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.n_true == 2
    assert m.n_pairs == 6
    assert m.roc_auc == 1.0
    assert m.pr_auc == 1.0
    assert m.precision_at_k == 1.0
    assert m.recall_at_k == 1.0


def test_reversed_ranking_scores_zero_auc() -> None:
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [
        _pair(0, 2, 0.0), _pair(1, 3, 0.1),    # true pairs ranked lowest
        _pair(0, 3, 9.0), _pair(0, 4, 8.0), _pair(1, 4, 7.0), _pair(2, 4, 6.0),
    ]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.roc_auc == 0.0
    assert m.precision_at_k == 0.0


def test_all_tied_scores_give_half_auc() -> None:
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [
        _pair(0, 2, 1.0), _pair(1, 3, 1.0), _pair(0, 3, 1.0),
        _pair(0, 4, 1.0), _pair(1, 4, 1.0), _pair(2, 4, 1.0),
    ]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.roc_auc == 0.5


def test_missing_pairs_rank_last() -> None:
    # scorer omits the true pair (1,3); it must be treated as worst-ranked
    truth = _truth(5, [(0, 2), (1, 3)])
    scores = [_pair(0, 2, 9.0), _pair(0, 3, 8.0), _pair(0, 4, 7.0), _pair(2, 4, 6.0)]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.n_pairs == 6                      # full candidate universe, not just provided
    assert m.precision_at_k <= 0.5             # only (0,2) can be recovered in top-2


def test_no_positive_pairs_returns_sentinel() -> None:
    truth = _truth(5, [])                       # no true edges
    scores = [_pair(0, 2, 1.0), _pair(1, 3, 0.5)]
    m = evaluate_scores(scores, truth, min_sequence_separation=2)
    assert m.n_true == 0
    assert m.roc_auc == 0.5
    assert m.pr_auc == 0.0
