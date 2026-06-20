from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScoreMetrics:
    roc_auc: float
    pr_auc: float
    precision_at_k: float
    recall_at_k: float
    n_true: int
    n_pairs: int


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Ranks (1-based) with ties assigned their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # average ranks within tie groups
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                avg = ranks[order[start:end]].mean()
                ranks[order[start:end]] = avg
            start = end
    return ranks


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rank_average(scores)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(np.float64)
    cumulative_tp = np.cumsum(labels_sorted)
    precision = cumulative_tp / (np.arange(len(labels_sorted)) + 1.0)
    return float((precision * labels_sorted).sum() / n_pos)


def evaluate_scores(
    pair_scores: list[dict],
    coupling_matrix: np.ndarray,
    *,
    min_sequence_separation: int = 2,
) -> ScoreMetrics:
    n = coupling_matrix.shape[0]
    score_map: dict[tuple[int, int], float] = {}
    for item in pair_scores:
        i = int(item["residue_i"]["index"])
        j = int(item["residue_j"]["index"])
        score_map[(min(i, j), max(i, j))] = float(item["score"])

    scores: list[float] = []
    labels: list[bool] = []
    for i in range(n):
        for j in range(i + min_sequence_separation, n):
            scores.append(score_map.get((i, j), float("-inf")))
            labels.append(bool(coupling_matrix[i, j]))

    score_array = np.array(scores, dtype=np.float64)
    label_array = np.array(labels, dtype=bool)
    n_true = int(label_array.sum())
    n_pairs = int(len(label_array))

    roc = _roc_auc(score_array, label_array)
    ap = _average_precision(score_array, label_array)

    if n_true == 0:
        precision_at_k = 0.0
        recall_at_k = 0.0
    else:
        top = np.argsort(-score_array, kind="mergesort")[:n_true]
        hits = int(label_array[top].sum())
        precision_at_k = hits / float(n_true)
        recall_at_k = hits / float(n_true)

    return ScoreMetrics(
        roc_auc=roc,
        pr_auc=ap,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        n_true=n_true,
        n_pairs=n_pairs,
    )


__all__ = ["ScoreMetrics", "evaluate_scores"]
