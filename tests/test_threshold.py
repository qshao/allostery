from __future__ import annotations

import pytest

from allostery.network import detect_threshold, format_score_histogram


def test_detect_threshold_step_function() -> None:
    # Clear drop between rank 3 and rank 4 — knee must NOT be at rank 1 (highest score)
    scores = [0.9, 0.85, 0.8, 0.1, 0.05, 0.02]
    t, rank = detect_threshold(scores)
    assert 2 <= rank <= 4  # must NOT be rank 1 (highest score)


def test_detect_threshold_returns_score_at_knee() -> None:
    scores = [0.9, 0.85, 0.8, 0.1, 0.05, 0.02]
    t, rank = detect_threshold(scores)
    sorted_desc = sorted(scores, reverse=True)
    assert t == pytest.approx(sorted_desc[rank - 1])


def test_detect_threshold_uniform_returns_first() -> None:
    scores = [0.5, 0.5, 0.5, 0.5]
    t, rank = detect_threshold(scores)
    assert rank == 1
    assert t == pytest.approx(0.5)


def test_detect_threshold_fewer_than_3_returns_first() -> None:
    scores = [0.9, 0.8]
    t, rank = detect_threshold(scores)
    assert rank == 1
    assert t == pytest.approx(0.9)


def test_detect_threshold_single_score() -> None:
    scores = [0.7]
    t, rank = detect_threshold(scores)
    assert rank == 1
    assert t == pytest.approx(0.7)


def test_format_score_histogram_bin_counts_sum_to_total() -> None:
    scores = [i / 10.0 for i in range(11)]  # 11 scores 0.0–1.0
    hist = format_score_histogram(scores, bins=5)
    bin_lines = [l for l in hist.splitlines() if "|" in l]
    counts = [int(l.split("|")[2].split()[0]) for l in bin_lines]
    assert sum(counts) == len(scores)


def test_format_score_histogram_contains_header() -> None:
    scores = [0.1, 0.5, 0.9]
    hist = format_score_histogram(scores, bins=3)
    assert "Score Distribution" in hist
    assert "3 pairs" in hist


def test_format_score_histogram_threshold_marker_present() -> None:
    scores = [0.9, 0.8, 0.7, 0.1, 0.05]
    hist = format_score_histogram(scores, bins=5, threshold_rank=3)
    assert "▶ threshold" in hist


def test_format_score_histogram_no_marker_when_rank_none() -> None:
    scores = [0.9, 0.8, 0.7]
    hist = format_score_histogram(scores, bins=3)
    assert "▶ threshold" not in hist


def test_format_score_histogram_uniform_scores() -> None:
    scores = [0.5, 0.5, 0.5]
    hist = format_score_histogram(scores, bins=5)
    assert "Score Distribution" in hist
    assert "all scores equal" in hist


def test_detect_threshold_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        detect_threshold([])


def test_format_score_histogram_empty() -> None:
    hist = format_score_histogram([])
    assert "Score Distribution" in hist
    assert "0 pairs" in hist
