from __future__ import annotations

from allostery.training.runtime import split_samples


def test_split_samples_is_deterministic_and_respects_validation_fraction() -> None:
    samples = list(range(10))

    train_a, val_a = split_samples(samples, validation_fraction=0.3, seed=11)
    train_b, val_b = split_samples(samples, validation_fraction=0.3, seed=11)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 7
    assert len(val_a) == 3
    assert sorted(train_a + val_a) == samples


def test_split_samples_disables_validation_when_fraction_is_zero() -> None:
    samples = list(range(4))

    train, val = split_samples(samples, validation_fraction=0.0, seed=1)

    assert train == samples
    assert val == []
