from __future__ import annotations

import numpy as np

from zoo.adapters.synth import make_zoo
from zoo.bench.hot_swap import measure_swap_distribution, swap_cost_ms
from zoo.bench.win_rate import simulated_accuracy_matrix


def test_swap_cost_positive() -> None:
    zoo = make_zoo(n=2)
    t = swap_cost_ms(zoo[0])
    assert t >= 0


def test_swap_distribution_returns_n_samples() -> None:
    zoo = make_zoo(n=3)
    samples = measure_swap_distribution(zoo, reps=10)
    assert len(samples) == 10
    assert all(s >= 0 for s in samples)


def test_accuracy_diagonal_dominates() -> None:
    zoo = make_zoo(n=4)
    acc = simulated_accuracy_matrix(zoo)
    n = acc.shape[0]
    diag = float(np.mean(np.diag(acc)))
    mask = ~np.eye(n, dtype=bool)
    off = float(acc[mask].mean())
    # diagonal (own task) should beat off-diagonal (other tasks)
    assert diag > off + 0.1


def test_accuracy_matrix_bounded() -> None:
    zoo = make_zoo(n=3)
    acc = simulated_accuracy_matrix(zoo)
    assert (acc >= 0).all()
    assert (acc <= 1).all()
