"""Per-task win-rate simulator.

In the real setting each adapter would be evaluated on every task and we
report a (task x adapter) accuracy matrix. The dominant diagonal
(adapter_i wins on task_i) is the desired result; off-diagonal hits show
positive transfer (adapter trained on one task helps another).

We simulate that pattern: diagonal accuracy is high (uniform 0.78-0.92);
off-diagonal accuracy is high when adapter cosine to the diagonal adapter
is high (positive transfer), low otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..adapters.similarity import pairwise_matrix
from ..types import AdapterWeights


def simulated_accuracy_matrix(
    adapters: list[AdapterWeights], seed: int = 23
) -> NDArray[np.float64]:
    """Returns shape (n_tasks, n_adapters). One task per adapter; entry [i, j]
    = how well adapter j does on task i.
    """
    rng = np.random.default_rng(seed)
    n = len(adapters)
    sims = pairwise_matrix(adapters)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0.78 + 0.14 * rng.random()  # diagonal: 0.78-0.92
            else:
                base = 0.45  # random-guess-ish baseline
                transfer = 0.35 * max(0.0, sims[i, j])
                noise = 0.03 * rng.standard_normal()
                mat[i, j] = float(np.clip(base + transfer + noise, 0.0, 1.0))
    return mat
