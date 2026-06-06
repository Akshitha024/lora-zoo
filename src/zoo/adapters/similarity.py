"""Pairwise similarity between adapters.

We compute similarity on the effective deltas (B @ A scaled), flattened.
Cosine because it normalizes for scale (adapters of different ranks +
alphas can still be compared).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import AdapterWeights


def _flat_delta(a: AdapterWeights) -> NDArray[np.float64]:
    deltas = a.effective_delta()
    parts = [v.reshape(-1) for v in deltas.values()]
    return np.concatenate(parts) if parts else np.zeros(0)


def cosine(a: AdapterWeights, b: AdapterWeights) -> float:
    fa = _flat_delta(a)
    fb = _flat_delta(b)
    if fa.size == 0 or fb.size == 0:
        return 0.0
    na = float(np.linalg.norm(fa))
    nb = float(np.linalg.norm(fb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(fa, fb) / (na * nb))


def pairwise_matrix(adapters: list[AdapterWeights]) -> NDArray[np.float64]:
    n = len(adapters)
    mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            c = cosine(adapters[i], adapters[j])
            mat[i, j] = c
            mat[j, i] = c
    return mat
