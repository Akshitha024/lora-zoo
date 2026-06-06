"""Synthetic adapter generation.

For CI and demonstration we generate LoRA-shaped A/B matrices directly,
labeled by task. Real adapter training happens via peft.LoraConfig + a
Trainer; that path is wired in `adapters.real.train` but skipped in CI.

We deliberately give the synthetic adapters partially overlapping support
on a small "feature" subspace, so the inter-adapter cosine matrix has
both bright (related tasks) and dark (unrelated tasks) cells.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import AdapterMeta, AdapterWeights

_TARGETS = ("q_proj", "v_proj")


def make(name: str, d_in: int, d_out: int, rank: int = 8, alpha: int = 16,
         seed: int = 7, shared_subspace_dim: int = 4) -> AdapterWeights:
    """Two adapters trained on related tasks share `shared_subspace_dim` columns
    of A (so their effective deltas are partially aligned in that subspace).
    """
    rng = np.random.default_rng(seed)
    meta = AdapterMeta(name=name, base_model="synthetic", rank=rank, alpha=alpha,
                       target_modules=_TARGETS)
    A: dict[str, NDArray[np.float64]] = {}
    B: dict[str, NDArray[np.float64]] = {}
    for t in _TARGETS:
        # the shared subspace uses a fixed seed (per target) so all adapters share it
        shared_rng = np.random.default_rng(hash(t) % (2**32))
        shared_A = shared_rng.standard_normal((shared_subspace_dim, d_in)) * 0.02
        unique_A = rng.standard_normal((rank - shared_subspace_dim, d_in)) * 0.02
        A[t] = np.vstack([shared_A, unique_A]).astype(np.float64)
        B[t] = rng.standard_normal((d_out, rank)).astype(np.float64) * 0.02
    return AdapterWeights(meta=meta, A=A, B=B)


def make_zoo(d_in: int = 64, d_out: int = 64, rank: int = 8, n: int = 6) -> list[AdapterWeights]:
    return [
        make(name=f"task_{i}", d_in=d_in, d_out=d_out, rank=rank, seed=100 + i)
        for i in range(n)
    ]
