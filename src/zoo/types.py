"""Core types: adapter metadata + benchmark rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AdapterMeta:
    name: str  # e.g. "sql", "summarization"
    base_model: str  # e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    rank: int
    alpha: int
    target_modules: tuple[str, ...]


@dataclass
class AdapterWeights:
    """LoRA delta: per-target {A: (r, in), B: (out, r)}."""

    meta: AdapterMeta
    A: dict[str, NDArray[np.float64]]
    B: dict[str, NDArray[np.float64]]

    def effective_delta(self) -> dict[str, NDArray[np.float64]]:
        """B @ A per target, scaled by alpha/rank (LoRA scaling)."""
        scale = self.meta.alpha / self.meta.rank
        return {k: scale * (self.B[k] @ self.A[k]) for k in self.A}


@dataclass
class TaskScore:
    task: str
    adapter: str
    accuracy: float
    n: int
    extras: dict[str, float]
