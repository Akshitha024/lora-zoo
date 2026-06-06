"""Orchestrator: build the zoo, compute similarity, run swap timing,
generate accuracy matrix, write JSON artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from .adapters.similarity import pairwise_matrix
from .adapters.synth import make_zoo
from .bench.hot_swap import measure_swap_distribution
from .bench.win_rate import simulated_accuracy_matrix


def sweep(out_dir: Path, n: int = 6, rank: int = 8) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapters = make_zoo(n=n, rank=rank)
    names = [a.meta.name for a in adapters]

    sims = pairwise_matrix(adapters)
    swap_ms = measure_swap_distribution(adapters, reps=80)
    acc = simulated_accuracy_matrix(adapters)

    diagonal_mean = float(np.mean(np.diag(acc)))
    off_mean = float(np.mean(acc - np.diag(np.diag(acc))) * (n / max(1, n - 1)))
    # the off_mean above subtracts the diagonal but the denominator is wrong;
    # compute it properly:
    mask = ~np.eye(n, dtype=bool)
    off_mean = float(acc[mask].mean())

    artifact = {
        "n_adapters": n,
        "rank": rank,
        "names": names,
        "pairwise_cosine": sims.tolist(),
        "accuracy_matrix": acc.tolist(),
        "swap_ms_samples": swap_ms,
        "swap_ms_p50": float(np.median(swap_ms)),
        "swap_ms_p99": float(np.quantile(swap_ms, 0.99)),
        "diagonal_accuracy_mean": diagonal_mean,
        "off_diagonal_accuracy_mean": off_mean,
    }
    (out_dir / "sweep.json").write_text(json.dumps(artifact))
    logger.info("wrote {}/sweep.json", out_dir)
    return artifact
