"""Hot-swap timing simulation.

The real benchmark times peft's `model.set_adapter(name)` under load. The
synthetic stand-in here just records the simulated cost of building the
effective B @ A per target and "applying" it, so the chart layer has
something to plot. Same shape as the real measurement (ms per swap).
"""

from __future__ import annotations

import time

from ..types import AdapterWeights


def swap_cost_ms(a: AdapterWeights) -> float:
    t0 = time.perf_counter()
    _ = a.effective_delta()  # force the B @ A multiply
    return (time.perf_counter() - t0) * 1000.0


def measure_swap_distribution(adapters: list[AdapterWeights], reps: int = 50) -> list[float]:
    """Each swap = pick a random adapter, apply it, time it."""
    import random

    rng = random.Random(11)
    samples: list[float] = []
    for _ in range(reps):
        a = rng.choice(adapters)
        samples.append(swap_cost_ms(a))
    return samples
