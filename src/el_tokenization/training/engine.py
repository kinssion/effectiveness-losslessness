from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SyntheticTrainingReceipt:
    steps: int
    initial_loss: float
    final_loss: float
    finite: bool
    scientific_result: bool = False


def synthetic_train(*, steps: int = 100, seed: int = 20260819) -> SyntheticTrainingReceipt:
    """Dependency-light optimizer smoke test, not a miniature paper experiment."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(64, 4))
    target = features @ np.array([0.8, -0.3, 0.2, 0.5])
    weights = np.zeros(4)
    losses: list[float] = []
    for _ in range(steps):
        residual = features @ weights - target
        losses.append(float(np.mean(residual**2)))
        gradient = 2.0 * features.T @ residual / len(features)
        weights -= 0.05 * gradient
    return SyntheticTrainingReceipt(
        steps=steps,
        initial_loss=losses[0],
        final_loss=losses[-1],
        finite=bool(np.isfinite(losses).all()),
    )
