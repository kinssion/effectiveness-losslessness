from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    path: str
    bits_per_declared_fact: float
    completed_target_exposure: int


def select_checkpoint(
    candidates: Iterable[ValidationCandidate], *, tie_tolerance: float = 1e-12
) -> ValidationCandidate:
    materialized = list(candidates)
    if not materialized:
        raise ValueError("no validation candidates")
    best_value = min(candidate.bits_per_declared_fact for candidate in materialized)
    tied = [
        candidate
        for candidate in materialized
        if abs(candidate.bits_per_declared_fact - best_value) <= tie_tolerance
    ]
    return min(tied, key=lambda candidate: candidate.completed_target_exposure)
