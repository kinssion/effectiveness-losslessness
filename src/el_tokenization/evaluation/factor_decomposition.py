from __future__ import annotations

from .predictive_ledger import PredictiveLedgerEntry


def factor_sum_equals_total(entry: PredictiveLedgerEntry, *, tolerance: float = 1e-12) -> bool:
    return (
        abs(sum(entry.factor_bits_per_declared_fact().values()) - entry.bits_per_declared_fact)
        <= tolerance
    )


def paired_delta(left: PredictiveLedgerEntry, right: PredictiveLedgerEntry) -> dict[str, float]:
    left_factors = left.factor_bits_per_declared_fact()
    right_factors = right.factor_bits_per_declared_fact()
    return {factor: left_factors[factor] - right_factors[factor] for factor in left_factors}
