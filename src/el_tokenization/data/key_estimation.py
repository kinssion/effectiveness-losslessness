from __future__ import annotations

from dataclasses import dataclass


def relative_major_frame_pc(tonic_pc: int, mode: str) -> int:
    mode = mode.lower()
    if mode == "major":
        return tonic_pc % 12
    if mode == "minor":
        return (tonic_pc + 3) % 12
    raise ValueError(f"unsupported mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class AnalyzerEstimate:
    tonic_pc: int
    mode: str
    correlation: float
    distinct_frame_margin: float

    @property
    def frame_pc(self) -> int:
        return relative_major_frame_pc(self.tonic_pc, self.mode)


def high_confidence_acceptance(
    first: AnalyzerEstimate,
    second: AnalyzerEstimate,
    *,
    minimum_correlation: float = 0.6,
    minimum_distinct_frame_margin: float = 0.05,
) -> bool:
    return (
        first.frame_pc == second.frame_pc
        and first.correlation >= minimum_correlation
        and second.correlation >= minimum_correlation
        and first.distinct_frame_margin >= minimum_distinct_frame_margin
        and second.distinct_frame_margin >= minimum_distinct_frame_margin
    )


def canonical_shift(frame_pc: int) -> int:
    unsigned = (-frame_pc) % 12
    return unsigned if unsigned <= 6 else unsigned - 12
