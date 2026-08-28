from __future__ import annotations

import math


def chromatic_phase(pitch_class: int) -> tuple[float, float]:
    if not 0 <= pitch_class < 12:
        raise ValueError("pitch class must be in [0, 11]")
    angle = 2.0 * math.pi * pitch_class / 12.0
    return math.sin(angle), math.cos(angle)


def fifths_index(pitch_class: int) -> int:
    if not 0 <= pitch_class < 12:
        raise ValueError("pitch class must be in [0, 11]")
    return (7 * pitch_class) % 12


def fifths_phase(pitch_class: int) -> tuple[float, float]:
    return chromatic_phase(fifths_index(pitch_class))


def phase_is_injective(function: object, *, tolerance: float = 1e-12) -> bool:
    values = [function(index) for index in range(12)]  # type: ignore[operator]
    return all(
        math.dist(values[left], values[right]) > tolerance
        for left in range(12)
        for right in range(left + 1, 12)
    )
