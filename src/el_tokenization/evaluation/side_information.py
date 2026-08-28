from __future__ import annotations

import math


def fixed_length_bits(support_size: int, count: int = 1) -> float:
    if support_size <= 0 or count < 0:
        raise ValueError("support_size must be positive and count non-negative")
    return float(math.ceil(math.log2(support_size)) * count)


def tonal_shift_bits(*, windows: int, source_songs: int, charge: str = "per_window") -> float:
    if charge == "per_window":
        return fixed_length_bits(12, windows)
    if charge == "per_source_song":
        return fixed_length_bits(12, source_songs)
    raise ValueError("charge must be per_window or per_source_song")
