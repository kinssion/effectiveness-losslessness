from __future__ import annotations

import math
from dataclasses import dataclass

from ..data.midi_events import TICKS_PER_BAR, TICKS_PER_BEAT


@dataclass(frozen=True, slots=True)
class MusicalTimeCoordinate:
    absolute_bar: int
    within_bar_tick: int
    within_beat_tick: int
    beat_index: int

    def to_onset_ticks(self) -> int:
        return self.absolute_bar * TICKS_PER_BAR + self.within_bar_tick


def musical_time_coordinate(onset_ticks: int) -> MusicalTimeCoordinate:
    if onset_ticks < 0:
        raise ValueError("onset must be non-negative")
    bar, within_bar = divmod(onset_ticks, TICKS_PER_BAR)
    beat, within_beat = divmod(within_bar, TICKS_PER_BEAT)
    return MusicalTimeCoordinate(bar, within_bar, within_beat, beat)


def cyclic_phase(index: int, period: int) -> tuple[float, float]:
    if period <= 0:
        raise ValueError("period must be positive")
    angle = 2.0 * math.pi * (index % period) / period
    return math.sin(angle), math.cos(angle)


def directed_multiscale_time(onset_ticks: int) -> dict[str, object]:
    coordinate = musical_time_coordinate(onset_ticks)
    sixteenth = onset_ticks // (TICKS_PER_BEAT // 4)
    return {
        "absolute_bar": coordinate.absolute_bar,
        "within_bar_tick": coordinate.within_bar_tick,
        "beat_phase": cyclic_phase(sixteenth, 4),
        "bar_phase": cyclic_phase(sixteenth, 16),
        "four_bar_phase": cyclic_phase(sixteenth, 64),
    }
