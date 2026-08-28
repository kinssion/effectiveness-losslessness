from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .midi_events import TICKS_PER_BAR, DeclaredEvent, canonical_event_order


@dataclass(frozen=True, slots=True)
class EventWindow:
    start_bar: int
    end_bar: int
    events: tuple[DeclaredEvent, ...]

    @property
    def declared_fact_count(self) -> int:
        return len(self.events)


def window_events(
    events: Sequence[DeclaredEvent],
    *,
    bars: int = 8,
    stride_bars: int = 8,
    maximum_events: int = 2048,
) -> tuple[EventWindow, ...]:
    if bars <= 0 or stride_bars <= 0:
        raise ValueError("bars and stride_bars must be positive")
    ordered = canonical_event_order(events)
    if not ordered:
        return ()
    last_bar = max(event.onset_ticks // TICKS_PER_BAR for event in ordered)
    windows: list[EventWindow] = []
    for start_bar in range(0, last_bar + 1, stride_bars):
        start = start_bar * TICKS_PER_BAR
        end = (start_bar + bars) * TICKS_PER_BAR
        selected = tuple(
            DeclaredEvent(
                event.event_type,
                event.onset_ticks - start,
                event.duration_ticks,
                event.pitch,
            )
            for event in ordered
            if start <= event.onset_ticks < end
        )
        if not selected:
            continue
        if len(selected) > maximum_events:
            raise ValueError(
                f"window {start_bar}:{start_bar + bars} has {len(selected)} events; "
                f"limit is {maximum_events}"
            )
        windows.append(EventWindow(start_bar, start_bar + bars, selected))
    return tuple(windows)
