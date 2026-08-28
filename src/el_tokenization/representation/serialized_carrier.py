from __future__ import annotations

from collections.abc import Sequence

from ..data.midi_events import TICKS_PER_BAR, DeclaredEvent, canonical_event_order


def serialize_events(events: Sequence[DeclaredEvent]) -> tuple[str, ...]:
    """Lossless BAR/POSITION/pitch-or-REST/duration serializer used by J."""

    tokens: list[str] = []
    current_bar = 0
    for event in canonical_event_order(events):
        bar, position = divmod(event.onset_ticks, TICKS_PER_BAR)
        if bar < current_bar:
            raise ValueError("events are not monotone")
        tokens.extend("BAR_ADVANCE" for _ in range(bar - current_bar))
        current_bar = bar
        tokens.append(f"POSITION_{position}")
        tokens.append("REST" if event.pitch is None else f"PITCH_{event.pitch}")
        duration_bars, remainder = divmod(event.duration_ticks, TICKS_PER_BAR)
        tokens.append(f"DURATION_BAR_{duration_bars}")
        tokens.append(f"DURATION_REMAINDER_{remainder}")
    return tuple(tokens)


def deserialize_events(tokens: Sequence[str]) -> tuple[DeclaredEvent, ...]:
    events: list[DeclaredEvent] = []
    current_bar = 0
    index = 0
    while index < len(tokens):
        while index < len(tokens) and tokens[index] == "BAR_ADVANCE":
            current_bar += 1
            index += 1
        if index + 3 >= len(tokens):
            raise ValueError("truncated serialized event")
        position_token, content_token, bars_token, remainder_token = tokens[index : index + 4]
        if not position_token.startswith("POSITION_"):
            raise ValueError("expected POSITION token")
        position = int(position_token.removeprefix("POSITION_"))
        if not 0 <= position < TICKS_PER_BAR:
            raise ValueError("POSITION leaves one-bar support")
        if content_token == "REST":
            event_type = "REST"
            pitch = None
        elif content_token.startswith("PITCH_"):
            event_type = "NOTE"
            pitch = int(content_token.removeprefix("PITCH_"))
        else:
            raise ValueError("expected PITCH or REST token")
        if not bars_token.startswith("DURATION_BAR_"):
            raise ValueError("expected DURATION_BAR token")
        if not remainder_token.startswith("DURATION_REMAINDER_"):
            raise ValueError("expected DURATION_REMAINDER token")
        duration = int(bars_token.removeprefix("DURATION_BAR_")) * TICKS_PER_BAR + int(
            remainder_token.removeprefix("DURATION_REMAINDER_")
        )
        events.append(
            DeclaredEvent(event_type, current_bar * TICKS_PER_BAR + position, duration, pitch)  # type: ignore[arg-type]
        )
        index += 4
    return canonical_event_order(events)
