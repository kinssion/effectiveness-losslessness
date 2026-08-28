from __future__ import annotations

from collections.abc import Sequence

from ..data.midi_events import DeclaredEvent, events_from_json, events_to_json


def encode_exact_events(events: Sequence[DeclaredEvent]) -> bytes:
    return events_to_json(events).encode("utf-8")


def decode_exact_events(payload: bytes) -> tuple[DeclaredEvent, ...]:
    return events_from_json(payload.decode("utf-8"))
