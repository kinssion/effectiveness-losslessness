from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

TICKS_PER_BEAT = 96
BEATS_PER_BAR = 4
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR
EventType = Literal["NOTE", "REST"]


@dataclass(frozen=True, slots=True)
class DeclaredEvent:
    """One declared score-level fact used by the paper ledger."""

    event_type: EventType
    onset_ticks: int
    duration_ticks: int
    pitch: int | None = None

    def __post_init__(self) -> None:
        if self.onset_ticks < 0:
            raise ValueError("onset_ticks must be non-negative")
        if self.duration_ticks <= 0:
            raise ValueError("duration_ticks must be positive")
        if self.event_type == "NOTE":
            if self.pitch is None or not 0 <= self.pitch <= 127:
                raise ValueError("NOTE pitch must be an integer in [0, 127]")
        elif self.event_type == "REST":
            if self.pitch is not None:
                raise ValueError("REST must not carry pitch")
        else:
            raise ValueError(f"unsupported event type: {self.event_type!r}")

    @property
    def end_ticks(self) -> int:
        return self.onset_ticks + self.duration_ticks

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DeclaredEvent:
        pitch = value.get("pitch")
        return cls(
            event_type=str(value["event_type"]),  # type: ignore[arg-type]
            onset_ticks=int(str(value["onset_ticks"])),
            duration_ticks=int(str(value["duration_ticks"])),
            pitch=None if pitch is None else int(str(pitch)),
        )


def canonical_event_order(events: Iterable[DeclaredEvent]) -> tuple[DeclaredEvent, ...]:
    """Sort by onset and ascending pitch for simultaneous notes.

    REST uses a sentinel below MIDI pitch so ties remain deterministic.  No chord,
    voice, melody, or onset-group identity is introduced.
    """

    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.onset_ticks,
                -1 if event.pitch is None else event.pitch,
                event.duration_ticks,
                event.event_type,
            ),
        )
    )


def insert_explicit_rests(notes: Sequence[DeclaredEvent]) -> tuple[DeclaredEvent, ...]:
    """Insert maximal silent intervals between sounding regions.

    This helper is intentionally score-level: overlap is resolved from onset and
    duration, and no track or instrument role is inferred.
    """

    ordered = canonical_event_order(notes)
    if any(event.event_type != "NOTE" for event in ordered):
        raise ValueError("insert_explicit_rests expects NOTE events only")
    if not ordered:
        return ()
    result: list[DeclaredEvent] = []
    sounding_until = 0
    for event in ordered:
        if event.onset_ticks > sounding_until:
            result.append(
                DeclaredEvent("REST", sounding_until, event.onset_ticks - sounding_until)
            )
        result.append(event)
        sounding_until = max(sounding_until, event.end_ticks)
    return canonical_event_order(result)


def events_to_json(events: Sequence[DeclaredEvent]) -> str:
    return json.dumps(
        [event.to_dict() for event in events], sort_keys=True, separators=(",", ":")
    )


def events_from_json(payload: str) -> tuple[DeclaredEvent, ...]:
    value = json.loads(payload)
    if not isinstance(value, list):
        raise TypeError("event payload must be a list")
    return tuple(DeclaredEvent.from_dict(row) for row in value)


def read_midi_notes(
    path: Path, *, target_ticks_per_beat: int = TICKS_PER_BEAT
) -> tuple[DeclaredEvent, ...]:
    """Read non-drum MIDI notes and quantize ticks deterministically.

    Drum channel 10 (zero-based channel 9) is excluded.  Dangling note-ons are a
    hard error so a silent parse failure cannot change the declared fact surface.
    """

    import mido

    midi = mido.MidiFile(path)
    scale = target_ticks_per_beat / int(midi.ticks_per_beat)
    active: dict[tuple[int, int], list[int]] = {}
    notes: list[DeclaredEvent] = []
    absolute = 0
    for message in mido.merge_tracks(midi.tracks):
        absolute += int(message.time)
        if message.type not in {"note_on", "note_off"}:
            continue
        channel = int(getattr(message, "channel", 0))
        if channel == 9:
            continue
        pitch = int(message.note)
        key = (channel, pitch)
        is_on = message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0
        if is_on:
            active.setdefault(key, []).append(absolute)
            continue
        starts = active.get(key)
        if not starts:
            continue
        start = starts.pop(0)
        onset = round(start * scale)
        end = round(absolute * scale)
        notes.append(DeclaredEvent("NOTE", onset, max(1, end - onset), pitch))
    dangling = sum(len(starts) for starts in active.values())
    if dangling:
        raise ValueError(f"MIDI contains {dangling} unmatched note-on messages")
    return canonical_event_order(notes)
