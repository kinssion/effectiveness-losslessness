from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from .midi_events import DeclaredEvent, canonical_event_order


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_event_hash(events: Sequence[DeclaredEvent]) -> str:
    return _hash([event.to_dict() for event in canonical_event_order(events)])


def transposition_invariant_hash(events: Sequence[DeclaredEvent]) -> str:
    ordered = canonical_event_order(events)
    pitches = [event.pitch for event in ordered if event.pitch is not None]
    origin = pitches[0] if pitches else 0
    value = [
        (
            event.event_type,
            event.onset_ticks,
            None if event.pitch is None else event.pitch - origin,
            event.duration_ticks,
        )
        for event in ordered
    ]
    return _hash(value)


def rhythm_interval_hash(events: Sequence[DeclaredEvent]) -> str:
    ordered = canonical_event_order(events)
    previous_pitch: int | None = None
    previous_onset = 0
    value: list[tuple[object, ...]] = []
    for event in ordered:
        interval = None
        if event.pitch is not None:
            interval = 0 if previous_pitch is None else event.pitch - previous_pitch
            previous_pitch = event.pitch
        value.append(
            (
                event.event_type,
                event.onset_ticks - previous_onset,
                interval,
                event.duration_ticks,
            )
        )
        previous_onset = event.onset_ticks
    return _hash(value)


def audit_components_do_not_cross_splits(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    fields = (
        "lineage_id",
        "whole_exact_hash",
        "whole_transposition_invariant_hash",
        "whole_rhythm_interval_hash",
        "exact_hash",
        "transposition_invariant_hash",
        "rhythm_interval_hash",
    )
    crossings: dict[str, int] = {}
    materialized = list(rows)
    for field in fields:
        groups: dict[str, set[str]] = defaultdict(set)
        for row in materialized:
            if field in row and row[field] is not None:
                groups[str(row[field])].add(str(row["split"]))
        crossings[field] = sum(len(splits) > 1 for splits in groups.values())
    return crossings
