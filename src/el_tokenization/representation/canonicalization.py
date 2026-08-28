from __future__ import annotations

from collections.abc import Iterable


def canonicalize_pitch(pitch: int, shift: int) -> int:
    result = pitch + shift
    if not 0 <= result <= 127:
        raise ValueError("canonicalized pitch leaves MIDI support")
    return result


def restore_pitch(canonical_pitch: int, shift: int) -> int:
    return canonicalize_pitch(canonical_pitch, -shift)


def canonicalize_pitches(pitches: Iterable[int], shift: int) -> tuple[int, ...]:
    return tuple(canonicalize_pitch(pitch, shift) for pitch in pitches)


def restore_pitches(pitches: Iterable[int], shift: int) -> tuple[int, ...]:
    return tuple(restore_pitch(pitch, shift) for pitch in pitches)
