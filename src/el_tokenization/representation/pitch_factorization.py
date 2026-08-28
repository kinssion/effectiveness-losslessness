from __future__ import annotations


def split_pitch(pitch: int) -> tuple[int, int]:
    if not 0 <= pitch <= 127:
        raise ValueError("MIDI pitch must be in [0, 127]")
    return pitch % 12, pitch // 12


def join_pitch(pitch_class: int, register: int) -> int:
    if not 0 <= pitch_class < 12 or register < 0:
        raise ValueError("invalid pitch-class/register pair")
    pitch = 12 * register + pitch_class
    if pitch > 127:
        raise ValueError("pair leaves MIDI support")
    return pitch
