from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import mido


SCHEMA_VERSION = "m4l.pop1k7_music21_key_estimation.v1"
_PITCH_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def relative_major_frame_pc(tonic_pc: int, mode: str) -> int:
    normalized_mode = str(mode).lower()
    if normalized_mode == "major":
        return int(tonic_pc) % 12
    if normalized_mode == "minor":
        return (int(tonic_pc) + 3) % 12
    raise ValueError(f"unsupported mode: {mode!r}")


def canonical_shift_from_frame_pc(frame_pc: int) -> int:
    candidates = [shift for shift in range(-5, 7) if (int(frame_pc) + shift) % 12 == 0]
    if len(candidates) != 1:
        raise AssertionError("frame has no unique shift in (-6,+6]")
    return int(candidates[0])


def compact_key_label(tonic_pc: int, mode: str) -> str:
    name = _PITCH_NAMES[int(tonic_pc) % 12]
    if str(mode).lower() == "minor":
        return name[0].lower() + name[1:]
    if str(mode).lower() != "major":
        raise ValueError(f"unsupported mode: {mode!r}")
    return name


def source_song_id(relative_path: Path) -> str:
    if len(relative_path.parts) != 2 or not relative_path.parent.name.startswith("src_"):
        raise ValueError(f"unexpected Pop1K7 relative path: {relative_path}")
    int(relative_path.stem)
    return f"pop1k7:{relative_path.parent.name}:{relative_path.stem}"


def _midi_note_rows(payload: bytes) -> tuple[tuple[int, int, int], ...]:
    midi = mido.MidiFile(file=io.BytesIO(payload), clip=False)
    rows: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        tick = 0
        active: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        for message in track:
            tick += int(message.time)
            if not hasattr(message, "channel"):
                continue
            channel = int(message.channel)
            pitch = int(getattr(message, "note", -1))
            key = (channel, pitch)
            if message.type == "note_on" and int(message.velocity) > 0:
                active[key].append(tick)
                continue
            if message.type == "note_off" or (
                message.type == "note_on" and int(message.velocity) == 0
            ):
                queue = active.get(key)
                if not queue:
                    raise ValueError("unmatched MIDI note-off")
                onset = queue.popleft()
                duration = tick - onset
                if duration <= 0:
                    raise ValueError("non-positive MIDI note duration")
                if channel != 9:
                    rows.append((onset, pitch, duration))
        if any(active.values()):
            raise ValueError("dangling MIDI note-on")
    rows.sort()
    if not rows:
        raise ValueError("MIDI has no pitched notes")
    return tuple(rows)


def midi_note_content_sha256(path: Path) -> tuple[str, int]:
    rows = _midi_note_rows(path.read_bytes())
    payload = json.dumps(rows, separators=(",", ":")).encode("ascii")
    return sha256_bytes(payload), len(rows)


def _key_record(value: Any) -> dict[str, Any]:
    tonic_pc = int(value.tonic.pitchClass)
    mode = str(value.mode).lower()
    frame_pc = relative_major_frame_pc(tonic_pc, mode)
    alternatives = []
    for alternate in value.alternateInterpretations:
        alternate_tonic = int(alternate.tonic.pitchClass)
        alternate_mode = str(alternate.mode).lower()
        alternatives.append(
            {
                "label": compact_key_label(alternate_tonic, alternate_mode),
                "tonic_pc": alternate_tonic,
                "mode": alternate_mode,
                "frame_pc": relative_major_frame_pc(alternate_tonic, alternate_mode),
                "fifths": int(alternate.sharps),
                "correlation": float(alternate.correlationCoefficient),
            }
        )
    distinct_frame_runner = next(
        (candidate for candidate in alternatives if int(candidate["frame_pc"]) != frame_pc),
        None,
    )
    correlation = float(value.correlationCoefficient)
    runner_correlation = (
        float(distinct_frame_runner["correlation"])
        if distinct_frame_runner is not None
        else float("-inf")
    )
    return {
        "label": compact_key_label(tonic_pc, mode),
        "tonic_pc": tonic_pc,
        "mode": mode,
        "frame_pc": frame_pc,
        "fifths": int(value.sharps),
        "correlation": correlation,
        "distinct_frame_runner_up": distinct_frame_runner,
        "distinct_frame_margin": correlation - runner_correlation,
        "top_alternatives": alternatives[:5],
    }


def analyze_music21_key(path: Path) -> dict[str, dict[str, Any]]:
    from music21 import converter

    score = converter.parse(path)
    return {
        "krumhansl_schmuckler": _key_record(
            score.analyze("key.krumhanslschmuckler")
        ),
        "aarden_essen": _key_record(score.analyze("key.aardenessen")),
    }


def classify_consensus(
    analyses: Mapping[str, Mapping[str, Any]],
    *,
    minimum_correlation: float,
    minimum_distinct_frame_margin: float,
) -> dict[str, Any]:
    primary = analyses["krumhansl_schmuckler"]
    secondary = analyses["aarden_essen"]
    same_frame = int(primary["frame_pc"]) == int(secondary["frame_pc"])
    threshold_pass = all(
        float(record["correlation"]) >= float(minimum_correlation)
        and float(record["distinct_frame_margin"]) >= float(minimum_distinct_frame_margin)
        for record in (primary, secondary)
    )
    if not same_frame:
        status = "AMBIGUOUS_FRAME_DISAGREEMENT"
    elif not threshold_pass:
        status = "REVIEW_LOW_CONFIDENCE"
    else:
        status = "ADMITTED_HIGH_CONFIDENCE"
    frame_pc = int(primary["frame_pc"]) if same_frame else None
    return {
        "status": status,
        "same_relative_major_frame": same_frame,
        "same_mode": str(primary["mode"]) == str(secondary["mode"]),
        "estimated_frame_pc": frame_pc,
        "canonical_shift": (
            canonical_shift_from_frame_pc(frame_pc) if frame_pc is not None else None
        ),
        "selected_key_label": str(primary["label"]) if same_frame else None,
        "selected_tonic_pc": int(primary["tonic_pc"]) if same_frame else None,
        "selected_mode": str(primary["mode"]) if same_frame else None,
    }


def analyze_pop1k7_pair(
    synchronized_path: Path,
    model_source_path: Path,
    *,
    repository_root: Path,
    minimum_correlation: float,
    minimum_distinct_frame_margin: float,
) -> dict[str, Any]:
    synchronized_relative = Path(synchronized_path.parent.name) / synchronized_path.name
    model_relative = Path(model_source_path.parent.name) / model_source_path.name
    if synchronized_relative != model_relative:
        raise ValueError("Pop1K7 same-release relative paths differ")
    synchronized_note_hash, synchronized_notes = midi_note_content_sha256(synchronized_path)
    model_note_hash, model_notes = midi_note_content_sha256(model_source_path)
    if synchronized_note_hash != model_note_hash or synchronized_notes != model_notes:
        raise ValueError("synchronized and analyzed MIDI note content differs")
    analyses = analyze_music21_key(synchronized_path)
    consensus = classify_consensus(
        analyses,
        minimum_correlation=minimum_correlation,
        minimum_distinct_frame_margin=minimum_distinct_frame_margin,
    )
    relative = Path(synchronized_path.parent.name) / synchronized_path.name
    return {
        "schema_version": SCHEMA_VERSION,
        "source_song_id": source_song_id(relative),
        "numeric_song_id": int(synchronized_path.stem),
        "source_bucket": synchronized_path.parent.name,
        "estimation_source_path": synchronized_path.relative_to(repository_root).as_posix(),
        "estimation_source_sha256": sha256_file(synchronized_path),
        "model_source_path": model_source_path.relative_to(repository_root).as_posix(),
        "model_source_sha256": sha256_file(model_source_path),
        "note_content_sha256": synchronized_note_hash,
        "note_count": synchronized_notes,
        "dataset_key_label_used": False,
        "analysis_scope": "full_song_all_pitched_notes",
        "analyses": analyses,
        **consensus,
    }
