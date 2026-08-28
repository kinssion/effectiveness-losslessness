from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Iterator, Mapping, Sequence

import mido


SCHEMA_VERSION = "m4l.popk_clean_split.v1"
PREPROCESSING_VERSION = "popk_lineage_audit.v1"
HASH_VERSION = "popk_canonical_music_hash.v1"


@dataclass(frozen=True, slots=True)
class CanonicalNoteFact:
    onset_tick: int
    pitch: int
    duration_tick: int


@dataclass(slots=True)
class UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, size: int) -> "UnionFind":
        return cls(list(range(size)), [0] * size)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fraction_token(ticks: int, ticks_per_beat: int) -> str:
    value = Fraction(int(ticks), int(ticks_per_beat))
    return f"{value.numerator}/{value.denominator}"


def _sorted_notes(notes: Sequence[CanonicalNoteFact]) -> list[CanonicalNoteFact]:
    return sorted(
        notes,
        key=lambda note: (note.onset_tick, note.pitch, note.duration_tick),
    )


def canonical_hashes(
    notes: Sequence[CanonicalNoteFact], ticks_per_beat: int
) -> tuple[str, str, str]:
    """Return exact, globally transposition-invariant, and loose structural hashes.

    All hashes ignore MIDI serialization order and leading silence. The exact hash
    retains absolute pitch. The transposition hash subtracts a deterministic pitch
    reference while retaining exact interval/register geometry. The rhythm-interval
    hash retains the complete metric pattern, duration/pitch-class pairing,
    simultaneity, and bass/top interval motion modulo the octave; it deliberately
    drops absolute pitch and octave placement and is therefore only a near-duplicate
    risk signal.
    """

    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be positive")
    ordered = _sorted_notes(notes)
    if not ordered:
        raise ValueError("cannot hash an empty note sequence")
    first_onset = min(note.onset_tick for note in ordered)
    first_group = [note.pitch for note in ordered if note.onset_tick == first_onset]
    pitch_reference = min(first_group)

    exact_lines = [HASH_VERSION, "kind=exact"]
    transposed_lines = [HASH_VERSION, "kind=transposition_invariant"]
    for note in ordered:
        onset = _fraction_token(note.onset_tick - first_onset, ticks_per_beat)
        duration = _fraction_token(note.duration_tick, ticks_per_beat)
        exact_lines.append(f"NOTE|{onset}|{note.pitch}|{duration}")
        transposed_lines.append(
            f"NOTE|{onset}|{note.pitch - pitch_reference}|{duration}"
        )

    grouped: dict[int, list[CanonicalNoteFact]] = defaultdict(list)
    for note in ordered:
        grouped[note.onset_tick].append(note)
    rhythm_lines = [HASH_VERSION, "kind=rhythm_interval"]
    previous_onset = first_onset
    previous_bass: int | None = None
    previous_top: int | None = None
    for onset in sorted(grouped):
        group = sorted(
            grouped[onset], key=lambda note: (note.pitch, note.duration_tick)
        )
        bass = group[0].pitch
        top = group[-1].pitch
        onset_delta = _fraction_token(onset - previous_onset, ticks_per_beat)
        bass_motion = "START" if previous_bass is None else str((bass - previous_bass) % 12)
        top_motion = "START" if previous_top is None else str((top - previous_top) % 12)
        pitch_duration = ",".join(
            f"{(note.pitch - bass) % 12}:{_fraction_token(note.duration_tick, ticks_per_beat)}"
            for note in group
        )
        rhythm_lines.append(
            f"ONSET|{onset_delta}|N={len(group)}|B={bass_motion}|T={top_motion}|{pitch_duration}"
        )
        previous_onset = onset
        previous_bass = bass
        previous_top = top

    return (
        sha256_bytes(("\n".join(exact_lines) + "\n").encode("utf-8")),
        sha256_bytes(("\n".join(transposed_lines) + "\n").encode("utf-8")),
        sha256_bytes(("\n".join(rhythm_lines) + "\n").encode("utf-8")),
    )


def _parse_midi_bytes(payload: bytes) -> tuple[list[CanonicalNoteFact], dict[str, Any]]:
    midi = mido.MidiFile(file=io.BytesIO(payload), clip=False)
    if midi.type not in (0, 1):
        raise ValueError(f"unsupported MIDI type: {midi.type}")
    if int(midi.ticks_per_beat) <= 0:
        raise ValueError("SMPTE or invalid MIDI time division is unsupported")

    notes: list[CanonicalNoteFact] = []
    time_signatures: set[tuple[int, int]] = set()
    global_end_tick = 0
    drum_note_count = 0
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        active: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        for message in track:
            tick += int(message.time)
            global_end_tick = max(global_end_tick, tick)
            if message.type == "time_signature":
                time_signatures.add(
                    (int(message.numerator), int(message.denominator))
                )
            if not hasattr(message, "channel"):
                continue
            channel = int(message.channel)
            note_number = int(getattr(message, "note", -1))
            key = (channel, note_number)
            if message.type == "note_on" and int(message.velocity) > 0:
                active[key].append(tick)
                continue
            if message.type == "note_off" or (
                message.type == "note_on" and int(message.velocity) == 0
            ):
                queue = active.get(key)
                if not queue:
                    raise ValueError(
                        "unmatched note-off: "
                        f"track={track_index},channel={channel},pitch={note_number},tick={tick}"
                    )
                onset = queue.popleft()
                duration = tick - onset
                if duration <= 0:
                    raise ValueError(
                        "non-positive note duration: "
                        f"track={track_index},channel={channel},pitch={note_number},tick={tick}"
                    )
                if channel == 9:
                    drum_note_count += 1
                else:
                    notes.append(CanonicalNoteFact(onset, note_number, duration))
        dangling = [
            (channel, pitch, list(queue))
            for (channel, pitch), queue in active.items()
            if queue
        ]
        if dangling:
            raise ValueError(
                f"dangling note-on in track {track_index}: {dangling[:3]}"
            )
    if not notes:
        raise ValueError("MIDI contains no non-drum notes")
    notes = _sorted_notes(notes)
    return notes, {
        "midi_type": int(midi.type),
        "ticks_per_beat": int(midi.ticks_per_beat),
        "global_end_tick": int(global_end_tick),
        "time_signatures": sorted(time_signatures) or [(4, 4)],
        "drum_note_count": int(drum_note_count),
    }


def scan_popk_file(path: Path, *, source_path: str) -> dict[str, Any]:
    payload = path.read_bytes()
    source_digest = sha256_bytes(payload)
    sample_id = path.stem
    try:
        release_index = int(sample_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        release_index = -1
    base: dict[str, Any] = {
        "sample_id": sample_id,
        "release_index": release_index,
        "source_path": source_path,
        "source_song_id": None,
        "augmentation_type": "release_augmented; per-sample operation unavailable",
        "transpose_offset": None,
        "crop_window_information": "released_8_bar_excerpt; original crop/window unavailable",
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_byte_sha256": source_digest,
        "source_metadata_status": "no source-work or augmentation provenance in released MIDI",
    }
    try:
        notes, metadata = _parse_midi_bytes(payload)
        exact_hash, transposition_hash, rhythm_hash = canonical_hashes(
            notes, metadata["ticks_per_beat"]
        )
        first_onset = min(note.onset_tick for note in notes)
        last_end = max(note.onset_tick + note.duration_tick for note in notes)
        onset_count = len({note.onset_tick for note in notes})
        span_ticks = last_end - first_onset
        meter = ";".join(f"{left}/{right}" for left, right in metadata["time_signatures"])
        base.update(
            {
                "exact_hash": exact_hash,
                "transposition_invariant_hash": transposition_hash,
                "rhythm_interval_hash": rhythm_hash,
                "source_ticks_per_beat": metadata["ticks_per_beat"],
                "note_count": len(notes),
                "onset_count": onset_count,
                "duration_ticks": span_ticks,
                "duration_beats": span_ticks / metadata["ticks_per_beat"],
                "time_signature_summary": meter,
                "drum_note_count": metadata["drum_note_count"],
                "parse_status": "ok",
                "parse_error": None,
            }
        )
    except Exception as error:  # keep proven corruption visible in the lineage table
        base.update(
            {
                "exact_hash": None,
                "transposition_invariant_hash": None,
                "rhythm_interval_hash": None,
                "source_ticks_per_beat": None,
                "note_count": 0,
                "onset_count": 0,
                "duration_ticks": 0,
                "duration_beats": 0.0,
                "time_signature_summary": "",
                "drum_note_count": 0,
                "parse_status": "corrupt",
                "parse_error": f"{type(error).__name__}: {error}",
            }
        )
    return base


def _scan_chunk(arguments: tuple[str, str, tuple[str, ...]]) -> list[dict[str, Any]]:
    root_string, source_prefix, names = arguments
    root = Path(root_string)
    return [
        scan_popk_file(
            root / name,
            source_path=f"{source_prefix.rstrip('/')}/{name}",
        )
        for name in names
    ]


def scan_corpus(
    source_root: Path,
    *,
    source_prefix: str,
    workers: int,
    chunk_size: int = 256,
) -> list[dict[str, Any]]:
    names = tuple(sorted(path.name for path in source_root.glob("*.mid")))
    if not names:
        raise FileNotFoundError(f"no MIDI files found under {source_root}")
    chunks = [names[index : index + chunk_size] for index in range(0, len(names), chunk_size)]
    arguments = [
        (str(source_root), source_prefix, tuple(chunk)) for chunk in chunks
    ]
    records: list[dict[str, Any]] = []
    if workers <= 1:
        results: Iterable[list[dict[str, Any]]] = map(_scan_chunk, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_scan_chunk, arguments, chunksize=1)
    try:
        for chunk_index, rows in enumerate(results, start=1):
            records.extend(rows)
            if chunk_index % 100 == 0 or chunk_index == len(chunks):
                print(
                    f"scanned {len(records):,}/{len(names):,} MIDI files",
                    flush=True,
                )
    finally:
        if workers > 1:
            executor.shutdown(wait=True)
    records.sort(key=lambda row: (int(row["release_index"]), str(row["sample_id"])))
    return records


def _union_duplicate_relation(
    records: Sequence[dict[str, Any]],
    union_find: UnionFind,
    field: str,
) -> dict[str, list[int]]:
    first: dict[str, int] = {}
    duplicates: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        if row["parse_status"] != "ok":
            continue
        value = row.get(field)
        if not value:
            continue
        if value not in first:
            first[value] = index
            continue
        first_index = first[value]
        union_find.union(first_index, index)
        if value not in duplicates:
            duplicates[value] = [first_index]
        duplicates[value].append(index)
    return duplicates


def build_lineage_components(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    union_find = UnionFind.create(len(records))
    duplicate_groups = {
        "exact_hash": _union_duplicate_relation(
            records, union_find, "exact_hash"
        ),
        "transposition_invariant_hash": _union_duplicate_relation(
            records, union_find, "transposition_invariant_hash"
        ),
        "rhythm_interval_hash": _union_duplicate_relation(
            records, union_find, "rhythm_interval_hash"
        ),
    }
    components_by_root: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        if row["parse_status"] == "ok":
            components_by_root[union_find.find(index)].append(index)

    reasons_by_index: dict[int, set[str]] = defaultdict(set)
    for relation, groups in duplicate_groups.items():
        for member_indices in groups.values():
            for index in member_indices:
                reasons_by_index[index].add(relation)

    components: dict[str, list[int]] = {}
    for member_indices in components_by_root.values():
        sample_ids = sorted(str(records[index]["sample_id"]) for index in member_indices)
        digest = sha256_bytes(("\n".join(sample_ids) + "\n").encode("utf-8"))
        lineage_id = f"popk_content_family_{digest[:24]}"
        components[lineage_id] = sorted(
            member_indices,
            key=lambda index: (
                int(records[index]["release_index"]),
                str(records[index]["sample_id"]),
            ),
        )
        component_reasons = sorted(
            {reason for index in member_indices for reason in reasons_by_index[index]}
        )
        for index in member_indices:
            records[index]["lineage_id"] = lineage_id
            records[index]["lineage_confidence"] = "unresolved_source_lineage"
            records[index]["lineage_status"] = (
                "content_family_detected_without_source_provenance"
                if len(member_indices) > 1
                else "singleton_without_source_provenance"
            )
            records[index]["merge_reasons"] = ",".join(component_reasons)

    for index, row in enumerate(records):
        if row["parse_status"] != "ok":
            digest = sha256_bytes(str(row["sample_id"]).encode("utf-8"))
            row["lineage_id"] = f"popk_corrupt_{digest[:24]}"
            row["lineage_confidence"] = "excluded_proven_corruption"
            row["lineage_status"] = "excluded_proven_corruption"
            row["merge_reasons"] = ""
    return components, duplicate_groups


def _seeded_key(seed: int, purpose: str, lineage_id: str) -> str:
    return sha256_bytes(f"{seed}|{purpose}|{lineage_id}".encode("utf-8"))


def _balanced_component_order(
    component_rows: Sequence[dict[str, Any]], *, seed: int, purpose: str
) -> list[dict[str, Any]]:
    if not component_rows:
        return []
    by_density = sorted(
        component_rows,
        key=lambda row: (float(row["mean_note_count"]), str(row["lineage_id"])),
    )
    bin_count = min(10, len(by_density))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(bin_count)]
    for rank, row in enumerate(by_density):
        bin_index = min(bin_count - 1, (rank * bin_count) // len(by_density))
        bins[bin_index].append(row)
    for bin_rows in bins:
        bin_rows.sort(
            key=lambda row: _seeded_key(seed, purpose, str(row["lineage_id"]))
        )
    order: list[dict[str, Any]] = []
    maximum = max(len(bin_rows) for bin_rows in bins)
    bin_order = sorted(
        range(bin_count),
        key=lambda index: _seeded_key(seed, f"{purpose}:bin", str(index)),
    )
    for offset in range(maximum):
        for bin_index in bin_order:
            if offset < len(bins[bin_index]):
                order.append(bins[bin_index][offset])
    return order


def _select_components(
    available: dict[str, dict[str, Any]],
    *,
    target_samples: int,
    seed: int,
    purpose: str,
) -> set[str]:
    selected: set[str] = set()
    remaining = int(target_samples)
    order = _balanced_component_order(
        list(available.values()), seed=seed, purpose=purpose
    )
    for component in order:
        size = int(component["sample_count"])
        if size <= remaining:
            selected.add(str(component["lineage_id"]))
            remaining -= size
            if remaining == 0:
                return selected
    if remaining > 0:
        candidates = [
            component
            for component in order
            if str(component["lineage_id"]) not in selected
        ]
        if candidates:
            best = min(
                candidates,
                key=lambda row: (
                    abs(int(row["sample_count"]) - remaining),
                    _seeded_key(seed, f"{purpose}:overshoot", str(row["lineage_id"])),
                ),
            )
            selected.add(str(best["lineage_id"]))
    return selected


def assign_splits(
    records: list[dict[str, Any]],
    components: Mapping[str, Sequence[int]],
    *,
    seed: int,
    validation_target: int,
    test_target: int,
    old_holdout_count: int,
) -> dict[str, Any]:
    component_rows: dict[str, dict[str, Any]] = {}
    for lineage_id, indices in components.items():
        note_total = sum(int(records[index]["note_count"]) for index in indices)
        component_rows[lineage_id] = {
            "lineage_id": lineage_id,
            "sample_count": len(indices),
            "note_count": note_total,
            "mean_note_count": note_total / len(indices),
        }
    test_ids = _select_components(
        component_rows,
        target_samples=test_target,
        seed=seed,
        purpose="test",
    )
    remaining = {
        lineage_id: row
        for lineage_id, row in component_rows.items()
        if lineage_id not in test_ids
    }
    validation_ids = _select_components(
        remaining,
        target_samples=validation_target,
        seed=seed,
        purpose="validation",
    )

    eligible = [row for row in records if row["parse_status"] == "ok"]
    old_boundary = max(0, len(eligible) - int(old_holdout_count))
    old_holdout_ids = {
        str(row["sample_id"])
        for row in sorted(
            eligible,
            key=lambda row: (int(row["release_index"]), str(row["sample_id"])),
        )[old_boundary:]
    }
    moved = 0
    for row in records:
        if row["parse_status"] != "ok":
            row["split"] = "excluded_corrupt"
            row["old_split"] = "excluded_corrupt"
            row["moved_relative_to_old"] = False
            continue
        lineage_id = str(row["lineage_id"])
        if lineage_id in test_ids:
            split = "test"
        elif lineage_id in validation_ids:
            split = "validation"
        else:
            split = "train"
        old_split = (
            "holdout"
            if str(row["sample_id"]) in old_holdout_ids
            else "train"
        )
        moved_relative = (old_split == "train" and split != "train") or (
            old_split == "holdout" and split == "train"
        )
        row["split"] = split
        row["old_split"] = old_split
        row["moved_relative_to_old"] = bool(moved_relative)
        moved += int(moved_relative)
    actual = {
        split: sum(row["split"] == split for row in records)
        for split in ("train", "validation", "test", "excluded_corrupt")
    }
    return {
        "targets": {
            "validation": int(validation_target),
            "test": int(test_target),
            "train": max(
                0,
                len(eligible) - int(validation_target) - int(test_target),
            ),
        },
        "actual": actual,
        "moved_relative_to_old": moved,
        "old_holdout_count": int(old_holdout_count),
    }


def validate_no_crossing(
    records: Sequence[Mapping[str, Any]],
    duplicate_groups: Mapping[str, Mapping[str, Sequence[int]]],
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    relation_groups: dict[str, dict[str, list[int]]] = {
        "lineage_id": defaultdict(list)
    }
    for index, row in enumerate(records):
        if row["parse_status"] == "ok":
            relation_groups["lineage_id"][str(row["lineage_id"])].append(index)
    relation_groups.update(
        {relation: dict(groups) for relation, groups in duplicate_groups.items()}
    )
    for relation, groups in relation_groups.items():
        violations = 0
        for key, indices in groups.items():
            splits = sorted(
                {
                    str(records[index]["split"])
                    for index in indices
                    if records[index]["parse_status"] == "ok"
                }
            )
            if len(splits) <= 1:
                continue
            violations += 1
            audit_rows.append(
                {
                    "relation": relation,
                    "key": key,
                    "sample_count": len(indices),
                    "splits": ",".join(splits),
                    "status": "FAIL",
                    "details": "cross-split connected relation",
                }
            )
        audit_rows.insert(
            0,
            {
                "relation": relation,
                "key": "__SUMMARY__",
                "sample_count": sum(len(indices) for indices in groups.values()),
                "splits": "",
                "status": "PASS" if violations == 0 else "FAIL",
                "details": f"groups={len(groups)};cross_split_groups={violations}",
            },
        )
    failures = [row for row in audit_rows if row["status"] == "FAIL"]
    if failures:
        raise AssertionError(f"cross-split leakage detected: {failures[:3]}")
    return audit_rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            count += 1
    os.replace(temporary, path)
    return count


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    os.replace(temporary, path)
    return count


LINEAGE_COLUMNS = (
    "sample_id",
    "release_index",
    "source_path",
    "source_song_id",
    "lineage_id",
    "augmentation_type",
    "transpose_offset",
    "crop_window_information",
    "preprocessing_version",
    "source_byte_sha256",
    "exact_hash",
    "transposition_invariant_hash",
    "rhythm_interval_hash",
    "source_ticks_per_beat",
    "note_count",
    "onset_count",
    "duration_ticks",
    "duration_beats",
    "time_signature_summary",
    "drum_note_count",
    "source_metadata_status",
    "lineage_confidence",
    "lineage_status",
    "merge_reasons",
    "split",
    "old_split",
    "moved_relative_to_old",
    "parse_status",
    "parse_error",
)


def write_lineage_parquet(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "pyarrow is required for lineage_table.parquet; install pyarrow first"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), False),
            pa.field("release_index", pa.int32(), False),
            pa.field("source_path", pa.string(), False),
            pa.field("source_song_id", pa.string(), True),
            pa.field("lineage_id", pa.string(), False),
            pa.field("augmentation_type", pa.string(), False),
            pa.field("transpose_offset", pa.int16(), True),
            pa.field("crop_window_information", pa.string(), False),
            pa.field("preprocessing_version", pa.string(), False),
            pa.field("source_byte_sha256", pa.string(), False),
            pa.field("exact_hash", pa.string(), True),
            pa.field("transposition_invariant_hash", pa.string(), True),
            pa.field("rhythm_interval_hash", pa.string(), True),
            pa.field("source_ticks_per_beat", pa.int32(), True),
            pa.field("note_count", pa.int32(), False),
            pa.field("onset_count", pa.int32(), False),
            pa.field("duration_ticks", pa.int64(), False),
            pa.field("duration_beats", pa.float64(), False),
            pa.field("time_signature_summary", pa.string(), False),
            pa.field("drum_note_count", pa.int32(), False),
            pa.field("source_metadata_status", pa.string(), False),
            pa.field("lineage_confidence", pa.string(), False),
            pa.field("lineage_status", pa.string(), False),
            pa.field("merge_reasons", pa.string(), False),
            pa.field("split", pa.string(), False),
            pa.field("old_split", pa.string(), False),
            pa.field("moved_relative_to_old", pa.bool_(), False),
            pa.field("parse_status", pa.string(), False),
            pa.field("parse_error", pa.string(), True),
        ]
    )
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    try:
        for start in range(0, len(records), 25_000):
            batch_rows = [
                {column: row.get(column) for column in LINEAGE_COLUMNS}
                for row in records[start : start + 25_000]
            ]
            writer.write_table(pa.Table.from_pylist(batch_rows, schema=schema))
    finally:
        writer.close()
    os.replace(temporary, path)


def _duplicate_membership_rows(
    records: Sequence[Mapping[str, Any]], groups: Mapping[str, Sequence[int]]
) -> Iterator[dict[str, Any]]:
    for digest in sorted(groups):
        indices = groups[digest]
        for index in indices:
            row = records[index]
            yield {
                "hash": digest,
                "group_size": len(indices),
                "lineage_id": row["lineage_id"],
                "sample_id": row["sample_id"],
                "source_path": row["source_path"],
                "split": row["split"],
            }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution_row(
    scheme: str, split: str, rows: Sequence[Mapping[str, Any]], total: int
) -> dict[str, Any]:
    note_counts = [float(row["note_count"]) for row in rows]
    onset_counts = [float(row["onset_count"]) for row in rows]
    durations = [float(row["duration_beats"]) for row in rows]
    return {
        "partition_scheme": scheme,
        "split": split,
        "sample_count": len(rows),
        "sample_percent": 100.0 * len(rows) / max(1, total),
        "mean_note_count": statistics.fmean(note_counts) if note_counts else 0.0,
        "p05_note_count": _percentile(note_counts, 0.05),
        "p50_note_count": _percentile(note_counts, 0.50),
        "p95_note_count": _percentile(note_counts, 0.95),
        "mean_onset_count": statistics.fmean(onset_counts) if onset_counts else 0.0,
        "mean_duration_beats": statistics.fmean(durations) if durations else 0.0,
    }


def _file_receipt(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def build_outputs(config: Mapping[str, Any]) -> dict[str, Any]:
    source_root = Path(str(config["source_root"]))
    manifest_dir = Path(str(config["manifest_dir"]))
    audit_dir = Path(str(config["audit_dir"]))
    report_path = Path(str(config["report_path"]))
    workers = int(config.get("workers", 1))
    seed = int(config["seed"])
    records = scan_corpus(
        source_root,
        source_prefix=str(config["source_path_prefix"]),
        workers=workers,
        chunk_size=int(config.get("scan_chunk_size", 256)),
    )
    components, duplicate_groups = build_lineage_components(records)
    split_receipt = assign_splits(
        records,
        components,
        seed=seed,
        validation_target=int(config["validation_target"]),
        test_target=int(config["test_target"]),
        old_holdout_count=int(config["old_holdout_count"]),
    )
    leakage_rows = validate_no_crossing(records, duplicate_groups)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = manifest_dir / "lineage_table.parquet"
    write_lineage_parquet(lineage_path, records)

    jsonl_columns = (
        "sample_id",
        "source_path",
        "source_song_id",
        "lineage_id",
        "exact_hash",
        "transposition_invariant_hash",
        "rhythm_interval_hash",
        "note_count",
        "onset_count",
        "duration_beats",
        "preprocessing_version",
    )
    split_file_rows: dict[str, int] = {}
    for split, filename in (
        ("train", "train.jsonl"),
        ("validation", "validation.jsonl"),
        ("test", "test.jsonl"),
    ):
        selected = [
            {column: row.get(column) for column in jsonl_columns}
            for row in records
            if row["split"] == split
        ]
        split_file_rows[filename] = write_jsonl(manifest_dir / filename, selected)

    duplicate_paths = {
        "exact_hash": audit_dir / "exact_duplicates.csv",
        "transposition_invariant_hash": audit_dir / "transposition_duplicates.csv",
        "rhythm_interval_hash": audit_dir / "rhythm_interval_duplicates.csv",
    }
    duplicate_file_rows: dict[str, int] = {}
    for relation, path in duplicate_paths.items():
        duplicate_file_rows[path.name] = write_csv(
            path,
            ("hash", "group_size", "lineage_id", "sample_id", "source_path", "split"),
            _duplicate_membership_rows(records, duplicate_groups[relation]),
        )
    leakage_count = write_csv(
        audit_dir / "lineage_leakage.csv",
        ("relation", "key", "sample_count", "splits", "status", "details"),
        leakage_rows,
    )

    ambiguous_rows: list[dict[str, Any]] = []
    singleton_count = 0
    for lineage_id, indices in sorted(components.items()):
        if len(indices) == 1:
            singleton_count += 1
            continue
        reasons = sorted(
            {
                reason
                for index in indices
                for reason in str(records[index]["merge_reasons"]).split(",")
                if reason
            }
        )
        member_ids = sorted(str(records[index]["sample_id"]) for index in indices)
        ambiguous_rows.append(
            {
                "lineage_id": lineage_id,
                "sample_count": len(indices),
                "relation_types": ",".join(reasons),
                "representative_sample_id": member_ids[0],
                "member_id_sha256": sha256_bytes(("\n".join(member_ids) + "\n").encode("utf-8")),
                "source_lineage_confidence": "unresolved",
                "ambiguity": "content-connected family; original source work is unavailable",
            }
        )
    ambiguous_rows.append(
        {
            "lineage_id": "__UNRESOLVED_SINGLETONS__",
            "sample_count": singleton_count,
            "relation_types": "none_detected",
            "representative_sample_id": "",
            "member_id_sha256": "",
            "source_lineage_confidence": "unresolved",
            "ambiguity": "no released source-work provenance and no detected content duplicate relation",
        }
    )
    ambiguous_count = write_csv(
        audit_dir / "ambiguous_lineage.csv",
        (
            "lineage_id",
            "sample_count",
            "relation_types",
            "representative_sample_id",
            "member_id_sha256",
            "source_lineage_confidence",
            "ambiguity",
        ),
        ambiguous_rows,
    )

    eligible_rows = [row for row in records if row["parse_status"] == "ok"]
    distribution_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        distribution_rows.append(
            _distribution_row(
                "popk_clean_v1",
                split,
                [row for row in eligible_rows if row["split"] == split],
                len(eligible_rows),
            )
        )
    for old_split in ("train", "holdout"):
        distribution_rows.append(
            _distribution_row(
                "legacy_last_2048",
                old_split,
                [row for row in eligible_rows if row["old_split"] == old_split],
                len(eligible_rows),
            )
        )
    distribution_count = write_csv(
        audit_dir / "distribution_comparison.csv",
        (
            "partition_scheme",
            "split",
            "sample_count",
            "sample_percent",
            "mean_note_count",
            "p05_note_count",
            "p50_note_count",
            "p95_note_count",
            "mean_onset_count",
            "mean_duration_beats",
        ),
        distribution_rows,
    )

    duplicate_component_count = sum(len(indices) > 1 for indices in components.values())
    moved = int(split_receipt["moved_relative_to_old"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "release": "Patchbanks Pop-K MIDI Dataset",
            "source_root": str(config["source_path_prefix"]),
            "official_repository": "https://github.com/patchbanks/Pop-K-MIDI-Dataset",
            "official_archive_doi": "10.5281/zenodo.14791511",
            "released_sample_count": len(records),
            "metadata_finding": (
                "The available release contains sequentially named MIDI excerpts but no "
                "original work ID, parent MIDI identity, per-sample augmentation record, "
                "transpose offset, or crop provenance."
            ),
        },
        "lineage": {
            "high_confidence_source_lineage_samples": 0,
            "high_confidence_source_lineage_percent": 0.0,
            "content_connected_components": len(components),
            "connected_duplicate_families": duplicate_component_count,
            "unresolved_source_lineage_samples": len(records),
        },
        "duplicates": {
            relation: {
                "groups": len(groups),
                "samples": sum(len(indices) for indices in groups.values()),
            }
            for relation, groups in duplicate_groups.items()
        },
        "split": split_receipt,
        "legacy_comparison": {
            "moved_samples": moved,
            "moved_percent": 100.0 * moved / max(1, len(eligible_rows)),
            "legacy_policy": "last 2048 release-order samples as one holdout",
        },
        "corruption": {
            "excluded_samples": sum(row["parse_status"] != "ok" for row in records),
            "policy": "only proven parse corruption is explicitly excluded; no sample is silently dropped",
        },
        "leakage": {
            "observable_relations_cross_split": 0,
            "lineage_id_cross_split": 0,
            "exact_hash_cross_split": 0,
            "transposition_invariant_hash_cross_split": 0,
            "rhythm_interval_hash_cross_split": 0,
            "fully_leakage_free_for_observable_relations": True,
            "fully_leakage_free_for_original_source_works_provable": False,
        },
        "claim_boundary": (
            "The split is leakage-free under all observable provenance/content relations. "
            "A source-work-level leakage-free claim is not provable because the Pop-K "
            "release does not expose original lineage."
        ),
    }
    write_json(audit_dir / "split_summary.json", summary)

    output_receipts: dict[str, Any] = {}
    for path, row_count in (
        (lineage_path, len(records)),
        (manifest_dir / "train.jsonl", split_file_rows["train.jsonl"]),
        (manifest_dir / "validation.jsonl", split_file_rows["validation.jsonl"]),
        (manifest_dir / "test.jsonl", split_file_rows["test.jsonl"]),
        (audit_dir / "split_summary.json", None),
        (audit_dir / "lineage_leakage.csv", leakage_count),
        (audit_dir / "exact_duplicates.csv", duplicate_file_rows["exact_duplicates.csv"]),
        (audit_dir / "transposition_duplicates.csv", duplicate_file_rows["transposition_duplicates.csv"]),
        (audit_dir / "rhythm_interval_duplicates.csv", duplicate_file_rows["rhythm_interval_duplicates.csv"]),
        (audit_dir / "ambiguous_lineage.csv", ambiguous_count),
        (audit_dir / "distribution_comparison.csv", distribution_count),
    ):
        output_receipts[path.as_posix()] = _file_receipt(path, rows=row_count)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "input": {
            "root": str(config["source_path_prefix"]),
            "sample_count": len(records),
            "corpus_semantic_digest": sha256_bytes(
                ("\n".join(
                    f"{row['sample_id']}|{row.get('exact_hash') or row['source_byte_sha256']}"
                    for row in records
                ) + "\n").encode("utf-8")
            ),
        },
        "lineage_policy": {
            "source_priority": [
                "original source work ID",
                "original MIDI/file identity before augmentation",
                "explicit parent/sample provenance",
                "deterministic canonical content family",
            ],
            "available_level": "deterministic canonical content family only",
            "component_edges": [
                "source lineage equality when available",
                "exact_hash equality",
                "transposition_invariant_hash equality",
                "rhythm_interval_hash equality",
            ],
        },
        "hash_policy": {
            "version": HASH_VERSION,
            "exact_hash": "event type, beat-normalized onset, absolute pitch, duration, simultaneity, order",
            "transposition_invariant_hash": "exact canonical sequence after subtracting the minimum pitch at the first sounding onset",
            "rhythm_interval_hash": "onset deltas, durations, simultaneity, pitch-class vertical intervals, and bass/top interval motion with absolute pitch/octave removed",
        },
        "split_policy": {
            "unit": "connected lineage/content component",
            "validation_target": int(config["validation_target"]),
            "test_target": int(config["test_target"]),
            "selection": "seeded deterministic component selection stratified by mean note count; no labels or metrics",
            "actual": split_receipt["actual"],
        },
        "outputs": output_receipts,
    }
    manifest_path = manifest_dir / "split_manifest.json"
    write_json(manifest_path, manifest)
    manifest_digest = sha256_file(manifest_path)
    _atomic_write_bytes(
        manifest_dir / "split_manifest.sha256",
        f"{manifest_digest}  split_manifest.json\n".encode("ascii"),
    )

    report = f"""# Pop-K clean split v1: lineage and leakage report

## Result

The deterministic split contains **{split_receipt['actual']['train']:,} train**, **{split_receipt['actual']['validation']:,} validation**, and **{split_receipt['actual']['test']:,} test** samples. All observable relations (`lineage_id`, exact content, global-transposition-invariant content, and rhythm/interval near-duplicate structure) are confined to one split.

This does **not** establish a fully source-work-lineage-clean Pop-K test set. The released corpus provides sequential MIDI filenames but no original work ID, pre-augmentation file identity, explicit parent mapping, per-sample transpose offset, or crop/window provenance. We therefore report source lineage as unresolved rather than inventing it.

## Provenance coverage

- Released samples scanned: **{len(records):,}**
- High-confidence original-source lineage: **0 / {len(records):,} (0.000%)**
- Content-connected components: **{len(components):,}**
- Connected duplicate families (component size > 1): **{duplicate_component_count:,}**
- Proven corrupt samples explicitly excluded: **{summary['corruption']['excluded_samples']:,}**

The official release describes 305,815 augmented eight-bar MIDI excerpts normalized to 120 BPM and PPQ 96, derived from vocal-lead/chord/bass material. Neither the local archive nor the public repository supplies the missing per-sample lineage mapping. See the [official repository](https://github.com/patchbanks/Pop-K-MIDI-Dataset) and [Zenodo archive](https://doi.org/10.5281/zenodo.14791511).

## Duplicate relations

| Relation | Duplicate groups | Samples in groups |
|---|---:|---:|
| Exact content | {len(duplicate_groups['exact_hash']):,} | {sum(len(v) for v in duplicate_groups['exact_hash'].values()):,} |
| Transposition-invariant | {len(duplicate_groups['transposition_invariant_hash']):,} | {sum(len(v) for v in duplicate_groups['transposition_invariant_hash'].values()):,} |
| Rhythm/interval | {len(duplicate_groups['rhythm_interval_hash']):,} | {sum(len(v) for v in duplicate_groups['rhythm_interval_hash'].values()):,} |

These hashes are relation signals, not aliases for an original composition ID. The final `lineage_id` is the connected component induced by all available relations, so any ambiguous near-duplicate family is assigned as one unit.

## Leakage checks

| Check | Cross-split groups |
|---|---:|
| `lineage_id` | 0 |
| `exact_hash` | 0 |
| `transposition_invariant_hash` | 0 |
| `rhythm_interval_hash` | 0 |

The observable split is clean. A stronger claim that no train/test samples descend from the same unreleased source work remains **unverifiable from the available Pop-K metadata**. This unresolved ambiguity must be disclosed in an ICLR claim.

## Comparison with the legacy split

The legacy policy treated the last 2,048 release-order files as one holdout. Under the new lineage-level validation/test assignment, **{moved:,} samples ({100.0 * moved / max(1, len(eligible_rows)):.3f}%)** change train-versus-holdout membership. Release order is not used by the new assignment except to reproduce this comparison.

## Deterministic rebuild

From the repository root:

```powershell
python tools/build_popk_clean_split.py --config configs/popk_clean_v1.json
```

The command does not train a model and does not inspect or allocate any GPU. The manifest digest is:

```text
{manifest_digest}
```

Training code must load `train.jsonl`, `validation.jsonl`, or `test.jsonl` explicitly; implicit cache-order splitting is outside this contract.
"""
    _atomic_write_bytes(report_path, report.encode("utf-8"))

    return {
        "manifest_sha256": manifest_digest,
        "summary": summary,
        "report_path": report_path.as_posix(),
    }


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unexpected config schema: {value.get('schema_version')!r}"
        )
    return value
