from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import mido

from .iclr_matched_harness import _stream_from_notes, load_json_yaml
from .note_centric_music import linearize_note_tokens
from .popk_clean_split import CanonicalNoteFact, canonical_hashes
from .sparse_melody_bpe import SparseMelodyStream


SCHEMA_VERSION = "m4l.commu_4_4_clean_split.v1"
PREPROCESSING_VERSION = "commu_4_4_fixed_tonal_frame.v1"
TARGET_TICKS_PER_BEAT = 96
SOURCE_TICKS_PER_BEAT = 480
SplitName = Literal["train", "validation", "test"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(payload)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _midi_key_signatures(payload: bytes) -> tuple[str, ...]:
    midi = mido.MidiFile(file=io.BytesIO(payload), clip=False)
    return tuple(
        str(message.key)
        for track in midi.tracks
        for message in track
        if message.type == "key_signature"
    )


def parse_commu_midi(
    path: Path,
    *,
    expected_audio_key: str,
) -> tuple[list[CanonicalNoteFact], dict[str, Any]]:
    """Read one official ComMU clip and losslessly map PPQ 480 to PPQ 96.

    ComMU's released raw clips are already normalized to C major or A minor.
    This parser verifies that fact from both metadata and the MIDI key-signature
    event. It never transposes pitch and never performs timing quantization.
    """

    payload = path.read_bytes()
    midi = mido.MidiFile(file=io.BytesIO(payload), clip=False)
    if midi.type not in (0, 1):
        raise ValueError(f"unsupported MIDI type: {midi.type}")
    if int(midi.ticks_per_beat) != SOURCE_TICKS_PER_BEAT:
        raise ValueError(
            f"unexpected ComMU PPQ: {midi.ticks_per_beat} != {SOURCE_TICKS_PER_BEAT}"
        )
    expected_midi_key = {"cmajor": "C", "aminor": "Am"}.get(expected_audio_key)
    if expected_midi_key is None:
        raise ValueError(f"unsupported ComMU tonal frame: {expected_audio_key!r}")
    key_signatures = _midi_key_signatures(payload)
    if key_signatures != (expected_midi_key,):
        raise ValueError(
            f"MIDI/metadata key mismatch: {key_signatures!r} != {(expected_midi_key,)!r}"
        )

    notes: list[CanonicalNoteFact] = []
    time_signatures: set[tuple[int, int]] = set()
    drum_note_count = 0
    global_end_tick = 0
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
            pitch = int(getattr(message, "note", -1))
            key = (channel, pitch)
            if message.type == "note_on" and int(message.velocity) > 0:
                active[key].append(tick)
            elif message.type == "note_off" or (
                message.type == "note_on" and int(message.velocity) == 0
            ):
                queue = active.get(key)
                if not queue:
                    raise ValueError(
                        "unmatched note-off: "
                        f"track={track_index},channel={channel},pitch={pitch},tick={tick}"
                    )
                onset = queue.popleft()
                duration = tick - onset
                if duration <= 0:
                    raise ValueError("non-positive ComMU note duration")
                if channel == 9:
                    drum_note_count += 1
                    continue
                onset_scaled, onset_remainder = divmod(
                    onset * TARGET_TICKS_PER_BEAT, SOURCE_TICKS_PER_BEAT
                )
                duration_scaled, duration_remainder = divmod(
                    duration * TARGET_TICKS_PER_BEAT, SOURCE_TICKS_PER_BEAT
                )
                if onset_remainder or duration_remainder:
                    raise ValueError(
                        "ComMU timing is not exactly representable at PPQ 96"
                    )
                if duration_scaled <= 0:
                    raise ValueError("PPQ conversion produced a zero duration")
                notes.append(
                    CanonicalNoteFact(
                        onset_tick=int(onset_scaled),
                        pitch=pitch,
                        duration_tick=int(duration_scaled),
                    )
                )
        dangling = [key for key, queue in active.items() if queue]
        if dangling:
            raise ValueError(
                f"dangling note-on in track {track_index}: {dangling[:3]}"
            )
    if time_signatures != {(4, 4)}:
        raise ValueError(f"non-constant 4/4 MIDI entered ComMU replication: {time_signatures}")
    if not notes:
        raise ValueError("ComMU clip contains no non-drum notes")
    notes.sort(key=lambda note: (note.onset_tick, note.pitch, note.duration_tick))
    return notes, {
        "source_ticks_per_beat": SOURCE_TICKS_PER_BEAT,
        "target_ticks_per_beat": TARGET_TICKS_PER_BEAT,
        "time_signatures": ((4, 4),),
        "key_signatures": key_signatures,
        "audio_key": expected_audio_key,
        "global_end_source_tick": global_end_tick,
        "drum_note_count": drum_note_count,
    }


@dataclass(frozen=True, slots=True)
class ComMUManifestEntry:
    sample_id: str
    source_path: str
    source_song_id: str
    lineage_id: str
    exact_hash: str
    transposition_invariant_hash: str
    rhythm_interval_hash: str
    note_count: int
    onset_count: int
    duration_beats: float
    preprocessing_version: str
    split: SplitName
    original_split: str
    audio_key: str
    num_measures: int
    target_event_count: int


class ComMU4x4Manifest:
    """Explicit ComMU 4/4 split with a sealed test-row firewall."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        repository_root: str | Path,
        expected_sha256: str,
        verify_source_files: bool = True,
        defer_test_rows: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.repository_root = Path(repository_root)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if self.manifest_sha256 != str(expected_sha256):
            raise ValueError("ComMU manifest SHA-256 mismatch")
        sidecar = self.manifest_path.with_suffix(".sha256")
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        if sidecar.read_text(encoding="ascii").split()[0] != self.manifest_sha256:
            raise ValueError("ComMU manifest hash sidecar mismatch")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported ComMU manifest schema")
        self._verify_source_files = bool(verify_source_files)
        self._entries: dict[SplitName, tuple[ComMUManifestEntry, ...] | None] = {}
        self._defer_test_rows = bool(defer_test_rows)
        self.test_reads = 0
        self.test_row_file_reads = 0
        self.test_audit_reads = 0
        self.development_reads = 0
        seen: set[str] = set()
        relation_splits: dict[str, dict[str, SplitName]] = {
            "lineage_id": {},
            "exact_hash": {},
            "transposition_invariant_hash": {},
        }
        expected_counts = self.manifest["split_policy"]["actual"]
        for split in ("train", "validation", "test"):
            path = self.manifest_path.parent / f"{split}.jsonl"
            receipt = self._receipt(path)
            if receipt is None:
                raise ValueError(f"missing receipt for {path}")
            if split == "test" and self._defer_test_rows:
                if int(receipt["rows"]) != int(expected_counts[split]):
                    raise ValueError("sealed ComMU test count mismatch")
                self._entries[split] = None
                continue
            rows = self._read_rows(path, split)
            for entry in rows:
                if entry.sample_id in seen:
                    raise ValueError(f"duplicate ComMU sample ID: {entry.sample_id}")
                seen.add(entry.sample_id)
                for relation, mapping in relation_splits.items():
                    value = str(getattr(entry, relation))
                    previous = mapping.setdefault(value, split)
                    if previous != split:
                        raise ValueError(f"{relation} crosses ComMU split")
            if len(rows) != int(expected_counts[split]):
                raise ValueError(f"{split} ComMU row count mismatch")
            self._entries[split] = rows
        observed = len(seen) + (
            int(expected_counts["test"]) if self._defer_test_rows else 0
        )
        if observed != int(self.manifest["input"]["sample_count"]):
            raise ValueError("ComMU manifest does not cover every admitted sample")

    def _receipt(self, path: Path) -> dict[str, Any] | None:
        for key, value in self.manifest["outputs"].items():
            if Path(key).name == path.name:
                return value
        return None

    def _read_rows(
        self, path: Path, split: SplitName
    ) -> tuple[ComMUManifestEntry, ...]:
        receipt = self._receipt(path)
        if receipt is None or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"ComMU split receipt mismatch: {path}")
        rows: list[ComMUManifestEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                entry = ComMUManifestEntry(
                    sample_id=str(value["sample_id"]),
                    source_path=str(value["source_path"]),
                    source_song_id=str(value["source_song_id"]),
                    lineage_id=str(value["lineage_id"]),
                    exact_hash=str(value["exact_hash"]),
                    transposition_invariant_hash=str(
                        value["transposition_invariant_hash"]
                    ),
                    rhythm_interval_hash=str(value["rhythm_interval_hash"]),
                    note_count=int(value["note_count"]),
                    onset_count=int(value["onset_count"]),
                    duration_beats=float(value["duration_beats"]),
                    preprocessing_version=str(value["preprocessing_version"]),
                    split=split,
                    original_split=str(value["original_split"]),
                    audio_key=str(value["audio_key"]),
                    num_measures=int(value["num_measures"]),
                    target_event_count=int(value["target_event_count"]),
                )
                if self._verify_source_files and not self.source_path(entry).is_file():
                    raise FileNotFoundError(self.source_path(entry))
                rows.append(entry)
        if len(rows) != int(receipt["rows"]):
            raise ValueError(f"ComMU split row receipt mismatch: {path}")
        return tuple(rows)

    def source_path(self, entry: ComMUManifestEntry) -> Path:
        path = Path(entry.source_path)
        return path if path.is_absolute() else self.repository_root / path

    @property
    def train(self) -> tuple[ComMUManifestEntry, ...]:
        rows = self._entries["train"]
        assert rows is not None
        return rows

    @property
    def validation(self) -> tuple[ComMUManifestEntry, ...]:
        self.development_reads += 1
        rows = self._entries["validation"]
        assert rows is not None
        return rows

    @property
    def test_rows_loaded(self) -> bool:
        return self._entries["test"] is not None

    def test(self, *, allow_test_evaluation: bool = False) -> tuple[ComMUManifestEntry, ...]:
        if not allow_test_evaluation:
            raise PermissionError("ComMU clean test is sealed before checkpoint freeze")
        self.test_reads += 1
        if self._entries["test"] is None:
            self._entries["test"] = self._read_rows(
                self.manifest_path.parent / "test.jsonl", "test"
            )
            self.test_row_file_reads += 1
        return self._entries["test"]  # type: ignore[return-value]


def _scan_one(arguments: tuple[str, str, str, str, int]) -> dict[str, Any]:
    sample_id, source_path, audio_key, original_split, num_measures = arguments
    notes, metadata = parse_commu_midi(Path(source_path), expected_audio_key=audio_key)
    exact_hash, transposition_hash, rhythm_hash = canonical_hashes(
        notes, TARGET_TICKS_PER_BEAT
    )
    stream = _stream_from_notes(sample_id, notes)
    target_event_count = len(linearize_note_tokens(stream)) + 1
    first_onset = min(note.onset_tick for note in notes)
    last_end = max(note.onset_tick + note.duration_tick for note in notes)
    return {
        "sample_id": sample_id,
        "source_path": source_path,
        "source_song_id": sample_id,
        "audio_key": audio_key,
        "time_signature": "4/4",
        "original_split": original_split,
        "num_measures": int(num_measures),
        "preprocessing_version": PREPROCESSING_VERSION,
        "tonal_operation": "verified_fixed_C_major_or_A_minor; no pitch transposition",
        "timing_operation": "lossless_integer_rescale_PPQ480_to_PPQ96",
        "exact_hash": exact_hash,
        "transposition_invariant_hash": transposition_hash,
        "rhythm_interval_hash": rhythm_hash,
        "source_ticks_per_beat": metadata["source_ticks_per_beat"],
        "target_ticks_per_beat": metadata["target_ticks_per_beat"],
        "note_count": len(notes),
        "onset_count": len({note.onset_tick for note in notes}),
        "duration_beats": (last_end - first_onset) / TARGET_TICKS_PER_BEAT,
        "target_event_count": target_event_count,
    }


def scan_commu_4_4(
    metadata_path: Path,
    *,
    repository_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    dataset_root = metadata_path.parent
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = [
            row for row in csv.DictReader(handle) if row["time_signature"] == "4/4"
        ]
    tasks: list[tuple[str, str, str, str, int]] = []
    for row in metadata_rows:
        relative = (
            dataset_root
            / "commu_midi"
            / row["split_data"]
            / "raw"
            / f"{row['id']}.mid"
        ).relative_to(repository_root)
        tasks.append(
            (
                row["id"],
                str(repository_root / relative),
                row["audio_key"],
                row["split_data"],
                int(row["num_measures"]),
            )
        )
    if workers <= 1:
        records = [_scan_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            records = list(pool.map(_scan_one, tasks, chunksize=32))
    for row in records:
        row["source_path"] = Path(row["source_path"]).relative_to(repository_root).as_posix()
    records.sort(key=lambda row: row["sample_id"])
    return records


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def assign_commu_splits(
    records: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    """Keep official train/val separation, then halve official val into val/test.

    Exact and global-transposition-equivalent families are indivisible. Rhythm
    hashes remain an audit signal rather than a lineage assertion because many
    short independently composed clips legitimately share rhythm templates.
    """

    union = _UnionFind(len(records))
    duplicates: dict[str, dict[str, list[int]]] = {}
    for field in ("exact_hash", "transposition_invariant_hash"):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(records):
            groups[str(row[field])].append(index)
        duplicates[field] = {
            key: values for key, values in groups.items() if len(values) > 1
        }
        for indices in duplicates[field].values():
            for index in indices[1:]:
                union.union(indices[0], index)

    rhythm_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        rhythm_groups[str(row["rhythm_interval_hash"])].append(index)
    duplicate_rhythm = {
        key: values for key, values in rhythm_groups.items() if len(values) > 1
    }

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union.find(index)].append(index)
    component_rows: list[dict[str, Any]] = []
    for indices in components.values():
        ids = sorted(records[index]["sample_id"] for index in indices)
        lineage_id = "commu_family_" + sha256_bytes(
            ("\n".join(ids) + "\n").encode("ascii")
        )[:24]
        original_splits = {records[index]["original_split"] for index in indices}
        heldout = "val" in original_splits
        for index in indices:
            records[index]["lineage_id"] = lineage_id
            records[index]["lineage_status"] = (
                "content_duplicate_family" if len(indices) > 1 else "official_source_singleton"
            )
        component_rows.append(
            {
                "lineage_id": lineage_id,
                "indices": tuple(indices),
                "sample_count": len(indices),
                "target_event_count": sum(
                    int(records[index]["target_event_count"]) for index in indices
                ),
                "heldout": heldout,
            }
        )

    heldout = [component for component in component_rows if component["heldout"]]
    heldout.sort(
        key=lambda component: (
            -int(component["target_event_count"]),
            sha256_bytes(
                f"{seed}|commu-heldout|{component['lineage_id']}".encode("ascii")
            ),
        )
    )
    total_heldout = sum(component["sample_count"] for component in heldout)
    total_heldout_events = sum(
        int(component["target_event_count"]) for component in heldout
    )
    validation_row_target = total_heldout // 2
    test_row_target = total_heldout - validation_row_target
    event_target = total_heldout_events / 2.0
    validation_count = 0
    test_count = 0
    validation_events = 0
    test_events = 0
    for component in heldout:
        size = int(component["sample_count"])
        events = int(component["target_event_count"])
        if validation_count + size > validation_row_target:
            put_validation = False
        elif test_count + size > test_row_target:
            put_validation = True
        else:
            validation_load = (
                validation_count / max(1, validation_row_target)
                + validation_events / max(1.0, event_target)
            )
            test_load = (
                test_count / max(1, test_row_target)
                + test_events / max(1.0, event_target)
            )
            put_validation = validation_load <= test_load
        split = "validation" if put_validation else "test"
        if put_validation:
            validation_count += size
            validation_events += events
        else:
            test_count += size
            test_events += events
        component["assigned_split"] = split

    # Preserve the exact row allocation while exchanging equally sized
    # components to minimize validation/test target-exposure imbalance.
    for _ in range(16):
        current_error = abs(validation_events - event_target)
        best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        validation_components = [
            component
            for component in heldout
            if component["assigned_split"] == "validation"
        ]
        test_components = [
            component
            for component in heldout
            if component["assigned_split"] == "test"
        ]
        for left in validation_components:
            for right in test_components:
                if int(left["sample_count"]) != int(right["sample_count"]):
                    continue
                candidate_events = (
                    validation_events
                    - int(left["target_event_count"])
                    + int(right["target_event_count"])
                )
                candidate_error = abs(candidate_events - event_target)
                if candidate_error + 1e-9 < current_error and (
                    best is None or candidate_error < best[0]
                ):
                    best = (candidate_error, left, right)
        if best is None:
            break
        _error, left, right = best
        left["assigned_split"] = "test"
        right["assigned_split"] = "validation"
        validation_events = (
            validation_events
            - int(left["target_event_count"])
            + int(right["target_event_count"])
        )
        test_events = total_heldout_events - validation_events

    for component in heldout:
        for index in component["indices"]:
            records[index]["split"] = component["assigned_split"]
    for component in component_rows:
        if component["heldout"]:
            continue
        for index in component["indices"]:
            records[index]["split"] = "train"

    relation_crossings: dict[str, int] = {}
    for field in (
        "lineage_id",
        "exact_hash",
        "transposition_invariant_hash",
        "rhythm_interval_hash",
    ):
        observed: dict[str, set[str]] = defaultdict(set)
        for row in records:
            observed[str(row[field])].add(str(row["split"]))
        relation_crossings[field] = sum(len(splits) > 1 for splits in observed.values())
    if any(
        relation_crossings[field]
        for field in ("lineage_id", "exact_hash", "transposition_invariant_hash")
    ):
        raise AssertionError("ComMU duplicate-family split leakage")
    return {
        "component_count": len(component_rows),
        "content_duplicate_families": sum(
            int(component["sample_count"]) > 1 for component in component_rows
        ),
        "exact_duplicate_groups": len(duplicates["exact_hash"]),
        "transposition_duplicate_groups": len(
            duplicates["transposition_invariant_hash"]
        ),
        "rhythm_interval_duplicate_groups": len(duplicate_rhythm),
        "cross_split_groups": relation_crossings,
        "official_val_rows_in_heldout": total_heldout,
        "official_val_family_target_events": total_heldout_events,
    }


def write_commu_manifest(
    records: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    metadata_path: Path,
    seed: int,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows = sorted(
            (row for row in records if row["split"] == split),
            key=lambda row: row["sample_id"],
        )
        path = output_dir / f"{split}.jsonl"
        _atomic_text(
            path,
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in rows
            ),
        )
        counts[split] = len(rows)
        target_counts[split] = sum(int(row["target_event_count"]) for row in rows)
        outputs[path.name] = {"rows": len(rows), "sha256": sha256_file(path)}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_4_4_FIXED_TONAL_FRAME",
        "input": {
            "dataset": "ComMU official raw MIDI",
            "metadata_path": metadata_path.as_posix(),
            "metadata_sha256": sha256_file(metadata_path),
            "sample_count": len(records),
            "admission": "metadata time_signature=4/4 and MIDI key signature C or Am",
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "tonal_frame": "official C major / A minor; verified, no transposition",
            "source_ppq": SOURCE_TICKS_PER_BEAT,
            "model_ppq": TARGET_TICKS_PER_BEAT,
            "time_conversion": "exact integer rescale; no rounding or quantization",
            "notes": "all non-drum notes; no role, chord, instrument, or melody labels",
            "minimum_semantic_rest_ticks": TARGET_TICKS_PER_BEAT,
        },
        "split_policy": {
            "seed": int(seed),
            "official_train": "train unless connected to official val by exact/transposition hash",
            "official_val": "content-family-level deterministic 1:1 validation/test split",
            "family_relations": ["exact_hash", "transposition_invariant_hash"],
            "rhythm_interval_hash": "audit-only near-duplicate signal",
            "actual": counts,
            "target_events": target_counts,
        },
        "audit": dict(audit),
        "outputs": outputs,
    }
    manifest_path = output_dir / "split_manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    digest = sha256_file(manifest_path)
    _atomic_text(output_dir / "split_manifest.sha256", f"{digest}  split_manifest.json\n")
    return {"manifest_sha256": digest, "manifest": manifest}


def load_commu_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json_yaml(path)
    if protocol.get("schema_version") != "m4l.commu_ad_replication.formal_protocol.v1":
        raise ValueError("unsupported ComMU A/D protocol schema")
    if tuple(protocol["scientific_scope"]["arms"]) != (
        "ARM_A_RAW_SEQUENCE",
        "ARM_D_RELATIONAL_TIME_GEOMETRY",
    ):
        raise ValueError("ComMU replication must contain only matched A/D")
    if len(set(int(seed) for seed in protocol["preregistered_seeds"])) != 3:
        raise ValueError("ComMU A/D protocol requires three distinct seeds")
    per_epoch = int(protocol["data"]["train_target_events_per_equivalent_epoch"])
    epochs = int(protocol["training"]["target_equivalent_epochs"])
    if int(protocol["training"]["target_event_budget"]) != per_epoch * epochs:
        raise ValueError("ComMU target-event budget is not an exact epoch multiple")
    if protocol["validation_and_checkpointing"]["test_evaluation"] != "disabled_and_not_loaded":
        raise ValueError("ComMU formal training must keep test sealed")
    return protocol


def commu_stream(entry: ComMUManifestEntry, repository_root: Path) -> SparseMelodyStream:
    notes, _metadata = parse_commu_midi(
        repository_root / entry.source_path,
        expected_audio_key=entry.audio_key,
    )
    stream = _stream_from_notes(entry.sample_id, notes)
    if len(linearize_note_tokens(stream)) + 1 != entry.target_event_count:
        raise ValueError(f"ComMU target count changed for {entry.sample_id}")
    return stream


def commu_source_paths(root: Path, protocol_path: Path) -> list[Path]:
    return [
        root / "unified_structured_music/commu_ad_replication.py",
        root / "unified_structured_music/iclr_matched_harness.py",
        root / "unified_structured_music/iclr_formal_training.py",
        root / "unified_structured_music/iclr_cached_training.py",
        root / "unified_structured_music/note_centric_music.py",
        root / "unified_structured_music/dual_coordinate_melody.py",
        root / "unified_structured_music/discrete_music_primitive.py",
        root / "unified_structured_music/musical_time_melody.py",
        root / "unified_structured_music/sparse_melody_bpe.py",
        root / "configs/iclr_ablation/arm_a_raw_sequence.yaml",
        root / "configs/iclr_ablation/arm_d_relational_time.yaml",
        protocol_path,
        root / "tools/build_commu_4_4_clean_split.py",
        root / "tools/build_commu_ad_tensor_cache.py",
        root / "tools/run_commu_ad_launch_gate.py",
        root / "tools/run_commu_ad_formal_arm.py",
        root / "cloud/run_commu_ad_replication.sh",
    ]
