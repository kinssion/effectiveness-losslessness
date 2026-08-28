from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from .discrete_music_primitive import (
    M4E_MAX_DURATION_BARS,
    M4E_TICKS_PER_BEAT,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .iclr_matched_harness import _stream_from_notes
from .note_centric_music import (
    INPUT_BOS,
    INPUT_NOTE,
    NoteCausalBatch,
    TARGET_EOS,
    TARGET_NOTE,
    TARGET_REST,
    linearize_note_tokens,
)
from .popk_clean_manifest import PopKCleanManifest, PopKManifestEntry
from .popk_clean_split import _parse_midi_bytes
from .sparse_melody_bpe import SparseMelodyStream


CACHE_SCHEMA = "m4l.iclr.shared_note_tensor_cache.v1"
TOKEN_DTYPE = np.dtype(
    [
        ("kind", "u1"),
        ("anchor", "<i4"),
        ("pitch", "u1"),
        ("time_bars", "<u2"),
        ("time_remainder", "<u2"),
        ("duration_bars", "<u2"),
        ("duration_remainder", "<u2"),
    ],
    align=False,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoded_tokens(stream: SparseMelodyStream) -> np.ndarray:
    tokens = linearize_note_tokens(stream)
    values = np.empty(len(tokens), dtype=TOKEN_DTYPE)
    previous_anchor = 0
    for index, token in enumerate(tokens):
        delta = int(token.anchor) - int(previous_anchor)
        if delta < 0:
            raise AssertionError("cached note time moves backward")
        time_bars, time_remainder = divmod(delta, M4E_TICKS_PER_SEMANTIC_BAR)
        duration_bars, duration_remainder = divmod(
            int(token.duration), M4E_TICKS_PER_SEMANTIC_BAR
        )
        if time_bars >= M4E_MAX_DURATION_BARS:
            raise ValueError("cached onset delta leaves model support")
        if duration_bars >= M4E_MAX_DURATION_BARS:
            raise ValueError("cached duration leaves model support")
        values[index] = (
            int(token.kind),
            int(token.anchor),
            int(token.pitch),
            int(time_bars),
            int(time_remainder),
            int(duration_bars),
            int(duration_remainder),
        )
        previous_anchor = int(token.anchor)
    return values


@dataclass(frozen=True, slots=True)
class CacheSourceTask:
    split: str
    sample_id: str
    path: str
    expected_notes: int
    expected_onsets: int
    maximum_sequence_tokens: int
    maximum_context_ticks: int


def _encode_source_task(task: CacheSourceTask) -> tuple[str, str, np.ndarray]:
    notes, metadata = _parse_midi_bytes(Path(task.path).read_bytes())
    if int(metadata["ticks_per_beat"]) != M4E_TICKS_PER_BEAT:
        raise ValueError(f"{task.sample_id} has non-frozen PPQ")
    if len(notes) != int(task.expected_notes):
        raise ValueError(f"{task.sample_id} note count differs from manifest")
    if len({note.onset_tick for note in notes}) != int(task.expected_onsets):
        raise ValueError(f"{task.sample_id} onset count differs from manifest")
    stream = _stream_from_notes(task.sample_id, notes)
    encoded = _encoded_tokens(stream)
    if len(encoded) + 1 > int(task.maximum_sequence_tokens):
        raise ValueError(f"{task.sample_id} exceeds frozen sequence support")
    if len(encoded) and int(encoded["anchor"].max()) >= int(task.maximum_context_ticks):
        raise ValueError(f"{task.sample_id} exceeds frozen context support")
    return task.split, task.sample_id, encoded


def _write_cache_rows(
    output: Path,
    *,
    rows: Iterable[tuple[str, str, np.ndarray]],
    manifest_sha256: str,
    maximum_sequence_tokens: int,
    maximum_context_bars: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"cache output already exists: {output}")
    temporary = output.with_name(f"{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"cache temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    token_path = temporary / "tokens.bin"
    ids_path = temporary / "samples.jsonl"
    offsets: list[int] = [0]
    split_counts: dict[str, int] = {}
    with token_path.open("wb") as token_handle, ids_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as ids_handle:
        for split, sample_id, encoded in rows:
            encoded.tofile(token_handle)
            offsets.append(offsets[-1] + len(encoded))
            split_counts[split] = split_counts.get(split, 0) + 1
            ids_handle.write(
                json.dumps(
                    {
                        "cache_row": len(offsets) - 2,
                        "sample_id": sample_id,
                        "split": split,
                        "token_count": len(encoded),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    offsets_path = temporary / "offsets.npy"
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    metadata = {
        "schema_version": CACHE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "maximum_sequence_tokens": int(maximum_sequence_tokens),
        "maximum_context_bars": int(maximum_context_bars),
        "sample_count": len(offsets) - 1,
        "token_count": offsets[-1],
        "split_counts": split_counts,
        "token_dtype": TOKEN_DTYPE.descr,
        "files": {
            "tokens.bin": _sha256_file(token_path),
            "offsets.npy": _sha256_file(offsets_path),
            "samples.jsonl": _sha256_file(ids_path),
        },
    }
    metadata_path = temporary / "cache_manifest.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return metadata


def write_stream_cache(
    output: Path,
    *,
    rows: Sequence[tuple[str, SparseMelodyStream]],
    manifest_sha256: str,
    maximum_sequence_tokens: int,
    maximum_context_bars: int,
) -> dict[str, Any]:
    """Write a deterministic cache from already materialized streams."""

    return _write_cache_rows(
        output,
        rows=((split, stream.song_id, _encoded_tokens(stream)) for split, stream in rows),
        manifest_sha256=manifest_sha256,
        maximum_sequence_tokens=maximum_sequence_tokens,
        maximum_context_bars=maximum_context_bars,
    )


def build_manifest_tensor_cache(
    manifest: PopKCleanManifest,
    output: Path,
    *,
    workers: int,
    maximum_sequence_tokens: int,
    maximum_context_bars: int,
    train_limit: int | None = None,
    validation_limit: int | None = None,
) -> dict[str, Any]:
    """Parse each unsealed source exactly once and emit a shared mmap cache."""

    if manifest.test_rows_loaded or manifest.test_reads or manifest.test_row_file_reads:
        raise AssertionError("cache builder read sealed test rows")
    train = manifest.train[:train_limit] if train_limit is not None else manifest.train
    validation = (
        manifest.validation[:validation_limit]
        if validation_limit is not None
        else manifest.validation
    )
    maximum_context_ticks = int(maximum_context_bars) * 4 * M4E_TICKS_PER_BEAT
    entries = [("train", entry) for entry in train] + [
        ("validation", entry) for entry in validation
    ]
    tasks = [
        CacheSourceTask(
            split=split,
            sample_id=entry.sample_id,
            path=str(manifest.source_path(entry)),
            expected_notes=int(entry.note_count),
            expected_onsets=int(entry.onset_count),
            maximum_sequence_tokens=int(maximum_sequence_tokens),
            maximum_context_ticks=maximum_context_ticks,
        )
        for split, entry in entries
    ]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        encoded_rows = pool.map(_encode_source_task, tasks, chunksize=16)
        metadata = _write_cache_rows(
            output,
            rows=encoded_rows,
            manifest_sha256=manifest.manifest_sha256,
            maximum_sequence_tokens=maximum_sequence_tokens,
            maximum_context_bars=maximum_context_bars,
        )
    if manifest.test_rows_loaded or manifest.test_reads or manifest.test_row_file_reads:
        raise AssertionError("cache builder read sealed test rows")
    return metadata


class SharedNoteTensorCache:
    def __init__(
        self,
        root: Path,
        *,
        expected_manifest_sha256: str,
        verify_files: bool = True,
    ) -> None:
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "cache_manifest.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("unsupported tensor cache schema")
        if self.metadata.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("tensor cache manifest hash mismatch")
        if verify_files:
            for name, expected in self.metadata["files"].items():
                if _sha256_file(self.root / name) != expected:
                    raise ValueError(f"tensor cache file hash mismatch: {name}")
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        if len(self.offsets) != int(self.metadata["sample_count"]) + 1:
            raise ValueError("tensor cache offset count mismatch")
        if int(self.offsets[-1]) != int(self.metadata["token_count"]):
            raise ValueError("tensor cache token count mismatch")
        self.tokens = np.memmap(
            self.root / "tokens.bin",
            dtype=TOKEN_DTYPE,
            mode="r",
            shape=(int(self.metadata["token_count"]),),
        )
        self.sample_rows: dict[str, int] = {}
        self.sample_splits: dict[str, str] = {}
        with (self.root / "samples.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if sample_id in self.sample_rows:
                    raise ValueError(f"duplicate cache sample: {sample_id}")
                self.sample_rows[sample_id] = int(row["cache_row"])
                self.sample_splits[sample_id] = str(row["split"])
        if len(self.sample_rows) != int(self.metadata["sample_count"]):
            raise ValueError("tensor cache sample index mismatch")

    def row(self, sample_id: str) -> np.ndarray:
        index = self.sample_rows[sample_id]
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.tokens[start:end]

    def batch(
        self,
        entries: Sequence[PopKManifestEntry],
        *,
        pin_memory: bool = False,
    ) -> NoteCausalBatch:
        return tensorize_cached_rows(
            [self.row(entry.sample_id) for entry in entries],
            pin_memory=pin_memory,
        )


def tensorize_cached_rows(
    rows: Sequence[np.ndarray], *, pin_memory: bool = False
) -> NoteCausalBatch:
    if not rows:
        raise ValueError("cannot tensorize an empty cached batch")
    length = max(len(row) + 1 for row in rows)
    batch_size = len(rows)

    def zeros(dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((batch_size, length), dtype=dtype)

    input_type = zeros(torch.long)
    input_pitch = zeros(torch.long)
    input_duration_bars = zeros(torch.long)
    input_duration_remainders = zeros(torch.long)
    input_anchor_ticks = zeros(torch.long)
    valid = zeros(torch.bool)
    target_type = torch.full((batch_size, length), TARGET_EOS, dtype=torch.long)
    target_time_bars = zeros(torch.long)
    target_time_remainders = zeros(torch.long)
    target_pitch = zeros(torch.long)
    target_duration_bars = zeros(torch.long)
    target_duration_remainders = zeros(torch.long)
    note_mask = zeros(torch.bool)
    support_mask = zeros(torch.bool)

    for row_index, encoded in enumerate(rows):
        count = len(encoded)
        valid[row_index, : count + 1] = True
        input_type[row_index, 0] = INPUT_BOS
        if count == 0:
            continue
        kinds = torch.from_numpy(np.asarray(encoded["kind"], dtype=np.int64))
        pitches = torch.from_numpy(np.asarray(encoded["pitch"], dtype=np.int64))
        time_bars = torch.from_numpy(np.asarray(encoded["time_bars"], dtype=np.int64))
        time_remainders = torch.from_numpy(
            np.asarray(encoded["time_remainder"], dtype=np.int64)
        )
        duration_bars = torch.from_numpy(
            np.asarray(encoded["duration_bars"], dtype=np.int64)
        )
        duration_remainders = torch.from_numpy(
            np.asarray(encoded["duration_remainder"], dtype=np.int64)
        )
        anchors = torch.from_numpy(np.asarray(encoded["anchor"], dtype=np.int64))
        is_note = kinds == INPUT_NOTE
        target_slice = slice(0, count)
        input_slice = slice(1, count + 1)
        input_type[row_index, input_slice] = kinds
        input_pitch[row_index, input_slice] = pitches
        input_duration_bars[row_index, input_slice] = duration_bars
        input_duration_remainders[row_index, input_slice] = duration_remainders
        input_anchor_ticks[row_index, input_slice] = anchors
        target_type[row_index, target_slice] = torch.where(
            is_note,
            torch.full_like(kinds, TARGET_NOTE),
            torch.full_like(kinds, TARGET_REST),
        )
        target_time_bars[row_index, target_slice] = time_bars
        target_time_remainders[row_index, target_slice] = time_remainders
        target_pitch[row_index, target_slice] = pitches
        target_duration_bars[row_index, target_slice] = duration_bars
        target_duration_remainders[row_index, target_slice] = duration_remainders
        note_mask[row_index, target_slice] = is_note
        support_mask[row_index, target_slice] = True

    result = NoteCausalBatch(
        input_type=input_type,
        input_pitch=input_pitch,
        input_duration_bars=input_duration_bars,
        input_duration_remainders=input_duration_remainders,
        input_anchor_ticks=input_anchor_ticks,
        valid=valid,
        target_type=target_type,
        target_time_bars=target_time_bars,
        target_time_remainders=target_time_remainders,
        target_pitch=target_pitch,
        target_duration_bars=target_duration_bars,
        target_duration_remainders=target_duration_remainders,
        note_mask=note_mask,
        support_mask=support_mask,
    )
    return result.pin_memory() if pin_memory else result


class CachedBatchPrefetcher:
    """Prepare the next cached CPU batch while the current batch trains."""

    def __init__(
        self,
        cache: SharedNoteTensorCache,
        entry_batches: Sequence[Sequence[PopKManifestEntry]],
        *,
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        self.cache = cache
        self.entry_batches = entry_batches
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")

    def __iter__(self) -> Iterator[NoteCausalBatch]:
        if not self.entry_batches:
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="m4l-cache") as pool:
            future: Future[NoteCausalBatch] = pool.submit(
                self.cache.batch,
                self.entry_batches[0],
                pin_memory=self.pin_memory,
            )
            for next_index in range(1, len(self.entry_batches) + 1):
                cpu_batch = future.result()
                if next_index < len(self.entry_batches):
                    future = pool.submit(
                        self.cache.batch,
                        self.entry_batches[next_index],
                        pin_memory=self.pin_memory,
                    )
                yield cpu_batch.to(
                    self.device,
                    non_blocking=self.pin_memory,
                )
