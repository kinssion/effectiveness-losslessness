from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from .iclr_cached_training import SharedNoteTensorCache
from .iclr_matched_harness import UnifiedMatchedMusicModel, _decoder_predictions
from .note_centric_music import (
    INPUT_BOS,
    INPUT_NOTE,
    NoteCausalBatch,
    TARGET_EOS,
    TARGET_NOTE,
    TARGET_REST,
)
from .popk_clean_manifest import PopKManifestEntry


def tensorize_cached_rows_v2(
    rows: Sequence[np.ndarray], *, pin_memory: bool = False
) -> NoteCausalBatch:
    """Tensorize cached rows without a second pageable-to-pinned copy.

    This is value-identical to ``tensorize_cached_rows``.  When CUDA transfer
    is requested, destination tensors are allocated in pinned memory directly
    rather than allocating pageable tensors and cloning the completed batch.
    """

    if not rows:
        raise ValueError("cannot tensorize an empty cached batch")
    length = max(len(row) + 1 for row in rows)
    batch_size = len(rows)

    def zeros(dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            (batch_size, length), dtype=dtype, pin_memory=pin_memory
        )

    input_type = zeros(torch.long)
    input_pitch = zeros(torch.long)
    input_duration_bars = zeros(torch.long)
    input_duration_remainders = zeros(torch.long)
    input_anchor_ticks = zeros(torch.long)
    valid = zeros(torch.bool)
    target_type = torch.full(
        (batch_size, length),
        TARGET_EOS,
        dtype=torch.long,
        pin_memory=pin_memory,
    )
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

    return NoteCausalBatch(
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


@dataclass(frozen=True, slots=True)
class PreparedDeviceBatch:
    batch: NoteCausalBatch
    token_count: int


class FastCachedBatchPrefetcher:
    """Bounded, ordered cache materialization with CPU-side batch metadata."""

    def __init__(
        self,
        cache: SharedNoteTensorCache,
        entry_batches: Sequence[Sequence[PopKManifestEntry]],
        *,
        device: torch.device | str,
        pin_memory: bool = True,
        workers: int = 2,
        prefetch_depth: int = 4,
        direct_pinned_allocation: bool = False,
    ) -> None:
        self.cache = cache
        self.entry_batches = entry_batches
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")
        self.workers = max(1, int(workers))
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.direct_pinned_allocation = bool(direct_pinned_allocation)

    def _prepare(self, index: int) -> tuple[NoteCausalBatch, int]:
        entries = self.entry_batches[index]
        if self.direct_pinned_allocation:
            rows = [self.cache.row(entry.sample_id) for entry in entries]
            batch = tensorize_cached_rows_v2(rows, pin_memory=self.pin_memory)
        else:
            batch = self.cache.batch(entries, pin_memory=self.pin_memory)
        return batch, int(batch.valid.sum().item())

    def __iter__(self) -> Iterator[PreparedDeviceBatch]:
        total = len(self.entry_batches)
        if not total:
            return
        depth = min(total, self.prefetch_depth)
        with ThreadPoolExecutor(
            max_workers=min(self.workers, depth), thread_name_prefix="m4l-cache-v2"
        ) as pool:
            futures: dict[int, Future[tuple[NoteCausalBatch, int]]] = {
                index: pool.submit(self._prepare, index) for index in range(depth)
            }
            next_submit = depth
            for index in range(total):
                cpu_batch, token_count = futures.pop(index).result()
                if next_submit < total:
                    futures[next_submit] = pool.submit(self._prepare, next_submit)
                    next_submit += 1
                yield PreparedDeviceBatch(
                    batch=cpu_batch.to(
                        self.device, non_blocking=self.pin_memory
                    ),
                    token_count=token_count,
                )


@dataclass(frozen=True, slots=True)
class StaticValidationBatch:
    entries: tuple[PopKManifestEntry, ...]
    batch: NoteCausalBatch
    token_count: int


def build_static_validation_batches(
    cache: SharedNoteTensorCache,
    entries: Sequence[PopKManifestEntry],
    batch_indices: Sequence[Sequence[int]],
    *,
    pin_memory: bool,
) -> list[StaticValidationBatch]:
    result: list[StaticValidationBatch] = []
    for indices in batch_indices:
        selected = tuple(entries[index] for index in indices)
        rows = [cache.row(entry.sample_id) for entry in selected]
        batch = tensorize_cached_rows_v2(rows, pin_memory=pin_memory)
        result.append(
            StaticValidationBatch(
                entries=selected,
                batch=batch,
                token_count=int(batch.valid.sum().item()),
            )
        )
    return result


def evaluate_cached_v2(
    model: UnifiedMatchedMusicModel,
    *,
    batches: Sequence[StaticValidationBatch],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validation-equivalent evaluator with one host synchronization per batch."""

    model.eval()
    names = ("total", "type", "time", "pitch", "duration")
    totals = {name: 0.0 for name in names}
    events = notes = support = eos_predictions = invalid_predictions = 0
    song_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for prepared in batches:
            batch = prepared.batch.to(
                device, non_blocking=bool(prepared.batch.valid.is_pinned())
            )
            losses, _representation, hidden = model.losses(batch)
            predictions = _decoder_predictions(model, hidden, batch)

            row_losses = torch.stack(
                tuple(losses[name].sum(dim=1) for name in names), dim=1
            )
            row_counts = torch.stack(
                (
                    batch.valid.sum(dim=1),
                    batch.note_mask.sum(dim=1),
                    batch.support_mask.sum(dim=1),
                ),
                dim=1,
            )
            predicted_support = batch.valid & (predictions["type"] != TARGET_EOS)
            invalid = predicted_support & (predictions["duration_bar"] == 0) & (
                predictions["duration_remainder"] == 0
            )
            aggregate_predictions = torch.stack(
                (
                    ((predictions["type"] == TARGET_EOS) & batch.valid).sum(),
                    invalid.sum(),
                )
            )
            row_losses_cpu = row_losses.detach().cpu()
            row_counts_cpu = row_counts.detach().cpu()
            aggregate_cpu = aggregate_predictions.detach().cpu()

            for row, entry in enumerate(prepared.entries):
                row_events, row_notes, row_support = (
                    int(value) for value in row_counts_cpu[row]
                )
                row_values = {
                    name: float(row_losses_cpu[row, column])
                    for column, name in enumerate(names)
                }
                for name, value in row_values.items():
                    totals[name] += value
                song_rows.append(
                    {
                        "sample_id": entry.sample_id,
                        "event_count": row_events,
                        "note_count": row_notes,
                        "support_event_count": row_support,
                        "total_bits_per_event": row_values["total"]
                        / max(1, row_events)
                        / math.log(2.0),
                        "type_nll_nats_per_event": row_values["type"]
                        / max(1, row_events),
                        "time_nll_nats_per_support_event": row_values["time"]
                        / max(1, row_support),
                        "pitch_nll_nats_per_note": row_values["pitch"]
                        / max(1, row_notes),
                        "duration_nll_nats_per_support_event": row_values["duration"]
                        / max(1, row_support),
                    }
                )
                events += row_events
                notes += row_notes
                support += row_support
            eos_predictions += int(aggregate_cpu[0])
            invalid_predictions += int(aggregate_cpu[1])

    elapsed = max(time.perf_counter() - started, 1e-9)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return (
        {
            "primary_metric_name": "micro_total_validation_bits_per_exact_musical_event",
            "total_bits_per_event": totals["total"] / max(1, events) / math.log(2.0),
            "type_nll_nats_per_event": totals["type"] / max(1, events),
            "time_nll_nats_per_support_event": totals["time"] / max(1, support),
            "pitch_nll_nats_per_note": totals["pitch"] / max(1, notes),
            "duration_nll_nats_per_support_event": totals["duration"]
            / max(1, support),
            "predicted_eos_rate": eos_predictions / max(1, events),
            "invalid_event_rate": invalid_predictions / max(1, events),
            "evaluated_event_count": events,
            "evaluated_note_count": notes,
            "evaluated_song_count": sum(len(item.entries) for item in batches),
            "throughput_events_per_second": events / elapsed,
            "peak_memory_bytes": peak,
        },
        song_rows,
    )
