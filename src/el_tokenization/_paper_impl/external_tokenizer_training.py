from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .external_tokenizers import (
    LM_BOS,
    LM_EOS,
    LM_PAD,
    LM_PAYLOAD_OFFSET,
    SerializedTokenCache,
)
from .popk_clean_manifest import PopKManifestEntry


EXTERNAL_LM_SCHEMA = "m4l.external_tokenizer_lm.v1"


@dataclass(slots=True)
class SerializedLMBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    valid: torch.Tensor
    original_event_counts: torch.Tensor

    @property
    def serialized_target_count(self) -> int:
        return int(self.valid.sum().item())

    @property
    def original_event_count(self) -> int:
        return int(self.original_event_counts.sum().item())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "SerializedLMBatch":
        return SerializedLMBatch(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            target_ids=self.target_ids.to(device, non_blocking=non_blocking),
            valid=self.valid.to(device, non_blocking=non_blocking),
            original_event_counts=self.original_event_counts.to(
                device, non_blocking=non_blocking
            ),
        )

    def pin_memory(self) -> "SerializedLMBatch":
        return SerializedLMBatch(
            input_ids=self.input_ids.pin_memory(),
            target_ids=self.target_ids.pin_memory(),
            valid=self.valid.pin_memory(),
            original_event_counts=self.original_event_counts.pin_memory(),
        )


def tensorize_serialized_payloads(
    payloads: Sequence[Sequence[int]],
    original_event_counts: Sequence[int],
    *,
    payload_vocabulary_size: int,
    pin_memory: bool = False,
) -> SerializedLMBatch:
    if not payloads or len(payloads) != len(original_event_counts):
        raise ValueError("serialized batch rows/counts differ or are empty")
    length = max(len(payload) + 1 for payload in payloads)
    batch_size = len(payloads)
    input_ids = torch.full((batch_size, length), LM_PAD, dtype=torch.long)
    target_ids = torch.full((batch_size, length), LM_PAD, dtype=torch.long)
    valid = torch.zeros((batch_size, length), dtype=torch.bool)
    for row_index, payload in enumerate(payloads):
        # Cache rows are read-only uint32 memory maps.  Converting explicitly
        # to a writable int64 array keeps behavior stable across PyTorch
        # versions (some builds reject uint32 in torch.tensor()).
        values = torch.from_numpy(
            np.asarray(payload, dtype=np.int64).copy()
        )
        if values.numel() == 0:
            raise ValueError("serialized LM payload cannot be empty")
        if int(values.min()) < 0 or int(values.max()) >= int(payload_vocabulary_size):
            raise ValueError("serialized LM payload leaves vocabulary")
        count = int(values.numel())
        shifted = values + LM_PAYLOAD_OFFSET
        valid[row_index, : count + 1] = True
        input_ids[row_index, 0] = LM_BOS
        input_ids[row_index, 1 : count + 1] = shifted
        target_ids[row_index, :count] = shifted
        target_ids[row_index, count] = LM_EOS
    result = SerializedLMBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        valid=valid,
        original_event_counts=torch.as_tensor(
            original_event_counts, dtype=torch.long
        ),
    )
    return result.pin_memory() if pin_memory else result


class GenericCausalBlock(nn.Module):
    """The same pre-LN MHA/FFN core as the frozen matched backbone.

    Generic tokenizer baselines use a standard causal mask and padding mask
    instead of M4L's pairwise musical-time attention bias.
    """

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        feedforward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        valid: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm_attention(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = values + attended
        values = values + self.feedforward(self.norm_feedforward(values))
        return values.masked_fill(~valid.unsqueeze(-1), 0.0)


class ExternalTokenizerCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        width: int = 64,
        heads: int = 4,
        layers: int = 16,
        feedforward_width: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocabulary_size = int(vocabulary_size)
        self.width = int(width)
        self.heads = int(heads)
        # Instantiate the shared core first.  This makes its seeded initial
        # weights byte-identical across J/K despite their different vocabularies.
        self.blocks = nn.ModuleList(
            GenericCausalBlock(
                width=width,
                heads=heads,
                feedforward_width=feedforward_width,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.token_embedding = nn.Embedding(vocabulary_size, width, padding_idx=LM_PAD)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.output_normalization = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, vocabulary_size)

    def _sequence_position(
        self, length: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype)
        half = max(1, self.width // 2)
        scale = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10_000.0) / max(1, half - 1))
        )
        angle = position[:, None] * scale[None, :]
        values = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)[
            :, : self.width
        ]
        if values.shape[-1] < self.width:
            values = F.pad(values, (0, self.width - values.shape[-1]))
        return values

    def hidden_sequence(self, batch: SerializedLMBatch) -> torch.Tensor:
        values = self.token_embedding(batch.input_ids)
        position = self._sequence_position(
            values.shape[1], device=values.device, dtype=values.dtype
        )
        values = self.input_projection(values + position.unsqueeze(0))
        values = values.masked_fill(~batch.valid.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones(
                (values.shape[1], values.shape[1]),
                dtype=torch.bool,
                device=values.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            values = block(values, valid=batch.valid, causal_mask=causal_mask)
        return self.output_normalization(values)

    def losses(self, batch: SerializedLMBatch) -> torch.Tensor:
        logits = self.output_head(self.hidden_sequence(batch))
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(batch.target_ids)
        return nll * batch.valid


def external_parameter_receipt(model: ExternalTokenizerCausalLM) -> dict[str, int]:
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    return {
        "total_parameters": count(model),
        "effective_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "token_embedding_parameters": count(model.token_embedding),
        "input_projection_parameters": count(model.input_projection),
        "backbone_parameters": count(model.blocks) + count(model.output_normalization),
        "decoder_parameters": count(model.output_head),
    }


class SerializedCacheBatcher:
    def __init__(
        self,
        cache: SerializedTokenCache,
        *,
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        self.cache = cache
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")
        self.payload_vocabulary_size = int(
            cache.metadata["payload_vocabulary_size"]
        )

    def cpu_batch(self, entries: Sequence[PopKManifestEntry]) -> SerializedLMBatch:
        return tensorize_serialized_payloads(
            [self.cache.row(entry.sample_id) for entry in entries],
            [self.cache.original_event_count(entry.sample_id) for entry in entries],
            payload_vocabulary_size=self.payload_vocabulary_size,
            pin_memory=self.pin_memory,
        )

    def batches(
        self,
        entry_batches: Sequence[Sequence[PopKManifestEntry]],
    ) -> Iterator[SerializedLMBatch]:
        if not entry_batches:
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="m4l-serialized") as pool:
            future: Future[SerializedLMBatch] = pool.submit(
                self.cpu_batch, entry_batches[0]
            )
            for next_index in range(1, len(entry_batches) + 1):
                cpu_batch = future.result()
                if next_index < len(entry_batches):
                    future = pool.submit(self.cpu_batch, entry_batches[next_index])
                yield cpu_batch.to(self.device, non_blocking=self.pin_memory)


def evaluate_external_lm(
    model: ExternalTokenizerCausalLM,
    *,
    entries: Sequence[PopKManifestEntry],
    batch_indices: Sequence[Sequence[int]],
    batcher: SerializedCacheBatcher,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_original_events = 0
    song_rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    selected_batches = [
        [entries[index] for index in indices] for indices in batch_indices
    ]
    with torch.no_grad():
        for selected, batch in zip(selected_batches, batcher.batches(selected_batches)):
            losses = model.losses(batch)
            for row, entry in enumerate(selected):
                token_count = int(batch.valid[row].sum().item())
                event_count = int(batch.original_event_counts[row].item())
                nats = float(losses[row].sum().detach().cpu())
                total_nats += nats
                total_tokens += token_count
                total_original_events += event_count
                song_rows.append(
                    {
                        "sample_id": entry.sample_id,
                        "serialized_target_count": token_count,
                        "original_event_count": event_count,
                        "total_bits_per_original_event": nats
                        / max(1, event_count)
                        / math.log(2.0),
                        "nll_nats_per_serialized_token": nats / max(1, token_count),
                    }
                )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return (
        {
            "primary_metric_name": "micro_total_bits_per_original_score_event",
            "total_bits_per_original_event": total_nats
            / max(1, total_original_events)
            / math.log(2.0),
            "nll_nats_per_serialized_token": total_nats / max(1, total_tokens),
            "serialized_tokens_per_original_event": total_tokens
            / max(1, total_original_events),
            "evaluated_serialized_target_count": total_tokens,
            "evaluated_original_event_count": total_original_events,
            "evaluated_song_count": len(entries),
            "throughput_original_events_per_second": total_original_events / elapsed,
            "throughput_serialized_tokens_per_second": total_tokens / elapsed,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        song_rows,
    )


def gradient_path_audit(model: ExternalTokenizerCausalLM) -> dict[str, bool]:
    modules = {
        "token_embedding": model.token_embedding,
        "input_projection": model.input_projection,
        "backbone": model.blocks,
        "output_normalization": model.output_normalization,
        "output_head": model.output_head,
    }
    result: dict[str, bool] = {}
    for name, module in modules.items():
        result[name] = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    return result
