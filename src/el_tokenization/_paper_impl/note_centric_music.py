from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .discrete_music_primitive import (
    M4E_BEATS_PER_SEMANTIC_BAR,
    M4E_MAX_ABSOLUTE_BARS,
    M4E_MAX_DURATION_BARS,
    M4E_TICKS_PER_BEAT,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .dual_coordinate_melody import (
    RelativeCausalBlock,
    TIME_RELATION_BUCKETS,
    prepare_relative_causal_attention_mask,
    signed_log_bucket,
)
from .musical_time_melody import cyclic_phase, signed_unit_distance
from .sparse_melody_bpe import SparseMelodyStream


NOTE_CENTRIC_SCHEMA: Final = "usmm.m4l.note_centric_full_event.v1"

INPUT_PAD: Final = 0
INPUT_BOS: Final = 1
INPUT_NOTE: Final = 2
INPUT_REST: Final = 3

TARGET_NOTE: Final = 0
TARGET_REST: Final = 1
TARGET_EOS: Final = 2


@dataclass(frozen=True, slots=True)
class NoteToken:
    kind: int
    anchor: int
    pitch: int
    duration: int

    def __post_init__(self) -> None:
        if self.kind not in (INPUT_NOTE, INPUT_REST):
            raise ValueError("invalid note-centric token kind")
        if self.anchor < 0 or self.duration <= 0:
            raise ValueError("invalid note-centric physical support")
        if self.kind == INPUT_NOTE and not 0 <= self.pitch < 128:
            raise ValueError("note-centric pitch leaves MIDI support")
        if self.kind == INPUT_REST and self.pitch != 0:
            raise ValueError("REST must not carry pitch")


def linearize_note_tokens(stream: SparseMelodyStream) -> tuple[NoteToken, ...]:
    """Linearize physical notes by onset, then ascending MIDI pitch.

    Equal-onset notes remain separate causal tokens.  No group identifier or
    chord label is emitted; simultaneity is recoverable solely from the equal
    physical anchor and zero onset delta.
    """

    values: list[NoteToken] = []
    for anchor, atom in zip(stream.anchors, stream.atoms):
        if atom.is_rest:
            values.append(
                NoteToken(
                    kind=INPUT_REST,
                    anchor=int(anchor),
                    pitch=0,
                    duration=int(atom.rest_ticks),
                )
            )
            continue
        for note in sorted(atom.notes, key=lambda item: item.midi_pitch):
            values.append(
                NoteToken(
                    kind=INPUT_NOTE,
                    anchor=int(anchor),
                    pitch=int(note.midi_pitch),
                    duration=int(note.duration_ticks),
                )
            )
    if not values:
        raise ValueError("cannot linearize an empty music stream")
    if any(right.anchor < left.anchor for left, right in zip(values, values[1:])):
        raise AssertionError("note-centric anchors are not ordered")
    for left, right in zip(values, values[1:]):
        if left.anchor == right.anchor and left.kind == right.kind == INPUT_NOTE:
            if right.pitch < left.pitch:
                raise AssertionError("same-onset notes are not pitch ascending")
    return tuple(values)


@dataclass(slots=True)
class NoteCausalBatch:
    input_type: torch.Tensor
    input_pitch: torch.Tensor
    input_duration_bars: torch.Tensor
    input_duration_remainders: torch.Tensor
    input_anchor_ticks: torch.Tensor
    valid: torch.Tensor
    target_type: torch.Tensor
    target_time_bars: torch.Tensor
    target_time_remainders: torch.Tensor
    target_pitch: torch.Tensor
    target_duration_bars: torch.Tensor
    target_duration_remainders: torch.Tensor
    note_mask: torch.Tensor
    support_mask: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.valid.sum().item())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "NoteCausalBatch":
        return NoteCausalBatch(
            **{
                name: getattr(self, name).to(device, non_blocking=non_blocking)
                for name in self.__dataclass_fields__
            }
        )

    def pin_memory(self) -> "NoteCausalBatch":
        return NoteCausalBatch(
            **{
                name: getattr(self, name).pin_memory()
                for name in self.__dataclass_fields__
            }
        )


def tensorize_note_streams(
    streams: Sequence[SparseMelodyStream],
    *,
    device: torch.device | str = "cpu",
) -> NoteCausalBatch:
    if not streams:
        raise ValueError("cannot tensorize an empty note-centric batch")
    token_rows = [linearize_note_tokens(stream) for stream in streams]
    length = max(len(tokens) + 1 for tokens in token_rows)
    batch = len(token_rows)

    def zeros(dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((batch, length), dtype=dtype)

    input_type = zeros(torch.long)
    input_pitch = zeros(torch.long)
    input_duration_bars = zeros(torch.long)
    input_duration_remainders = zeros(torch.long)
    input_anchor_ticks = zeros(torch.long)
    valid = zeros(torch.bool)
    target_type = torch.full((batch, length), TARGET_EOS, dtype=torch.long)
    target_time_bars = zeros(torch.long)
    target_time_remainders = zeros(torch.long)
    target_pitch = zeros(torch.long)
    target_duration_bars = zeros(torch.long)
    target_duration_remainders = zeros(torch.long)
    note_mask = zeros(torch.bool)
    support_mask = zeros(torch.bool)

    for row, tokens in enumerate(token_rows):
        count = len(tokens)
        valid[row, : count + 1] = True
        input_type[row, 0] = INPUT_BOS
        input_anchor_ticks[row, 0] = 0
        previous_anchor = 0
        for index, token in enumerate(tokens):
            target_column = index
            input_column = index + 1
            target_type[row, target_column] = (
                TARGET_NOTE if token.kind == INPUT_NOTE else TARGET_REST
            )
            delta = int(token.anchor) - int(previous_anchor)
            if delta < 0:
                raise AssertionError("note-centric target time moves backward")
            time_bars, time_remainder = divmod(
                delta, M4E_TICKS_PER_SEMANTIC_BAR
            )
            if time_bars >= M4E_MAX_DURATION_BARS:
                raise ValueError("note-centric onset delta leaves model support")
            duration_bars, duration_remainder = divmod(
                int(token.duration), M4E_TICKS_PER_SEMANTIC_BAR
            )
            if duration_bars >= M4E_MAX_DURATION_BARS:
                raise ValueError("note-centric duration leaves model support")
            target_time_bars[row, target_column] = time_bars
            target_time_remainders[row, target_column] = time_remainder
            target_duration_bars[row, target_column] = duration_bars
            target_duration_remainders[row, target_column] = duration_remainder
            support_mask[row, target_column] = True
            if token.kind == INPUT_NOTE:
                target_pitch[row, target_column] = int(token.pitch)
                note_mask[row, target_column] = True

            input_type[row, input_column] = int(token.kind)
            input_pitch[row, input_column] = int(token.pitch)
            input_duration_bars[row, input_column] = duration_bars
            input_duration_remainders[row, input_column] = duration_remainder
            input_anchor_ticks[row, input_column] = int(token.anchor)
            previous_anchor = int(token.anchor)

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
    ).to(device)


class NoteFactorizedDecoder(nn.Module):
    """Teacher-forced Type -> Time -> Pitch -> Duration decoder."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.condition_step = nn.GRUCell(width, width)
        self.type_head = nn.Linear(width, 3)
        self.type_embedding = nn.Embedding(3, width)
        self.time_bar_head = nn.Linear(width, M4E_MAX_DURATION_BARS)
        self.time_bar_embedding = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.time_remainder_head = nn.Linear(width, M4E_TICKS_PER_SEMANTIC_BAR)
        self.time_remainder_embedding = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )
        self.pitch_head = nn.Linear(width, 128)
        self.pitch_embedding = nn.Embedding(128, width)
        self.duration_bar_head = nn.Linear(width, M4E_MAX_DURATION_BARS)
        self.duration_bar_embedding = nn.Embedding(
            M4E_MAX_DURATION_BARS, width
        )
        self.duration_remainder_head = nn.Linear(
            width, M4E_TICKS_PER_SEMANTIC_BAR
        )
        self.duration_remainder_embedding = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )

    def _advance(
        self, state: torch.Tensor, embedding: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        candidate = self.condition_step(
            embedding.reshape(-1, embedding.shape[-1]),
            state.reshape(-1, state.shape[-1]),
        ).reshape_as(state)
        return torch.where(mask.unsqueeze(-1), candidate, state)

    @staticmethod
    def _nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
        ).reshape_as(target)

    def losses(
        self, hidden: torch.Tensor, batch: NoteCausalBatch
    ) -> dict[str, torch.Tensor]:
        state = self.normalization(hidden)
        valid = batch.valid
        type_nll = self._nll(self.type_head(state), batch.target_type) * valid
        state = self._advance(
            state, self.type_embedding(batch.target_type), valid
        )

        time_bar_nll = self._nll(
            self.time_bar_head(self.normalization(state)),
            batch.target_time_bars,
        ) * batch.support_mask
        state = self._advance(
            state,
            self.time_bar_embedding(batch.target_time_bars),
            batch.support_mask,
        )
        time_remainder_nll = self._nll(
            self.time_remainder_head(self.normalization(state)),
            batch.target_time_remainders,
        ) * batch.support_mask
        state = self._advance(
            state,
            self.time_remainder_embedding(batch.target_time_remainders),
            batch.support_mask,
        )
        time_nll = time_bar_nll + time_remainder_nll

        pitch_nll = self._nll(
            self.pitch_head(self.normalization(state)), batch.target_pitch
        ) * batch.note_mask
        state = self._advance(
            state, self.pitch_embedding(batch.target_pitch), batch.note_mask
        )

        duration_bar_nll = self._nll(
            self.duration_bar_head(self.normalization(state)),
            batch.target_duration_bars,
        ) * batch.support_mask
        state = self._advance(
            state,
            self.duration_bar_embedding(batch.target_duration_bars),
            batch.support_mask,
        )
        duration_remainder_nll = self._nll(
            self.duration_remainder_head(self.normalization(state)),
            batch.target_duration_remainders,
        ) * batch.support_mask
        duration_nll = duration_bar_nll + duration_remainder_nll
        total = type_nll + time_nll + pitch_nll + duration_nll
        return {
            "total": total,
            "type": type_nll,
            "time": time_nll,
            "pitch": pitch_nll,
            "duration": duration_nll,
        }


class NoteCentricCausalModel(nn.Module):
    def __init__(
        self,
        *,
        width: int = 64,
        heads: int = 4,
        layers: int = 16,
        feedforward_width: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.input_type = nn.Embedding(4, width)
        self.pitch_circle = nn.Linear(2, width, bias=False)
        self.register = nn.Embedding(11, width)
        self.duration_bars = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.duration_remainders = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )
        self.content_projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.time_projection = nn.Sequential(
            nn.Linear(10, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.tessitura_projection = nn.Sequential(
            nn.Linear(5, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.input_fusion = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.time_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        self.meter_bias = nn.Embedding(M4E_TICKS_PER_SEMANTIC_BAR, heads)
        self.bar_distance_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        self.beat_distance_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        for embedding in (
            self.time_bias,
            self.meter_bias,
            self.bar_distance_bias,
            self.beat_distance_bias,
        ):
            nn.init.zeros_(embedding.weight)
        self.blocks = nn.ModuleList(
            RelativeCausalBlock(
                width=width,
                heads=heads,
                feedforward_width=feedforward_width,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.output_normalization = nn.LayerNorm(width)
        self.decoder = NoteFactorizedDecoder(width)

    @staticmethod
    def _phase(value: torch.Tensor, period: int, dtype: torch.dtype) -> torch.Tensor:
        angle = value.to(dtype) * (2.0 * math.pi / float(period))
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def _content(self, batch: NoteCausalBatch) -> torch.Tensor:
        dtype = self.pitch_circle.weight.dtype
        pitch = batch.input_pitch
        note = (
            self.pitch_circle(self._phase(torch.remainder(pitch, 12), 12, dtype))
            + self.register(torch.div(pitch, 12, rounding_mode="floor").clamp(0, 10))
            + self.duration_bars(batch.input_duration_bars)
            + self.duration_remainders(batch.input_duration_remainders)
        )
        sounding = batch.input_type == INPUT_NOTE
        rest = batch.input_type == INPUT_REST
        duration = (
            self.duration_bars(batch.input_duration_bars)
            + self.duration_remainders(batch.input_duration_remainders)
        )
        content = self.input_type(batch.input_type)
        content = content + torch.where(
            sounding.unsqueeze(-1), note, torch.zeros_like(note)
        )
        content = content + torch.where(
            rest.unsqueeze(-1), duration, torch.zeros_like(duration)
        )
        return self.content_projection(content).masked_fill(
            ~batch.valid.unsqueeze(-1), 0.0
        )

    def _time(self, batch: NoteCausalBatch) -> torch.Tensor:
        dtype = self.time_projection[0].weight.dtype
        anchor = batch.input_anchor_ticks.clamp_min(0)
        values = torch.cat(
            (
                self._phase(anchor, M4E_TICKS_PER_SEMANTIC_BAR, dtype),
                self._phase(
                    torch.div(anchor, M4E_TICKS_PER_BEAT, rounding_mode="floor"),
                    M4E_BEATS_PER_SEMANTIC_BAR,
                    dtype,
                ),
                self._phase(
                    torch.remainder(anchor, M4E_TICKS_PER_BEAT),
                    M4E_TICKS_PER_BEAT,
                    dtype,
                ),
                self._phase(anchor, 4 * M4E_TICKS_PER_SEMANTIC_BAR, dtype),
                self._phase(anchor, 16 * M4E_TICKS_PER_SEMANTIC_BAR, dtype),
            ),
            dim=-1,
        )
        return self.time_projection(values).masked_fill(
            ~batch.valid.unsqueeze(-1), 0.0
        )

    def _tessitura(self, batch: NoteCausalBatch) -> torch.Tensor:
        dtype = self.tessitura_projection[0].weight.dtype
        present = batch.input_type == INPUT_NOTE
        low = batch.input_pitch.masked_fill(~present, 127)
        high = batch.input_pitch.masked_fill(~present, 0)
        prefix_low = torch.cummin(low, dim=1).values
        prefix_high = torch.cummax(high, dim=1).values
        seen = present.to(torch.long).cumsum(dim=1) > 0
        prefix_low = torch.where(seen, prefix_low, torch.zeros_like(prefix_low))
        prefix_high = torch.where(seen, prefix_high, torch.zeros_like(prefix_high))
        values = torch.stack(
            (
                prefix_low.to(dtype) / 127.0,
                prefix_high.to(dtype) / 127.0,
                (prefix_high - prefix_low).to(dtype) / 127.0,
                (prefix_high + prefix_low).to(dtype) / 254.0,
                seen.to(dtype),
            ),
            dim=-1,
        )
        return self.tessitura_projection(values).masked_fill(
            ~batch.valid.unsqueeze(-1), 0.0
        )

    def relative_attention_bias(self, batch: NoteCausalBatch) -> torch.Tensor:
        delta = (
            batch.input_anchor_ticks[:, :, None]
            - batch.input_anchor_ticks[:, None, :]
        )
        onset = self.time_bias(signed_log_bucket(delta))
        meter = self.meter_bias(
            torch.remainder(delta, M4E_TICKS_PER_SEMANTIC_BAR)
        )
        bar = self.bar_distance_bias(
            signed_log_bucket(
                signed_unit_distance(delta, M4E_TICKS_PER_SEMANTIC_BAR),
                exact_magnitudes=8,
                maximum_distance=M4E_MAX_ABSOLUTE_BARS,
            )
        )
        beat = self.beat_distance_bias(
            signed_log_bucket(
                signed_unit_distance(delta, M4E_TICKS_PER_BEAT),
                exact_magnitudes=16,
                maximum_distance=(
                    M4E_MAX_ABSOLUTE_BARS * M4E_BEATS_PER_SEMANTIC_BAR
                ),
            )
        )
        return (onset + meter + bar + beat).permute(0, 3, 1, 2)

    def hidden_sequence(self, batch: NoteCausalBatch) -> torch.Tensor:
        values = self.input_fusion(
            torch.cat(
                (self._content(batch), self._time(batch), self._tessitura(batch)),
                dim=-1,
            )
        )
        bias = self.relative_attention_bias(batch)
        attention_mask = prepare_relative_causal_attention_mask(
            relative_bias=bias,
            valid=batch.valid,
            dtype=values.dtype,
        )
        for block in self.blocks:
            values = block(
                values,
                valid=batch.valid,
                attention_mask=attention_mask,
            )
        return self.output_normalization(values)

    def losses(self, batch: NoteCausalBatch) -> dict[str, torch.Tensor]:
        return self.decoder.losses(self.hidden_sequence(batch), batch)


def batch_metrics(
    losses: dict[str, torch.Tensor], batch: NoteCausalBatch
) -> dict[str, float]:
    denominator = max(1, batch.token_count)
    return {
        f"{name}_nats_per_token": float(value.sum().detach().cpu()) / denominator
        for name, value in losses.items()
    }
