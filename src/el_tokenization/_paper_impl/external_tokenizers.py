from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .discrete_music_primitive import (
    M4E_MAX_DURATION_BARS,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .iclr_cached_training import SharedNoteTensorCache
from .note_centric_music import INPUT_NOTE, INPUT_REST, NoteToken


EXTERNAL_TOKENIZER_SCHEMA = "m4l.external_tokenizer_positioning.v1"
SERIALIZED_CACHE_SCHEMA = "m4l.serialized_music_token_cache.v1"

LM_PAD = 0
LM_BOS = 1
LM_EOS = 2
LM_PAYLOAD_OFFSET = 3


class RemiLikeVocabulary:
    """Exact event-stream vocabulary over the frozen Note/REST carrier.

    The layout follows the recognizable REMI pattern
    ``BAR, POSITION, PITCH, DURATION`` while preserving the explicit REST
    facts used by the frozen M4L score-level domain.  BAR is a one-bar advance;
    POSITION is the exact tick inside the semantic 4/4 bar.  No music fact is
    quantized, discarded, or inferred by this serializer.
    """

    BAR = 0
    REST = 1
    POSITION_START = 2
    POSITION_COUNT = M4E_TICKS_PER_SEMANTIC_BAR
    PITCH_START = POSITION_START + POSITION_COUNT
    PITCH_COUNT = 128
    DURATION_BAR_START = PITCH_START + PITCH_COUNT
    DURATION_BAR_COUNT = M4E_MAX_DURATION_BARS
    DURATION_REMAINDER_START = DURATION_BAR_START + DURATION_BAR_COUNT
    DURATION_REMAINDER_COUNT = M4E_TICKS_PER_SEMANTIC_BAR
    SIZE = DURATION_REMAINDER_START + DURATION_REMAINDER_COUNT

    @classmethod
    def position(cls, tick: int) -> int:
        if not 0 <= int(tick) < cls.POSITION_COUNT:
            raise ValueError("REMI position leaves exact semantic-bar support")
        return cls.POSITION_START + int(tick)

    @classmethod
    def pitch(cls, value: int) -> int:
        if not 0 <= int(value) < cls.PITCH_COUNT:
            raise ValueError("REMI pitch leaves MIDI support")
        return cls.PITCH_START + int(value)

    @classmethod
    def duration_bar(cls, value: int) -> int:
        if not 0 <= int(value) < cls.DURATION_BAR_COUNT:
            raise ValueError("REMI duration-bar component leaves support")
        return cls.DURATION_BAR_START + int(value)

    @classmethod
    def duration_remainder(cls, value: int) -> int:
        if not 0 <= int(value) < cls.DURATION_REMAINDER_COUNT:
            raise ValueError("REMI duration remainder leaves support")
        return cls.DURATION_REMAINDER_START + int(value)

    @classmethod
    def is_position(cls, token: int) -> bool:
        return cls.POSITION_START <= int(token) < cls.PITCH_START

    @classmethod
    def is_pitch(cls, token: int) -> bool:
        return cls.PITCH_START <= int(token) < cls.DURATION_BAR_START

    @classmethod
    def is_duration_bar(cls, token: int) -> bool:
        return cls.DURATION_BAR_START <= int(token) < cls.DURATION_REMAINDER_START

    @classmethod
    def is_duration_remainder(cls, token: int) -> bool:
        return cls.DURATION_REMAINDER_START <= int(token) < cls.SIZE


def _duration_components(row: np.void) -> tuple[int, int, int]:
    bars = int(row["duration_bars"])
    remainder = int(row["duration_remainder"])
    duration = bars * M4E_TICKS_PER_SEMANTIC_BAR + remainder
    if duration <= 0:
        raise ValueError("serialized physical event has non-positive duration")
    return bars, remainder, duration


def encode_remi_like_row(row: np.ndarray) -> tuple[int, ...]:
    """Serialize one cached exact Note/REST stream without loss."""

    if len(row) == 0:
        raise ValueError("cannot serialize an empty exact-event row")
    output: list[int] = []
    current_bar = -1
    current_anchor: int | None = None
    for event in row:
        anchor = int(event["anchor"])
        if current_anchor is not None and anchor < current_anchor:
            raise ValueError("event stream moves backward")
        if anchor != current_anchor:
            bar, position = divmod(anchor, M4E_TICKS_PER_SEMANTIC_BAR)
            if bar < current_bar:
                raise ValueError("REMI bar index moves backward")
            output.extend([RemiLikeVocabulary.BAR] * (bar - current_bar))
            current_bar = bar
            output.append(RemiLikeVocabulary.position(position))
            current_anchor = anchor

        bars, remainder, _duration = _duration_components(event)
        kind = int(event["kind"])
        if kind == INPUT_NOTE:
            output.append(RemiLikeVocabulary.pitch(int(event["pitch"])))
        elif kind == INPUT_REST:
            if int(event["pitch"]) != 0:
                raise ValueError("REST carries pitch in shared exact-event cache")
            output.append(RemiLikeVocabulary.REST)
        else:
            raise ValueError(f"unsupported exact-event kind: {kind}")
        output.append(RemiLikeVocabulary.duration_bar(bars))
        output.append(RemiLikeVocabulary.duration_remainder(remainder))
    return tuple(output)


def decode_remi_like_payload(payload: Sequence[int]) -> tuple[NoteToken, ...]:
    """Invert :func:`encode_remi_like_row` exactly."""

    result: list[NoteToken] = []
    bar = -1
    position: int | None = None
    index = 0
    while index < len(payload):
        token = int(payload[index])
        if token == RemiLikeVocabulary.BAR:
            bar += 1
            position = None
            index += 1
            continue
        if RemiLikeVocabulary.is_position(token):
            if bar < 0:
                raise ValueError("POSITION occurs before the first BAR")
            position = token - RemiLikeVocabulary.POSITION_START
            index += 1
            continue
        if position is None or bar < 0:
            raise ValueError("musical event occurs without an exact onset address")
        if token == RemiLikeVocabulary.REST:
            kind = INPUT_REST
            pitch = 0
        elif RemiLikeVocabulary.is_pitch(token):
            kind = INPUT_NOTE
            pitch = token - RemiLikeVocabulary.PITCH_START
        else:
            raise ValueError(f"unexpected REMI payload token: {token}")
        if index + 2 >= len(payload):
            raise ValueError("truncated REMI duration")
        bar_token = int(payload[index + 1])
        remainder_token = int(payload[index + 2])
        if not RemiLikeVocabulary.is_duration_bar(bar_token):
            raise ValueError("event is not followed by a duration-bar token")
        if not RemiLikeVocabulary.is_duration_remainder(remainder_token):
            raise ValueError("event is not followed by a duration-remainder token")
        duration = (
            (bar_token - RemiLikeVocabulary.DURATION_BAR_START)
            * M4E_TICKS_PER_SEMANTIC_BAR
            + remainder_token
            - RemiLikeVocabulary.DURATION_REMAINDER_START
        )
        result.append(
            NoteToken(
                kind=kind,
                anchor=bar * M4E_TICKS_PER_SEMANTIC_BAR + position,
                pitch=pitch,
                duration=duration,
            )
        )
        index += 3
    if not result:
        raise ValueError("decoded REMI stream has no physical events")
    return tuple(result)


def cached_row_as_note_tokens(row: np.ndarray) -> tuple[NoteToken, ...]:
    result: list[NoteToken] = []
    for event in row:
        _bars, _remainder, duration = _duration_components(event)
        result.append(
            NoteToken(
                kind=int(event["kind"]),
                anchor=int(event["anchor"]),
                pitch=int(event["pitch"]),
                duration=duration,
            )
        )
    return tuple(result)


def payload_to_private_unicode(payload: Sequence[int]) -> str:
    if any(not 0 <= int(value) < RemiLikeVocabulary.SIZE for value in payload):
        raise ValueError("base symbol leaves the frozen REMI-like vocabulary")
    return "".join(chr(0xE000 + int(value)) for value in payload)


def private_unicode_to_payload(value: str) -> tuple[int, ...]:
    result = tuple(ord(character) - 0xE000 for character in value)
    if any(not 0 <= token < RemiLikeVocabulary.SIZE for token in result):
        raise ValueError("BPE piece expansion leaves the base vocabulary")
    return result


@dataclass(frozen=True, slots=True)
class LosslessBPEModel:
    tokenizer_json: Path
    base_vocabulary_size: int
    vocabulary_size: int
    fit_split: str
    fit_sample_count: int

    @classmethod
    def load(cls, path: Path) -> "LosslessBPEModel":
        receipt_path = path.with_suffix(".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != EXTERNAL_TOKENIZER_SCHEMA + ".bpe":
            raise ValueError("unsupported lossless BPE receipt")
        if receipt["fit_split"] != "train":
            raise ValueError("BPE merge model was not fit exclusively on train")
        if _sha256_file(path) != receipt["tokenizer_sha256"]:
            raise ValueError("lossless BPE tokenizer hash mismatch")
        return cls(
            tokenizer_json=path,
            base_vocabulary_size=int(receipt["base_vocabulary_size"]),
            vocabulary_size=int(receipt["vocabulary_size"]),
            fit_split=str(receipt["fit_split"]),
            fit_sample_count=int(receipt["fit_sample_count"]),
        )

    def _tokenizer(self):
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "lossless BPE requires the pinned `tokenizers` runtime"
            ) from error
        return Tokenizer.from_file(str(self.tokenizer_json))

    def encode(self, payload: Sequence[int]) -> tuple[int, ...]:
        tokenizer = self._tokenizer()
        encoded = tokenizer.encode(
            payload_to_private_unicode(payload), add_special_tokens=False
        )
        expanded = tuple(
            token
            for piece in encoded.tokens
            for token in private_unicode_to_payload(piece)
        )
        if expanded != tuple(int(value) for value in payload):
            raise AssertionError("BPE encoding is not exactly reversible")
        return tuple(int(value) for value in encoded.ids)

    def decode_ids(self, ids: Sequence[int]) -> tuple[int, ...]:
        tokenizer = self._tokenizer()
        pieces = [tokenizer.id_to_token(int(value)) for value in ids]
        if any(piece is None for piece in pieces):
            raise ValueError("BPE token ID is absent from the frozen vocabulary")
        return tuple(
            token
            for piece in pieces
            for token in private_unicode_to_payload(str(piece))
        )


def fit_lossless_bpe(
    *,
    train_payloads: Iterable[Sequence[int]],
    train_sample_count: int,
    target_vocabulary_size: int,
    minimum_frequency: int,
    output: Path,
) -> dict[str, Any]:
    """Fit ordinary frequency BPE on train songs only.

    Each base REMI symbol is mapped bijectively to one private-use Unicode
    scalar.  A song is one BPE word, so merges may span adjacent event fields
    but can never cross a song boundary.  Expansion of learned pieces recovers
    the exact base stream.
    """

    try:
        from tokenizers import Tokenizer, models, trainers
    except ImportError as error:
        raise RuntimeError(
            "install the pinned `tokenizers` runtime before fitting BPE"
        ) from error
    if int(target_vocabulary_size) <= RemiLikeVocabulary.SIZE:
        raise ValueError("BPE target vocabulary must add at least one merge")
    initial_alphabet = [chr(0xE000 + value) for value in range(RemiLikeVocabulary.SIZE)]
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    trainer = trainers.BpeTrainer(
        vocab_size=int(target_vocabulary_size),
        min_frequency=int(minimum_frequency),
        show_progress=True,
        initial_alphabet=initial_alphabet,
        limit_alphabet=len(initial_alphabet),
        special_tokens=[],
    )
    tokenizer.train_from_iterator(
        (payload_to_private_unicode(payload) for payload in train_payloads),
        trainer=trainer,
        length=int(train_sample_count),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    tokenizer.save(str(temporary))
    os.replace(temporary, output)
    vocabulary_size = int(tokenizer.get_vocab_size(with_added_tokens=False))
    receipt = {
        "schema_version": EXTERNAL_TOKENIZER_SCHEMA + ".bpe",
        "status": "FROZEN_TRAIN_ONLY_LOSSLESS_BPE",
        "algorithm": "frequency_BPE",
        "fit_split": "train",
        "fit_sample_count": int(train_sample_count),
        "base_vocabulary_size": RemiLikeVocabulary.SIZE,
        "target_vocabulary_size": int(target_vocabulary_size),
        "vocabulary_size": vocabulary_size,
        "learned_merge_count": vocabulary_size - RemiLikeVocabulary.SIZE,
        "minimum_frequency": int(minimum_frequency),
        "song_boundary_crossing": False,
        "validation_used_for_merge_learning": False,
        "test_used_for_merge_learning": False,
        "tokenizer_sha256": _sha256_file(output),
    }
    receipt_path = output.with_suffix(".receipt.json")
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_serialized_cache(
    output: Path,
    *,
    rows: Iterable[tuple[str, str, Sequence[int], int]],
    manifest_sha256: str,
    tokenizer_name: str,
    payload_vocabulary_size: int,
    tokenizer_artifact_sha256: str | None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"serialized token cache already exists: {output}")
    temporary = output.with_name(f"{output.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    token_path = temporary / "tokens.bin"
    sample_path = temporary / "samples.jsonl"
    offsets = [0]
    original_events: list[int] = []
    split_counts: dict[str, int] = {}
    split_original_events: dict[str, int] = {}
    maximum_payload_tokens = 0
    with token_path.open("wb") as token_handle, sample_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as sample_handle:
        for split, sample_id, payload, original_event_count in rows:
            values = np.asarray(tuple(int(value) for value in payload), dtype="<u4")
            if len(values) == 0:
                raise ValueError(f"serialized payload is empty: {sample_id}")
            if int(values.max()) >= int(payload_vocabulary_size):
                raise ValueError(f"serialized token leaves vocabulary: {sample_id}")
            values.tofile(token_handle)
            offsets.append(offsets[-1] + len(values))
            original_events.append(int(original_event_count))
            split_counts[split] = split_counts.get(split, 0) + 1
            split_original_events[split] = (
                split_original_events.get(split, 0) + int(original_event_count)
            )
            maximum_payload_tokens = max(maximum_payload_tokens, len(values))
            sample_handle.write(
                json.dumps(
                    {
                        "cache_row": len(offsets) - 2,
                        "sample_id": sample_id,
                        "split": split,
                        "payload_token_count": len(values),
                        "original_event_count": int(original_event_count),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    offsets_path = temporary / "offsets.npy"
    original_events_path = temporary / "original_event_counts.npy"
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    np.save(
        original_events_path,
        np.asarray(original_events, dtype=np.int32),
        allow_pickle=False,
    )
    metadata = {
        "schema_version": SERIALIZED_CACHE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "tokenizer_name": tokenizer_name,
        "payload_vocabulary_size": int(payload_vocabulary_size),
        "lm_vocabulary_size": int(payload_vocabulary_size) + LM_PAYLOAD_OFFSET,
        "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
        "sample_count": len(original_events),
        "payload_token_count": offsets[-1],
        "original_event_count": sum(original_events),
        "tokens_per_original_event": offsets[-1] / max(1, sum(original_events)),
        "maximum_payload_tokens": maximum_payload_tokens,
        "split_counts": split_counts,
        "split_original_event_counts": split_original_events,
        "test_rows": int(split_counts.get("test", 0)),
        "files": {
            "tokens.bin": _sha256_file(token_path),
            "offsets.npy": _sha256_file(offsets_path),
            "original_event_counts.npy": _sha256_file(original_events_path),
            "samples.jsonl": _sha256_file(sample_path),
        },
    }
    _write_json_atomic(temporary / "cache_manifest.json", metadata)
    os.replace(temporary, output)
    return metadata


def build_remi_cache(
    shared_cache: SharedNoteTensorCache,
    output: Path,
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    ordered = sorted(shared_cache.sample_rows.items(), key=lambda item: item[1])

    def rows() -> Iterator[tuple[str, str, Sequence[int], int]]:
        for sample_id, _cache_row in ordered:
            source = shared_cache.row(sample_id)
            payload = encode_remi_like_row(source)
            if decode_remi_like_payload(payload) != cached_row_as_note_tokens(source):
                raise AssertionError(f"REMI-like roundtrip failed: {sample_id}")
            yield (
                shared_cache.sample_splits[sample_id],
                sample_id,
                payload,
                len(source) + 1,
            )

    return _write_serialized_cache(
        output,
        rows=rows(),
        manifest_sha256=manifest_sha256,
        tokenizer_name="J_LOSSLESS_REMI_LIKE",
        payload_vocabulary_size=RemiLikeVocabulary.SIZE,
        tokenizer_artifact_sha256=None,
    )


class SerializedTokenCache:
    def __init__(
        self,
        root: Path,
        *,
        expected_manifest_sha256: str,
        verify_files: bool = True,
        allow_test_rows: bool = False,
    ) -> None:
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "cache_manifest.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("schema_version") != SERIALIZED_CACHE_SCHEMA:
            raise ValueError("unsupported serialized token cache")
        if self.metadata.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("serialized cache manifest hash mismatch")
        if int(self.metadata.get("test_rows", -1)) != 0 and not allow_test_rows:
            raise AssertionError("serialized formal-training cache contains test rows")
        if verify_files:
            for name, expected in self.metadata["files"].items():
                if _sha256_file(self.root / name) != expected:
                    raise ValueError(f"serialized cache file hash mismatch: {name}")
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.original_event_counts = np.load(
            self.root / "original_event_counts.npy", mmap_mode="r"
        )
        self.tokens = np.memmap(
            self.root / "tokens.bin",
            dtype="<u4",
            mode="r",
            shape=(int(self.metadata["payload_token_count"]),),
        )
        self.sample_rows: dict[str, int] = {}
        self.sample_splits: dict[str, str] = {}
        with (self.root / "samples.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                self.sample_rows[sample_id] = int(row["cache_row"])
                self.sample_splits[sample_id] = str(row["split"])
        if len(self.sample_rows) != int(self.metadata["sample_count"]):
            raise ValueError("serialized cache sample index mismatch")

    def row(self, sample_id: str) -> np.ndarray:
        index = self.sample_rows[sample_id]
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.tokens[start:end]

    def original_event_count(self, sample_id: str) -> int:
        return int(self.original_event_counts[self.sample_rows[sample_id]])

    def split_payloads(self, split: str) -> Iterator[tuple[str, np.ndarray]]:
        ordered = sorted(self.sample_rows.items(), key=lambda item: item[1])
        for sample_id, _index in ordered:
            if self.sample_splits[sample_id] == split:
                yield sample_id, self.row(sample_id)


def build_bpe_cache(
    remi_cache: SerializedTokenCache,
    bpe: LosslessBPEModel,
    output: Path,
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    if bpe.fit_split != "train":
        raise AssertionError("BPE model crossed the train-only firewall")
    ordered = sorted(remi_cache.sample_rows.items(), key=lambda item: item[1])
    tokenizer = bpe._tokenizer()

    def rows() -> Iterator[tuple[str, str, Sequence[int], int]]:
        for sample_id, _index in ordered:
            base = tuple(int(value) for value in remi_cache.row(sample_id))
            encoded = tokenizer.encode(
                payload_to_private_unicode(base), add_special_tokens=False
            )
            recovered = tuple(
                token
                for piece in encoded.tokens
                for token in private_unicode_to_payload(piece)
            )
            if recovered != base:
                raise AssertionError(f"lossless BPE roundtrip failed: {sample_id}")
            yield (
                remi_cache.sample_splits[sample_id],
                sample_id,
                tuple(int(value) for value in encoded.ids),
                remi_cache.original_event_count(sample_id),
            )

    return _write_serialized_cache(
        output,
        rows=rows(),
        manifest_sha256=manifest_sha256,
        tokenizer_name="K_TRAIN_ONLY_LOSSLESS_BPE",
        payload_vocabulary_size=bpe.vocabulary_size,
        tokenizer_artifact_sha256=_sha256_file(bpe.tokenizer_json),
    )
