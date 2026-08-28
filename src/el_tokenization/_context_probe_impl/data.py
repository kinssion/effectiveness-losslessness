from __future__ import annotations

import csv
import hashlib
from collections import OrderedDict, defaultdict
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset


TARGET_SLOT_START = 8 * 4
TARGET_SLOT_END = 12 * 4
MELODY_PROGRAM = 13


class POP909RelationDataset(Dataset[dict[str, Any]]):
    """Clean adjacency with deterministic same-song and cross-song mismatches."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        seed: int = 20260801,
        cache_songs: int = 32,
        max_windows: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.cache_songs = int(cache_songs)
        if self.cache_songs <= 0:
            raise ValueError("cache_songs must be positive")
        with (self.root / "windows.csv").open(newline="", encoding="utf-8") as stream:
            self.records = [
                row for row in csv.DictReader(stream) if row["split"] == split
            ]
        if max_windows is not None:
            self.records = self.records[:max_windows]
        if not self.records:
            raise ValueError(f"no windows found for split={split!r}")
        self.by_song: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            record["start_bar_0based"] = int(record["start_bar_0based"])
            self.by_song[record["song_id"]].append(index)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _load_song(self, record: dict[str, Any]) -> torch.Tensor:
        song_id = record["song_id"]
        cached = self._cache.pop(song_id, None)
        if cached is not None:
            self._cache[song_id] = cached
            return cached
        payload = torch.load(
            self.root / record["latent_file"], map_location="cpu", weights_only=False
        )
        latents = payload["normalized_latents"].float()
        if latents.ndim != 3 or tuple(latents.shape[1:]) != (4, 64):
            raise ValueError(f"invalid latent shape for {song_id}: {tuple(latents.shape)}")
        self._cache[song_id] = latents
        while len(self._cache) > self.cache_songs:
            self._cache.popitem(last=False)
        return latents

    def _window(self, record: dict[str, Any]) -> torch.Tensor:
        song = self._load_song(record)
        start = int(record["start_bar_0based"])
        window = song[start : start + 16]
        if tuple(window.shape) != (16, 4, 64):
            raise ValueError(
                f"window {record['window_id']} has shape {tuple(window.shape)}"
            )
        return window.reshape(64, 64).clone()

    def _rng(self, index: int) -> random.Random:
        digest = hashlib.sha256(
            f"{self.seed}:{self.split}:{index}".encode("utf-8")
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _same_song_negative_index(self, index: int, rng: random.Random) -> int:
        record = self.records[index]
        start = int(record["start_bar_0based"])
        candidates = [
            other
            for other in self.by_song[record["song_id"]]
            if other != index
            and abs(int(self.records[other]["start_bar_0based"]) - start) >= 8
        ]
        if not candidates:
            candidates = [
                other for other in self.by_song[record["song_id"]] if other != index
            ]
        if not candidates:
            raise ValueError(f"song {record['song_id']} has no negative target window")
        return candidates[rng.randrange(len(candidates))]

    def _cross_song_negative_index(self, index: int, rng: random.Random) -> int:
        source_song = self.records[index]["song_id"]
        start = rng.randrange(len(self.records))
        for offset in range(len(self.records)):
            candidate = (start + offset) % len(self.records)
            if self.records[candidate]["song_id"] != source_song:
                return candidate
        raise ValueError(f"split={self.split!r} contains only one song")

    @staticmethod
    def _replace_target(source: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        result = source.clone()
        result[TARGET_SLOT_START:TARGET_SLOT_END] = donor[
            TARGET_SLOT_START:TARGET_SLOT_END
        ]
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        positive = self._window(self.records[index])
        same_index = self._same_song_negative_index(index, rng)
        cross_index = self._cross_song_negative_index(index, rng)
        same_donor = self._window(self.records[same_index])
        cross_donor = self._window(self.records[cross_index])
        return {
            "positive": positive,
            "same_song_mismatch": self._replace_target(positive, same_donor),
            "cross_song_mismatch": self._replace_target(positive, cross_donor),
            "window_id": self.records[index]["window_id"],
            "song_id": self.records[index]["song_id"],
            "same_song_donor_id": self.records[same_index]["window_id"],
            "cross_song_donor_id": self.records[cross_index]["window_id"],
        }


class POP909MelodyFactorialDataset(Dataset[dict[str, Any]]):
    """Paired A/B/C/D counterfactuals with explicit POP909 role ownership.

    The source target harmony/accompaniment is identical in every condition.
    Only the program-13 phrase slots change between A and B, while only the
    left/right context changes between A and C (and between B and D).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        seed: int = 20260801,
        cache_songs: int = 32,
        minimum_window_distance_bars: int = 8,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.cache_songs = int(cache_songs)
        self.minimum_window_distance_bars = int(minimum_window_distance_bars)
        if self.cache_songs <= 0:
            raise ValueError("cache_songs must be positive")
        if self.minimum_window_distance_bars <= 0:
            raise ValueError("minimum_window_distance_bars must be positive")
        with (self.root / "windows.csv").open(newline="", encoding="utf-8") as stream:
            self.records = [
                row for row in csv.DictReader(stream) if row["split"] == split
            ]
        if not self.records:
            raise ValueError(f"no windows found for split={split!r}")
        self.by_song: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            record["start_bar_0based"] = int(record["start_bar_0based"])
            self.by_song[record["song_id"]].append(index)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._programs: dict[str, list[list[int | None]]] = {}
        self._load_program_metadata()
        self.examples = self._build_examples()
        if not self.examples:
            raise ValueError(
                f"no full-coverage program-{MELODY_PROGRAM} factorial examples "
                f"found for split={split!r}"
            )

    def _load_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        song_id = record["song_id"]
        cached = self._cache.pop(song_id, None)
        if cached is not None:
            self._cache[song_id] = cached
            return cached
        payload = torch.load(
            self.root / record["latent_file"], map_location="cpu", weights_only=False
        )
        latents = payload.get("normalized_latents")
        programs = payload.get("programs_by_bar")
        if not isinstance(latents, torch.Tensor):
            raise ValueError(f"missing normalized_latents for {song_id}")
        if latents.ndim != 3 or tuple(latents.shape[1:]) != (4, 64):
            raise ValueError(f"invalid latent shape for {song_id}: {tuple(latents.shape)}")
        if not isinstance(programs, list) or len(programs) != latents.shape[0]:
            raise ValueError(f"missing or invalid programs_by_bar for {song_id}")
        normalized_programs: list[list[int | None]] = []
        for bar_index, row in enumerate(programs):
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(
                    f"invalid programs_by_bar row for {song_id} bar {bar_index}: {row!r}"
                )
            normalized_programs.append(
                [None if value is None else int(value) for value in row]
            )
        result = {
            "normalized_latents": latents.float(),
            "programs_by_bar": normalized_programs,
        }
        self._cache[song_id] = result
        while len(self._cache) > self.cache_songs:
            self._cache.popitem(last=False)
        return result

    def _load_program_metadata(self) -> None:
        first_record_by_song = {
            record["song_id"]: record for record in self.records
        }
        for song_id, record in first_record_by_song.items():
            payload = self._load_payload(record)
            self._programs[song_id] = payload["programs_by_bar"]

    def _target_melody_slots(self, record: dict[str, Any]) -> tuple[int, ...] | None:
        programs = self._programs[record["song_id"]]
        start = int(record["start_bar_0based"])
        result: list[int] = []
        for relative_bar in range(8, 12):
            row = programs[start + relative_bar]
            matches = [index for index, program in enumerate(row) if program == MELODY_PROGRAM]
            if len(matches) != 1:
                return None
            result.append(matches[0])
        return tuple(result)

    def _rng(self, index: int) -> random.Random:
        digest = hashlib.sha256(
            f"melody-factorial:{self.seed}:{self.split}:{index}".encode("utf-8")
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _build_examples(self) -> list[tuple[int, int, int]]:
        melody_slots = {
            index: self._target_melody_slots(record)
            for index, record in enumerate(self.records)
        }
        examples: list[tuple[int, int, int]] = []
        for source_index, source in enumerate(self.records):
            if melody_slots[source_index] is None:
                continue
            source_start = int(source["start_bar_0based"])
            distant = [
                other
                for other in self.by_song[source["song_id"]]
                if other != source_index
                and abs(int(self.records[other]["start_bar_0based"]) - source_start)
                >= self.minimum_window_distance_bars
            ]
            melody_candidates = [
                other for other in distant if melody_slots[other] is not None
            ]
            if not melody_candidates:
                continue
            rng = self._rng(source_index)
            melody_index = melody_candidates[rng.randrange(len(melody_candidates))]
            context_candidates = [other for other in distant if other != melody_index]
            if not context_candidates:
                context_candidates = distant
            if not context_candidates:
                continue
            context_index = context_candidates[rng.randrange(len(context_candidates))]
            examples.append((source_index, melody_index, context_index))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _window(self, record: dict[str, Any]) -> tuple[torch.Tensor, list[list[int | None]]]:
        payload = self._load_payload(record)
        start = int(record["start_bar_0based"])
        latents = payload["normalized_latents"][start : start + 16]
        programs = payload["programs_by_bar"][start : start + 16]
        if tuple(latents.shape) != (16, 4, 64) or len(programs) != 16:
            raise ValueError(f"invalid 16-bar window {record['window_id']}")
        return latents.clone(), programs

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index, melody_index, context_index = self.examples[index]
        source_record = self.records[source_index]
        melody_record = self.records[melody_index]
        context_record = self.records[context_index]
        source, source_programs = self._window(source_record)
        melody_donor, melody_programs = self._window(melody_record)
        context_donor, _ = self._window(context_record)

        aligned = source.reshape(64, 64)
        misplaced_melody = aligned.clone()
        source_slots: list[int] = []
        donor_slots: list[int] = []
        for relative_bar in range(8, 12):
            source_matches = [
                slot for slot, program in enumerate(source_programs[relative_bar])
                if program == MELODY_PROGRAM
            ]
            donor_matches = [
                slot for slot, program in enumerate(melody_programs[relative_bar])
                if program == MELODY_PROGRAM
            ]
            if len(source_matches) != 1 or len(donor_matches) != 1:
                raise AssertionError("factorial example lost unique melody ownership")
            source_slot = source_matches[0]
            donor_slot = donor_matches[0]
            absolute_source_slot = relative_bar * 4 + source_slot
            misplaced_melody[absolute_source_slot] = melody_donor[
                relative_bar, donor_slot
            ]
            source_slots.append(absolute_source_slot)
            donor_slots.append(relative_bar * 4 + donor_slot)

        misplaced_context = aligned.clone()
        misplaced_context[:TARGET_SLOT_START] = context_donor.reshape(64, 64)[
            :TARGET_SLOT_START
        ]
        misplaced_context[TARGET_SLOT_END:] = context_donor.reshape(64, 64)[
            TARGET_SLOT_END:
        ]
        both_misplaced = misplaced_melody.clone()
        both_misplaced[:TARGET_SLOT_START] = misplaced_context[:TARGET_SLOT_START]
        both_misplaced[TARGET_SLOT_END:] = misplaced_context[TARGET_SLOT_END:]

        return {
            "a_aligned_melody_aligned_context": aligned,
            "b_misplaced_melody_aligned_context": misplaced_melody,
            "c_aligned_melody_misplaced_context": misplaced_context,
            "d_misplaced_melody_misplaced_context": both_misplaced,
            "source_melody_slots": torch.tensor(source_slots, dtype=torch.long),
            "donor_melody_slots": torch.tensor(donor_slots, dtype=torch.long),
            "window_id": source_record["window_id"],
            "melody_donor_id": melody_record["window_id"],
            "context_donor_id": context_record["window_id"],
            "song_id": source_record["song_id"],
        }
