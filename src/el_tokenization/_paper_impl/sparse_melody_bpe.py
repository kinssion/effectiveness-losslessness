from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence

from .discrete_music_primitive import CanonicalPrimitiveSong


NOTE_GROUP = "NOTE_GROUP"
REST = "REST"
RELATIVE_START = "RELATIVE_START"
RELATIVE_CONTINUATION = "RELATIVE_CONTINUATION"
BOS = ("<BOS>",)
ESC = ("<ESC>",)

AtomKey = tuple[object, ...]
Symbol = tuple[AtomKey, ...]


def stable_digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class SparseNote:
    """One note value inside an onset-local NoteGroup.

    Pitch is explicitly factorized as pitch class plus scientific-register
    octave.  This is lossless and does not impose a key, scale, or chord label.
    """

    pitch_class: int
    octave: int
    duration_ticks: int

    def __post_init__(self) -> None:
        if not 0 <= self.pitch_class < 12:
            raise ValueError("pitch class leaves 12-class support")
        if not 0 <= self.midi_pitch <= 127:
            raise ValueError("factorized pitch leaves MIDI support")
        if self.duration_ticks <= 0:
            raise ValueError("note duration must be positive")

    @property
    def midi_pitch(self) -> int:
        return (self.octave + 1) * 12 + self.pitch_class

    @classmethod
    def from_midi(cls, pitch: int, duration_ticks: int) -> "SparseNote":
        if not 0 <= int(pitch) <= 127:
            raise ValueError("pitch leaves MIDI support")
        return cls(
            pitch_class=int(pitch) % 12,
            octave=int(pitch) // 12 - 1,
            duration_ticks=int(duration_ticks),
        )

    @property
    def key(self) -> tuple[int, int, int]:
        return self.pitch_class, self.octave, self.duration_ticks


@dataclass(frozen=True, slots=True)
class MelodyAtom:
    """A strict sequence atom: an onset-local NoteGroup or a true REST.

    Onset is occurrence/position information and is deliberately not stored in
    this content identity.  REST denotes a maximal interval during which no
    selected-stream note sounds; an onset-free point under sustain is not REST.
    """

    kind: str
    notes: tuple[SparseNote, ...] = ()
    rest_ticks: int = 0

    def __post_init__(self) -> None:
        if self.kind == NOTE_GROUP:
            if not self.notes or self.rest_ticks:
                raise ValueError("NoteGroup requires notes and no REST duration")
            expected = tuple(
                sorted(
                    self.notes,
                    key=lambda note: (note.midi_pitch, note.duration_ticks),
                )
            )
            if expected != self.notes:
                raise ValueError("same-onset notes must be pitch ascending")
        elif self.kind == REST:
            if self.notes or self.rest_ticks <= 0:
                raise ValueError("REST requires one positive silent duration")
        else:
            raise ValueError(f"unknown melody atom kind: {self.kind}")

    @classmethod
    def note_group(cls, notes: Iterable[SparseNote]) -> "MelodyAtom":
        ordered = tuple(
            sorted(notes, key=lambda note: (note.midi_pitch, note.duration_ticks))
        )
        return cls(kind=NOTE_GROUP, notes=ordered)

    @classmethod
    def rest(cls, duration_ticks: int) -> "MelodyAtom":
        return cls(kind=REST, rest_ticks=int(duration_ticks))

    @property
    def is_rest(self) -> bool:
        return self.kind == REST

    @property
    def key(self) -> AtomKey:
        if self.is_rest:
            return (REST, self.rest_ticks)
        return (NOTE_GROUP, tuple(note.key for note in self.notes))

    @classmethod
    def from_key(cls, key: AtomKey) -> "MelodyAtom":
        if key[0] == REST:
            return cls.rest(int(key[1]))
        if key[0] != NOTE_GROUP:
            raise ValueError("unknown serialized melody atom")
        rows = key[1]
        if not isinstance(rows, tuple):
            raise ValueError("malformed NoteGroup rows")
        return cls.note_group(
            SparseNote(int(pc), int(octave), int(duration))
            for pc, octave, duration in rows
        )


@dataclass(frozen=True, slots=True)
class SparseMelodyStream:
    song_id: str
    source_stream: int
    atoms: tuple[MelodyAtom, ...]
    anchors: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.atoms) != len(self.anchors):
            raise ValueError("atom and anchor counts differ")
        if any(right < left for left, right in zip(self.anchors, self.anchors[1:])):
            raise ValueError("melody occurrence anchors are not ordered")

    @property
    def stream_id(self) -> str:
        return f"{self.song_id}:{self.source_stream}"

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return tuple((atom.key,) for atom in self.atoms)

    @property
    def sounding_groups(self) -> int:
        return sum(not atom.is_rest for atom in self.atoms)

    @property
    def true_rests(self) -> int:
        return sum(atom.is_rest for atom in self.atoms)


@dataclass(frozen=True, slots=True)
class SparseCarrierReceipt:
    song_id: str
    source_stream: int
    raw_notes: int
    reconstructed_notes: int
    sounding_groups: int
    true_rests: int
    true_rest_ticks: int
    overlap_boundaries: int
    touching_boundaries: int
    exact_note_roundtrip: bool


@dataclass(frozen=True, slots=True)
class RelationalFamilyReceipt:
    raw_units: int
    sounding_groups: int
    rest_boundaries: int
    segment_starts: int
    continuations: int
    distinct_family_atoms: int


def build_relational_family_symbols(
    stream: SparseMelodyStream,
) -> tuple[tuple[Symbol, ...], RelationalFamilyReceipt]:
    """Project exact occurrences into a minimal transposition/time-scale family.

    This is an identity view only.  The source ``SparseMelodyStream`` retains
    every absolute pitch, onset, and duration needed for exact grounding.
    """

    symbols: list[Symbol] = []
    previous_low: int | None = None
    previous_onset: int | None = None
    segment_starts = 0
    continuations = 0
    for anchor, atom in zip(stream.anchors, stream.atoms):
        if atom.is_rest:
            symbols.append(((REST, 0),))
            previous_low = None
            previous_onset = None
            continue
        pitches = tuple(note.midi_pitch for note in atom.notes)
        durations = tuple(note.duration_ticks for note in atom.notes)
        low = min(pitches)
        pitch_shape = tuple(pitch - low for pitch in pitches)
        if previous_low is None or previous_onset is None:
            scale = 0
            for value in durations:
                scale = math.gcd(scale, int(value))
            scale = max(scale, 1)
            duration_shape = tuple(value // scale for value in durations)
            family: AtomKey = (
                RELATIVE_START,
                pitch_shape,
                duration_shape,
            )
            segment_starts += 1
        else:
            onset_gap = int(anchor) - previous_onset
            if onset_gap <= 0:
                raise AssertionError("ordered NoteGroups require positive onset gap")
            scale = onset_gap
            for value in durations:
                scale = math.gcd(scale, int(value))
            scale = max(scale, 1)
            family = (
                RELATIVE_CONTINUATION,
                low - previous_low,
                pitch_shape,
                onset_gap // scale,
                tuple(value // scale for value in durations),
            )
            continuations += 1
        symbols.append((family,))
        previous_low = low
        previous_onset = int(anchor)
    receipt = RelationalFamilyReceipt(
        raw_units=len(symbols),
        sounding_groups=stream.sounding_groups,
        rest_boundaries=stream.true_rests,
        segment_starts=segment_starts,
        continuations=continuations,
        distinct_family_atoms=len({symbol[0] for symbol in symbols}),
    )
    return tuple(symbols), receipt


def build_sparse_melody_stream(
    song: CanonicalPrimitiveSong,
    *,
    source_stream: int = 0,
) -> tuple[SparseMelodyStream, SparseCarrierReceipt]:
    """Build onset-local NoteGroups plus maximal true-silence REST atoms."""

    song.assert_contract()
    selected = song.track_entities.astype(int) == int(source_stream)
    grouped: dict[int, list[SparseNote]] = defaultdict(list)
    source_facts: Counter[tuple[int, int, int]] = Counter()
    for onset, pitch, duration in zip(
        song.onsets[selected].astype(int).tolist(),
        song.pitches[selected].astype(int).tolist(),
        song.durations[selected].astype(int).tolist(),
    ):
        grouped[int(onset)].append(SparseNote.from_midi(pitch, duration))
        source_facts[(int(onset), int(pitch), int(duration))] += 1

    atoms: list[MelodyAtom] = []
    anchors: list[int] = []
    active_until: int | None = None
    overlap_boundaries = 0
    touching_boundaries = 0
    for onset in sorted(grouped):
        notes = grouped[onset]
        if active_until is not None:
            if onset > active_until:
                atoms.append(MelodyAtom.rest(onset - active_until))
                anchors.append(active_until)
            elif onset < active_until:
                overlap_boundaries += 1
            else:
                touching_boundaries += 1
        atoms.append(MelodyAtom.note_group(notes))
        anchors.append(onset)
        group_end = max(onset + note.duration_ticks for note in notes)
        active_until = group_end if active_until is None else max(active_until, group_end)

    stream = SparseMelodyStream(
        song_id=song.song_id,
        source_stream=int(source_stream),
        atoms=tuple(atoms),
        anchors=tuple(anchors),
    )
    reconstructed: Counter[tuple[int, int, int]] = Counter()
    for anchor, atom in zip(stream.anchors, stream.atoms):
        if atom.is_rest:
            continue
        for note in atom.notes:
            reconstructed[(anchor, note.midi_pitch, note.duration_ticks)] += 1
    receipt = SparseCarrierReceipt(
        song_id=song.song_id,
        source_stream=int(source_stream),
        raw_notes=sum(source_facts.values()),
        reconstructed_notes=sum(reconstructed.values()),
        sounding_groups=stream.sounding_groups,
        true_rests=stream.true_rests,
        true_rest_ticks=sum(
            atom.rest_ticks for atom in stream.atoms if atom.is_rest
        ),
        overlap_boundaries=overlap_boundaries,
        touching_boundaries=touching_boundaries,
        exact_note_roundtrip=reconstructed == source_facts,
    )
    return stream, receipt


@dataclass(frozen=True, slots=True)
class MergeRule:
    left: Symbol
    right: Symbol
    merged: Symbol
    fit_count: int
    fit_song_support: int

    @property
    def support_units(self) -> int:
        return len(self.merged)


def decode_symbols(symbols: Sequence[Symbol]) -> tuple[AtomKey, ...]:
    return tuple(atom for symbol in symbols for atom in symbol)


def apply_merge_rule(
    symbols: Sequence[Symbol], rule: MergeRule
) -> tuple[Symbol, ...]:
    output: list[Symbol] = []
    index = 0
    while index < len(symbols):
        if (
            index + 1 < len(symbols)
            and symbols[index] == rule.left
            and symbols[index + 1] == rule.right
        ):
            output.append(rule.merged)
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return tuple(output)


def apply_merge_rules(
    symbols: Sequence[Symbol], rules: Sequence[MergeRule]
) -> tuple[Symbol, ...]:
    result = tuple(symbols)
    for rule in rules:
        result = apply_merge_rule(result, rule)
    return result


def frequency_pair_candidates(
    sequences: Mapping[str, Sequence[Symbol]],
    *,
    minimum_count: int,
    minimum_song_support: int,
    maximum_support_units: int,
    forbid_rest: bool = True,
    limit: int | None = None,
) -> tuple[tuple[tuple[Symbol, Symbol], int, int], ...]:
    counts: Counter[tuple[Symbol, Symbol]] = Counter()
    songs: dict[tuple[Symbol, Symbol], set[str]] = defaultdict(set)
    known = {symbol for sequence in sequences.values() for symbol in sequence}
    for stream_id, sequence in sequences.items():
        song_id = stream_id.split(":", 1)[0]
        seen: set[tuple[Symbol, Symbol]] = set()
        for pair in zip(sequence, sequence[1:]):
            counts[pair] += 1
            seen.add(pair)
        for pair in seen:
            songs[pair].add(song_id)
    eligible = [
        (pair, count, len(songs[pair]))
        for pair, count in counts.items()
        if count >= int(minimum_count)
        and len(songs[pair]) >= int(minimum_song_support)
        and len(pair[0]) + len(pair[1]) <= int(maximum_support_units)
        and pair[0] + pair[1] not in known
        and (
            not forbid_rest
            or (rest_units(pair[0]) == 0 and rest_units(pair[1]) == 0)
        )
    ]
    eligible.sort(key=lambda row: (-row[1], -row[2], stable_digest(row[0])))
    if limit is not None:
        eligible = eligible[: int(limit)]
    return tuple(eligible)


def fit_frequency_bpe(
    sequences: Mapping[str, Sequence[Symbol]],
    *,
    merge_budget: int,
    minimum_count: int,
    minimum_song_support: int,
    maximum_support_units: int,
    forbid_rest: bool = True,
) -> tuple[MergeRule, ...]:
    """Greedy adjacent-pair BPE; frequency is the only merge criterion.

    By default REST is punctuation: it remains a standalone raw symbol, is
    never part of a merge rule, and therefore forms an uncrossable boundary.
    """

    current = {key: tuple(value) for key, value in sequences.items()}
    known = {symbol for sequence in current.values() for symbol in sequence}
    rules: list[MergeRule] = []
    for _ in range(int(merge_budget)):
        eligible = frequency_pair_candidates(
            current,
            minimum_count=minimum_count,
            minimum_song_support=minimum_song_support,
            maximum_support_units=maximum_support_units,
            forbid_rest=forbid_rest,
        )
        if not eligible:
            break
        pair, count, song_support = eligible[0]
        rule = MergeRule(
            left=pair[0],
            right=pair[1],
            merged=pair[0] + pair[1],
            fit_count=count,
            fit_song_support=song_support,
        )
        rules.append(rule)
        known.add(rule.merged)
        current = {
            stream_id: apply_merge_rule(sequence, rule)
            for stream_id, sequence in current.items()
        }
    return tuple(rules)


def sounding_groups(symbol: Symbol) -> int:
    return sum(atom[0] != REST for atom in symbol)


def rest_units(symbol: Symbol) -> int:
    return sum(atom[0] == REST for atom in symbol)


@dataclass(slots=True)
class AtomLiteralModel:
    alpha: float
    kinds: Counter[str]
    group_sizes: Counter[int]
    pitch_classes: Counter[int]
    octaves: Counter[int]
    durations: Counter[int]
    rest_durations: Counter[int]
    generic_atoms: Counter[str]
    support_lengths: Counter[int]

    @classmethod
    def fit(
        cls, base_sequences: Iterable[Sequence[Symbol]], *, alpha: float
    ) -> "AtomLiteralModel":
        model = cls(
            alpha=float(alpha),
            kinds=Counter(),
            group_sizes=Counter(),
            pitch_classes=Counter(),
            octaves=Counter(),
            durations=Counter(),
            rest_durations=Counter(),
            generic_atoms=Counter(),
            support_lengths=Counter(),
        )
        for sequence in base_sequences:
            for symbol in sequence:
                model.support_lengths[len(symbol)] += 1
                for atom in symbol:
                    kind = str(atom[0])
                    model.kinds[kind] += 1
                    if kind == REST:
                        model.rest_durations[int(atom[1])] += 1
                    elif kind == NOTE_GROUP:
                        rows = atom[1]
                        model.group_sizes[len(rows)] += 1
                        for pitch_class, octave, duration in rows:
                            model.pitch_classes[int(pitch_class)] += 1
                            model.octaves[int(octave)] += 1
                            model.durations[int(duration)] += 1
                    else:
                        model.generic_atoms[repr(atom)] += 1
        return model

    @staticmethod
    def _bits(
        counter: Counter[object], value: object, alpha: float, floor: int
    ) -> float:
        vocabulary = max(int(floor), len(counter) + 1)
        probability = (counter[value] + alpha) / (
            sum(counter.values()) + alpha * vocabulary
        )
        result = -math.log2(probability)
        if value not in counter and isinstance(value, int):
            result += 2.0 * math.log2(abs(value) + 2.0) + 1.0
        return result

    def atom_bits(self, atom: AtomKey) -> float:
        kind = str(atom[0])
        bits = self._bits(self.kinds, kind, self.alpha, 2)
        if kind == REST:
            return bits + self._bits(
                self.rest_durations, int(atom[1]), self.alpha, 2
            )
        if kind != NOTE_GROUP:
            return bits + self._bits(
                self.generic_atoms, repr(atom), self.alpha, 2
            )
        rows = atom[1]
        bits += self._bits(self.group_sizes, len(rows), self.alpha, 2)
        for pitch_class, octave, duration in rows:
            bits += self._bits(
                self.pitch_classes, int(pitch_class), self.alpha, 12
            )
            bits += self._bits(self.octaves, int(octave), self.alpha, 12)
            bits += self._bits(self.durations, int(duration), self.alpha, 2)
        return bits

    def symbol_bits(self, symbol: Symbol) -> float:
        return self._bits(
            self.support_lengths, len(symbol), self.alpha, 2
        ) + sum(self.atom_bits(atom) for atom in symbol)


@dataclass(slots=True)
class CausalSymbolScorer:
    alpha: float
    bigram_weight: float
    vocabulary: set[Symbol]
    unigram: Counter[object]
    contexts: dict[object, Counter[object]]
    literal_model: AtomLiteralModel

    @classmethod
    def fit(
        cls,
        tokenized: Mapping[str, Sequence[Symbol]],
        base_sequences: Mapping[str, Sequence[Symbol]],
        *,
        alpha: float = 0.25,
        bigram_weight: float = 0.75,
        declared_vocabulary: Iterable[Symbol] = (),
    ) -> "CausalSymbolScorer":
        vocabulary = set(declared_vocabulary)
        vocabulary.update(
            symbol for sequence in tokenized.values() for symbol in sequence
        )
        unigram: Counter[object] = Counter()
        contexts: dict[object, Counter[object]] = defaultdict(Counter)
        for sequence in tokenized.values():
            previous: object = BOS
            for symbol in sequence:
                unigram[symbol] += 1
                contexts[previous][symbol] += 1
                previous = symbol
        unigram[ESC] += 0
        return cls(
            alpha=float(alpha),
            bigram_weight=float(bigram_weight),
            vocabulary=vocabulary,
            unigram=unigram,
            contexts=dict(contexts),
            literal_model=AtomLiteralModel.fit(
                base_sequences.values(), alpha=float(alpha)
            ),
        )

    def probability(self, target: object, previous: object) -> float:
        vocabulary_size = len(self.vocabulary) + 1
        unigram_probability = (self.unigram[target] + self.alpha) / (
            sum(self.unigram.values()) + self.alpha * vocabulary_size
        )
        context = self.contexts.get(previous, Counter())
        conditional_probability = (context[target] + self.alpha) / (
            sum(context.values()) + self.alpha * vocabulary_size
        )
        return (
            self.bigram_weight * conditional_probability
            + (1.0 - self.bigram_weight) * unigram_probability
        )

    def score(self, sequence: Sequence[Symbol]) -> dict[str, float]:
        previous: object = BOS
        total_bits = 0.0
        unknown_symbols = 0
        raw_units = 0
        raw_sounding_groups = 0
        raw_rests = 0
        for symbol in sequence:
            known = symbol in self.vocabulary
            target: object = symbol if known else ESC
            mapped_previous = (
                previous
                if previous == BOS or previous in self.vocabulary
                else ESC
            )
            bits = -math.log2(self.probability(target, mapped_previous))
            if not known:
                bits += self.literal_model.symbol_bits(symbol)
                unknown_symbols += 1
            total_bits += bits
            raw_units += len(symbol)
            raw_sounding_groups += sounding_groups(symbol)
            raw_rests += rest_units(symbol)
            previous = target
        return {
            "total_bits": total_bits,
            "raw_units": float(raw_units),
            "raw_sounding_groups": float(raw_sounding_groups),
            "raw_rests": float(raw_rests),
            "emitted_symbols": float(len(sequence)),
            "unknown_symbols": float(unknown_symbols),
        }


def evaluate_scorer(
    scorer: CausalSymbolScorer,
    tokenized: Mapping[str, Sequence[Symbol]],
) -> dict[str, object]:
    totals: Counter[str] = Counter()
    song_rows: list[dict[str, object]] = []
    for stream_id, sequence in sorted(tokenized.items()):
        row = scorer.score(sequence)
        totals.update(row)
        song_rows.append(
            {
                "song_id": stream_id.split(":", 1)[0],
                **row,
                "bits_per_raw_unit": row["total_bits"]
                / max(row["raw_units"], 1.0),
                "bits_per_sounding_group": row["total_bits"]
                / max(row["raw_sounding_groups"], 1.0),
            }
        )
    return {
        **dict(totals),
        "bits_per_raw_unit": totals["total_bits"]
        / max(totals["raw_units"], 1.0),
        "bits_per_sounding_group": totals["total_bits"]
        / max(totals["raw_sounding_groups"], 1.0),
        "sequence_reduction": 1.0
        - totals["emitted_symbols"] / max(totals["raw_units"], 1.0),
        "song_rows": song_rows,
    }


@dataclass(frozen=True, slots=True)
class PredictiveMergeStep:
    index: int
    rule: MergeRule
    candidates_scored: int
    selector_occurrences: int
    selector_bits_before: float
    selector_bits_after: float
    gain_bits: float
    gain_bits_per_raw_unit: float
    incremental_table_bits: float


def _pair_occurrences(
    sequences: Mapping[str, Sequence[Symbol]], pair: tuple[Symbol, Symbol]
) -> int:
    return sum(
        sum(
            left == pair[0] and right == pair[1]
            for left, right in zip(sequence, sequence[1:])
        )
        for sequence in sequences.values()
    )


def fit_predictive_bpe(
    fit_sequences: Mapping[str, Sequence[Symbol]],
    selector_sequences: Mapping[str, Sequence[Symbol]],
    *,
    merge_budget: int,
    candidate_pool_size: int,
    minimum_count: int,
    minimum_song_support: int,
    maximum_support_units: int,
    alpha: float = 0.25,
    bigram_weight: float = 0.75,
    minimum_gain_bits_per_raw_unit: float = 0.0,
) -> tuple[tuple[MergeRule, ...], tuple[PredictiveMergeStep, ...]]:
    """Greedily choose sounding-only merges by selector causal NLL gain.

    Frequency only proposes a small candidate pool from ``fit_sequences``.
    Every accepted rule must reduce total causal coding bits on the disjoint
    ``selector_sequences``.  REST remains an unmergeable boundary.
    """

    current_fit = {key: tuple(value) for key, value in fit_sequences.items()}
    current_selector = {
        key: tuple(value) for key, value in selector_sequences.items()
    }
    raw_fit = {key: tuple(value) for key, value in fit_sequences.items()}
    base_vocabulary = {
        symbol for sequence in raw_fit.values() for symbol in sequence
    }
    selected_rules: list[MergeRule] = []
    steps: list[PredictiveMergeStep] = []
    selector_raw_units = sum(len(sequence) for sequence in selector_sequences.values())

    for index in range(int(merge_budget)):
        candidates = frequency_pair_candidates(
            current_fit,
            minimum_count=minimum_count,
            minimum_song_support=minimum_song_support,
            maximum_support_units=maximum_support_units,
            forbid_rest=True,
            limit=candidate_pool_size,
        )
        candidates = tuple(
            row for row in candidates if _pair_occurrences(current_selector, row[0]) > 0
        )
        if not candidates:
            break
        current_vocabulary = base_vocabulary | {
            rule.merged for rule in selected_rules
        }
        baseline_scorer = CausalSymbolScorer.fit(
            current_fit,
            raw_fit,
            alpha=alpha,
            bigram_weight=bigram_weight,
            declared_vocabulary=current_vocabulary,
        )
        baseline_bits = float(
            evaluate_scorer(baseline_scorer, current_selector)["total_bits"]
        )
        scored: list[
            tuple[
                float,
                str,
                MergeRule,
                dict[str, tuple[Symbol, ...]],
                dict[str, tuple[Symbol, ...]],
                float,
                int,
            ]
        ] = []
        for pair, count, song_support in candidates:
            rule = MergeRule(
                left=pair[0],
                right=pair[1],
                merged=pair[0] + pair[1],
                fit_count=count,
                fit_song_support=song_support,
            )
            candidate_fit = {
                key: apply_merge_rule(sequence, rule)
                for key, sequence in current_fit.items()
            }
            candidate_selector = {
                key: apply_merge_rule(sequence, rule)
                for key, sequence in current_selector.items()
            }
            scorer = CausalSymbolScorer.fit(
                candidate_fit,
                raw_fit,
                alpha=alpha,
                bigram_weight=bigram_weight,
                declared_vocabulary=current_vocabulary | {rule.merged},
            )
            after_bits = float(
                evaluate_scorer(scorer, candidate_selector)["total_bits"]
            )
            gain = baseline_bits - after_bits
            selector_occurrences = sum(
                len(current_selector[key]) - len(candidate_selector[key])
                for key in current_selector
            )
            scored.append(
                (
                    gain,
                    stable_digest(rule.merged),
                    rule,
                    candidate_fit,
                    candidate_selector,
                    after_bits,
                    selector_occurrences,
                )
            )
        scored.sort(key=lambda row: (-row[0], row[1]))
        (
            gain,
            _digest,
            rule,
            next_fit,
            next_selector,
            after_bits,
            selector_occurrences,
        ) = scored[0]
        gain_per_unit = gain / max(selector_raw_units, 1)
        if gain_per_unit <= float(minimum_gain_bits_per_raw_unit):
            break
        reference_bits = math.ceil(
            math.log2(max(len(current_vocabulary) + 1, 2))
        )
        steps.append(
            PredictiveMergeStep(
                index=index,
                rule=rule,
                candidates_scored=len(scored),
                selector_occurrences=selector_occurrences,
                selector_bits_before=baseline_bits,
                selector_bits_after=after_bits,
                gain_bits=gain,
                gain_bits_per_raw_unit=gain_per_unit,
                incremental_table_bits=2.0 * reference_bits,
            )
        )
        selected_rules.append(rule)
        current_fit = next_fit
        current_selector = next_selector
    return tuple(selected_rules), tuple(steps)


def predictive_step_receipt(step: PredictiveMergeStep) -> dict[str, object]:
    return {
        "index": step.index,
        "candidates_scored": step.candidates_scored,
        "selector_occurrences": step.selector_occurrences,
        "selector_bits_before": step.selector_bits_before,
        "selector_bits_after": step.selector_bits_after,
        "gain_bits": step.gain_bits,
        "gain_bits_per_raw_unit": step.gain_bits_per_raw_unit,
        "incremental_table_bits_diagnostic": step.incremental_table_bits,
        "rule": rule_receipt(step.rule),
    }


def merge_table_bits(rules: Sequence[MergeRule], base_vocabulary_size: int) -> float:
    """Minimal merge-table reference cost, reported only as a diagnostic."""

    total = 0.0
    vocabulary_size = max(int(base_vocabulary_size), 2)
    for index, _rule in enumerate(rules):
        reference_bits = math.ceil(math.log2(vocabulary_size + index + 1))
        total += 2.0 * reference_bits
    return total


def rule_receipt(rule: MergeRule) -> dict[str, object]:
    return {
        "fit_count": rule.fit_count,
        "fit_song_support": rule.fit_song_support,
        "support_units": rule.support_units,
        "sounding_groups": sounding_groups(rule.merged),
        "rest_units": rest_units(rule.merged),
        "expansion_sha256": stable_digest(rule.merged),
        "expansion": rule.merged,
    }
