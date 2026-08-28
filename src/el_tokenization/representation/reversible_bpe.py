from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Symbol = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReversibleBPE:
    merges: tuple[tuple[Symbol, Symbol], ...]
    fit_split: str = "train"

    def encode(self, tokens: Sequence[str]) -> tuple[Symbol, ...]:
        symbols: list[Symbol] = [(token,) for token in tokens]
        for left, right in self.merges:
            merged = left + right
            result: list[Symbol] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == left
                    and symbols[index + 1] == right
                ):
                    result.append(merged)
                    index += 2
                else:
                    result.append(symbols[index])
                    index += 1
            symbols = result
        return tuple(symbols)

    @staticmethod
    def decode(symbols: Sequence[Symbol]) -> tuple[str, ...]:
        return tuple(token for symbol in symbols for token in symbol)


def fit_reversible_bpe(
    sequences: Iterable[Sequence[str]],
    *,
    merge_count: int,
    split: str,
) -> ReversibleBPE:
    if split != "train":
        raise ValueError("BPE fitting is allowed on train only")
    working: list[list[Symbol]] = [[(token,) for token in sequence] for sequence in sequences]
    merges: list[tuple[Symbol, Symbol]] = []
    for _ in range(merge_count):
        counts: Counter[tuple[Symbol, Symbol]] = Counter(
            (sequence[index], sequence[index + 1])
            for sequence in working
            for index in range(len(sequence) - 1)
        )
        if not counts:
            break
        pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency < 2:
            break
        merges.append(pair)
        model = ReversibleBPE(tuple(merges))
        working = [
            list(model.encode([token for symbol in sequence for token in symbol]))
            for sequence in working
        ]
    return ReversibleBPE(tuple(merges))
