from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictiveLedgerEntry:
    declared_fact_count: int
    type_bits: float
    time_bits: float
    pitch_bits: float
    duration_bits: float
    eos_bits: float = 0.0
    side_information_bits: float = 0.0

    def __post_init__(self) -> None:
        if self.declared_fact_count <= 0:
            raise ValueError("declared_fact_count must be positive")
        if (
            min(
                self.type_bits,
                self.time_bits,
                self.pitch_bits,
                self.duration_bits,
                self.eos_bits,
                self.side_information_bits,
            )
            < 0
        ):
            raise ValueError("ledger bits must be non-negative")

    @property
    def total_bits(self) -> float:
        return (
            self.type_bits
            + self.time_bits
            + self.pitch_bits
            + self.duration_bits
            + self.eos_bits
            + self.side_information_bits
        )

    @property
    def bits_per_declared_fact(self) -> float:
        return self.total_bits / self.declared_fact_count

    def factor_bits_per_declared_fact(self) -> dict[str, float]:
        denominator = self.declared_fact_count
        return {
            "type": self.type_bits / denominator,
            "time": self.time_bits / denominator,
            "pitch": self.pitch_bits / denominator,
            "duration": self.duration_bits / denominator,
            "EOS": self.eos_bits / denominator,
            "side": self.side_information_bits / denominator,
        }


class PredictiveLedger:
    def __init__(self) -> None:
        self._entries: list[PredictiveLedgerEntry] = []

    def add(self, entry: PredictiveLedgerEntry) -> None:
        self._entries.append(entry)

    def aggregate(self) -> PredictiveLedgerEntry:
        if not self._entries:
            raise ValueError("cannot aggregate an empty ledger")
        return PredictiveLedgerEntry(
            declared_fact_count=sum(entry.declared_fact_count for entry in self._entries),
            type_bits=sum(entry.type_bits for entry in self._entries),
            time_bits=sum(entry.time_bits for entry in self._entries),
            pitch_bits=sum(entry.pitch_bits for entry in self._entries),
            duration_bits=sum(entry.duration_bits for entry in self._entries),
            eos_bits=sum(entry.eos_bits for entry in self._entries),
            side_information_bits=sum(entry.side_information_bits for entry in self._entries),
        )
