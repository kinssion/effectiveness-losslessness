from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PARAMETER_COUNTS = {
    "A": 809_475,
    "B": 811_079,
    "C": 810_499,
    "D": 812_815,
    "E": 805_455,
    "F": 812_815,
    "G": 812_815,
    "H": 813_519,
    "I": 805_455,
    "J": 689_349,
    "K": 804_675,
}

DEFAULT_DECODER_SCHEMA = ("type", "time", "pitch", "duration", "EOS")


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    arm: str
    operation: str
    coordinate_schema: tuple[str, ...]
    side_information_schema: tuple[str, ...] = ()
    decoder_schema: tuple[str, ...] = DEFAULT_DECODER_SCHEMA
    expected_round_trip: str = "exact_declared_NOTE_REST_facts"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RepresentationConfig:
        return cls(
            arm=str(value["arm"]).upper(),
            operation=str(value["operation"]),
            coordinate_schema=tuple(str(item) for item in value["coordinate_schema"]),
            side_information_schema=tuple(
                str(item) for item in value.get("side_information_schema", ())
            ),
            decoder_schema=tuple(
                str(item) for item in value.get("decoder_schema", DEFAULT_DECODER_SCHEMA)
            ),
            expected_round_trip=str(
                value.get("expected_round_trip", "exact_declared_NOTE_REST_facts")
            ),
        )


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    config: RepresentationConfig
    active_parameter_count: int

    def receipt(self) -> dict[str, object]:
        return {
            "arm": self.config.arm,
            "operation": self.config.operation,
            "active_parameter_count": self.active_parameter_count,
            "coordinate_schema": list(self.config.coordinate_schema),
            "side_information_schema": list(self.config.side_information_schema),
            "decoder_schema": list(self.config.decoder_schema),
            "expected_round_trip": self.config.expected_round_trip,
        }


def build_representation(
    config: RepresentationConfig | Mapping[str, Any] | Path | str,
) -> RepresentationSpec:
    if isinstance(config, (str, Path)):
        payload = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        config = RepresentationConfig.from_mapping(payload)
    elif isinstance(config, Mapping):
        config = RepresentationConfig.from_mapping(config)
    if config.arm not in PARAMETER_COUNTS:
        raise ValueError(f"unknown paper arm: {config.arm!r}")
    return RepresentationSpec(config, PARAMETER_COUNTS[config.arm])


__all__ = ["RepresentationConfig", "RepresentationSpec", "build_representation"]
