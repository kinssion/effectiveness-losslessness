from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..training.test_firewall import TestAccessFirewall


@dataclass(frozen=True, slots=True)
class SealedTestReceipt:
    test_manifest_accessor_calls: int
    test_row_file_reads: int
    checkpoint_updates: int
    test_time_bpe_fit_updates: int
    enforceable_security_boundary: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def open_once(firewall: TestAccessFirewall, manifest: Path) -> list[str]:
    return firewall.read_test_rows_once(manifest)
