from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TestAccessFirewall:
    validation_locked: bool = False
    accessor_calls: int = 0
    row_file_reads: int = 0
    _consumed: bool = False

    def lock_validation(self) -> None:
        self.validation_locked = True

    def training_loader(self, split: str) -> str:
        if split == "test":
            raise PermissionError("test loader is unavailable during training")
        return split

    def read_test_rows_once(self, manifest: Path) -> list[str]:
        self.accessor_calls += 1
        if not self.validation_locked:
            raise PermissionError("validation must be locked before test access")
        if self._consumed:
            raise PermissionError("sealed test accessor was already consumed")
        self._consumed = True
        self.row_file_reads += 1
        return manifest.read_text(encoding="utf-8").splitlines()
