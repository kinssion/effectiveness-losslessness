from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Literal


FROZEN_POPK_CLEAN_V1_SHA256 = (
    "38f3b1b810e2ebc2168e8f73d001e403b047867e7eb47c985ee5ae2307b561f3"
)
SplitName = Literal["train", "validation", "test"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PopKManifestEntry:
    sample_id: str
    source_path: str
    source_song_id: str | None
    lineage_id: str
    exact_hash: str
    transposition_invariant_hash: str
    rhythm_interval_hash: str
    note_count: int
    onset_count: int
    duration_beats: float
    preprocessing_version: str
    split: SplitName


class PopKCleanManifest:
    """Frozen, explicit Pop-K split interface.

    There is intentionally no default path and no cache-order fallback. Test
    access is sealed unless the caller passes an explicit authorization flag.
    """

    def __init__(
        self,
        manifest_path: str | Path | None,
        *,
        repository_root: str | Path,
        expected_sha256: str = FROZEN_POPK_CLEAN_V1_SHA256,
        verify_source_files: bool = True,
        defer_test_rows: bool = False,
    ) -> None:
        if manifest_path is None:
            raise ValueError(
                "an explicit popk_clean_v1 split_manifest.json path is required; "
                "legacy last-N/cache-order splitting is forbidden"
            )
        self.manifest_path = Path(manifest_path)
        self.repository_root = Path(repository_root)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest_sha256 = _sha256(self.manifest_path)
        if self.manifest_sha256 != str(expected_sha256):
            raise ValueError(
                "Pop-K clean manifest SHA-256 mismatch: "
                f"{self.manifest_sha256} != {expected_sha256}"
            )
        sha_path = self.manifest_path.with_suffix(".sha256")
        if not sha_path.is_file():
            raise FileNotFoundError(sha_path)
        frozen_sidecar = sha_path.read_text(encoding="ascii").split()[0]
        if frozen_sidecar != self.manifest_sha256:
            raise ValueError("split_manifest.sha256 does not match split_manifest.json")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "m4l.popk_clean_split.v1":
            raise ValueError("unsupported Pop-K clean manifest schema")

        self._entries: dict[SplitName, tuple[PopKManifestEntry, ...] | None] = {}
        self._verify_source_files = bool(verify_source_files)
        self._defer_test_rows = bool(defer_test_rows)
        seen_ids: set[str] = set()
        relation_splits: dict[str, dict[str, SplitName]] = {
            "lineage_id": {},
            "exact_hash": {},
            "transposition_invariant_hash": {},
            "rhythm_interval_hash": {},
        }
        expected_counts = self.manifest["split_policy"]["actual"]
        for split in ("train", "validation", "test"):
            jsonl_path = self.manifest_path.parent / f"{split}.jsonl"
            if not jsonl_path.is_file():
                raise FileNotFoundError(jsonl_path)
            receipt = self._output_receipt(jsonl_path)
            if receipt is None:
                raise ValueError(f"manifest has no receipt for {jsonl_path}")
            if split == "test" and self._defer_test_rows:
                if int(receipt["rows"]) != int(expected_counts["test"]):
                    raise ValueError("sealed test receipt row count mismatch")
                self._entries["test"] = None
                continue
            if _sha256(jsonl_path) != receipt["sha256"]:
                raise ValueError(f"split JSONL hash mismatch: {jsonl_path}")
            rows: list[PopKManifestEntry] = []
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    value = json.loads(line)
                    sample_id = str(value["sample_id"])
                    if sample_id in seen_ids:
                        raise ValueError(
                            f"duplicate sample_id across clean manifests: {sample_id}"
                        )
                    seen_ids.add(sample_id)
                    entry = PopKManifestEntry(
                        sample_id=sample_id,
                        source_path=str(value["source_path"]),
                        source_song_id=value.get("source_song_id"),
                        lineage_id=str(value["lineage_id"]),
                        exact_hash=str(value["exact_hash"]),
                        transposition_invariant_hash=str(
                            value["transposition_invariant_hash"]
                        ),
                        rhythm_interval_hash=str(value["rhythm_interval_hash"]),
                        note_count=int(value["note_count"]),
                        onset_count=int(value["onset_count"]),
                        duration_beats=float(value["duration_beats"]),
                        preprocessing_version=str(value["preprocessing_version"]),
                        split=split,
                    )
                    if verify_source_files and not self.source_path(entry).is_file():
                        raise FileNotFoundError(
                            f"missing source for {sample_id} at line {line_number}: "
                            f"{self.source_path(entry)}"
                        )
                    for relation in relation_splits:
                        key = str(getattr(entry, relation))
                        prior = relation_splits[relation].setdefault(key, split)
                        if prior != split:
                            raise ValueError(
                                f"{relation} crosses split: {key}: {prior}/{split}"
                            )
                    rows.append(entry)
            if len(rows) != int(expected_counts[split]):
                raise ValueError(
                    f"{split} row count mismatch: {len(rows)} != {expected_counts[split]}"
                )
            if len(rows) != int(receipt["rows"]):
                raise ValueError(
                    f"{split} receipt row count mismatch: {len(rows)} != {receipt['rows']}"
                )
            self._entries[split] = tuple(rows)
        expected_total = int(self.manifest["input"]["sample_count"])
        observed_or_sealed = len(seen_ids) + (
            int(expected_counts["test"]) if self._defer_test_rows else 0
        )
        if observed_or_sealed != expected_total:
            raise ValueError(
                "clean manifest is missing entries: "
                f"{observed_or_sealed} != {expected_total}"
            )
        self.development_reads = 0
        self.test_reads = 0
        self.test_audit_reads = 0
        self.test_row_file_reads = 0

    def _output_receipt(self, path: Path) -> dict[str, object] | None:
        normalized = path.as_posix()
        for key, receipt in self.manifest["outputs"].items():
            if Path(key).as_posix() == normalized or Path(key).name == path.name:
                return receipt
        return None

    def source_path(self, entry: PopKManifestEntry) -> Path:
        value = Path(entry.source_path)
        return value if value.is_absolute() else self.repository_root / value

    @property
    def train(self) -> tuple[PopKManifestEntry, ...]:
        entries = self._entries["train"]
        assert entries is not None
        return entries

    @property
    def validation(self) -> tuple[PopKManifestEntry, ...]:
        self.development_reads += 1
        entries = self._entries["validation"]
        assert entries is not None
        return entries

    @property
    def test_rows_loaded(self) -> bool:
        return self._entries["test"] is not None

    def _load_deferred_test_rows(self) -> tuple[PopKManifestEntry, ...]:
        if self._entries["test"] is not None:
            return self._entries["test"]  # type: ignore[return-value]
        jsonl_path = self.manifest_path.parent / "test.jsonl"
        receipt = self._output_receipt(jsonl_path)
        if receipt is None:
            raise ValueError("manifest has no receipt for sealed test JSONL")
        if _sha256(jsonl_path) != receipt["sha256"]:
            raise ValueError(f"split JSONL hash mismatch: {jsonl_path}")
        occupied_ids = {
            entry.sample_id
            for split in ("train", "validation")
            for entry in (self._entries[split] or ())
        }
        occupied_relations: dict[str, dict[str, SplitName]] = {
            relation: {
                str(getattr(entry, relation)): split
                for split in ("train", "validation")
                for entry in (self._entries[split] or ())
            }
            for relation in (
                "lineage_id",
                "exact_hash",
                "transposition_invariant_hash",
                "rhythm_interval_hash",
            )
        }
        rows: list[PopKManifestEntry] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                sample_id = str(value["sample_id"])
                if sample_id in occupied_ids:
                    raise ValueError(f"duplicate sample_id in sealed test: {sample_id}")
                occupied_ids.add(sample_id)
                entry = PopKManifestEntry(
                    sample_id=sample_id,
                    source_path=str(value["source_path"]),
                    source_song_id=value.get("source_song_id"),
                    lineage_id=str(value["lineage_id"]),
                    exact_hash=str(value["exact_hash"]),
                    transposition_invariant_hash=str(value["transposition_invariant_hash"]),
                    rhythm_interval_hash=str(value["rhythm_interval_hash"]),
                    note_count=int(value["note_count"]),
                    onset_count=int(value["onset_count"]),
                    duration_beats=float(value["duration_beats"]),
                    preprocessing_version=str(value["preprocessing_version"]),
                    split="test",
                )
                if self._verify_source_files and not self.source_path(entry).is_file():
                    raise FileNotFoundError(
                        f"missing source for {sample_id} at line {line_number}: "
                        f"{self.source_path(entry)}"
                    )
                for relation, mapping in occupied_relations.items():
                    key = str(getattr(entry, relation))
                    prior = mapping.setdefault(key, "test")
                    if prior != "test":
                        raise ValueError(
                            f"{relation} crosses split: {key}: {prior}/test"
                        )
                rows.append(entry)
        expected = int(self.manifest["split_policy"]["actual"]["test"])
        if len(rows) != expected or len(rows) != int(receipt["rows"]):
            raise ValueError(f"sealed test row count mismatch: {len(rows)} != {expected}")
        self._entries["test"] = tuple(rows)
        self.test_row_file_reads += 1
        return self._entries["test"]  # type: ignore[return-value]

    def test(self, *, allow_test_evaluation: bool = False) -> tuple[PopKManifestEntry, ...]:
        if not allow_test_evaluation:
            raise PermissionError(
                "test evaluation is sealed by default; pass explicit "
                "allow_test_evaluation=True only after model selection is frozen"
            )
        self.test_reads += 1
        return self._load_deferred_test_rows()

    def audit_entries(
        self, split: SplitName, *, purpose: str
    ) -> tuple[PopKManifestEntry, ...]:
        """Expose frozen rows for non-model governance audits with an explicit receipt.

        This is not an evaluation path. In particular, it must not be used by a
        trainer, evaluator, checkpoint selector, or demo selector.
        """

        if not purpose.strip():
            raise ValueError("manifest audit access requires a non-empty purpose")
        if split == "test":
            self.test_audit_reads += 1
            return self._load_deferred_test_rows()
        entries = self._entries[split]
        assert entries is not None
        return entries

    def entries(
        self,
        split: SplitName,
        *,
        allow_test_evaluation: bool = False,
    ) -> tuple[PopKManifestEntry, ...]:
        if split == "train":
            return self.train
        if split == "validation":
            return self.validation
        if split == "test":
            return self.test(allow_test_evaluation=allow_test_evaluation)
        raise ValueError(f"unsupported split: {split}")

    def iter_sources(
        self,
        split: SplitName,
        *,
        allow_test_evaluation: bool = False,
    ) -> Iterator[tuple[PopKManifestEntry, Path]]:
        for entry in self.entries(
            split, allow_test_evaluation=allow_test_evaluation
        ):
            yield entry, self.source_path(entry)
