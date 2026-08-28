from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .lineage import sha256_file


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_official_sources(data_root: Path, source_manifest: Path) -> dict[str, int]:
    checked = 0
    mismatched = 0
    missing = 0
    for row in iter_jsonl(source_manifest):
        relative = str(row["relative_source_path"])
        path = data_root / relative
        if not path.is_file():
            missing += 1
            continue
        checked += 1
        if sha256_file(path) != str(row["content_sha256"]):
            mismatched += 1
    return {"checked": checked, "missing": missing, "mismatched": mismatched}
