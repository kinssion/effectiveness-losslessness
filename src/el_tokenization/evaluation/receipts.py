from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_receipt_fields(
    receipt: Mapping[str, object], required: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(field for field in required if field not in receipt)


def checkpoint_config_hash_matches(
    checkpoint_metadata: Mapping[str, object], config_path: Path
) -> bool:
    expected = checkpoint_metadata.get("config_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    return file_sha256(config_path) == expected
