#!/usr/bin/env python3
"""Remove machine-local paths and infrastructure identities from copied JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ABSOLUTE_PATH = re.compile(
    r"^(?:[A-Za-z]:[\\/]|/(?:data|home|mnt|scratch|workspace)/)", re.IGNORECASE
)
IDENTITY_KEYS = {
    "account",
    "cluster",
    "host",
    "hostname",
    "node",
    "slurm_account",
    "user",
    "username",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sanitize(value: Any, pointer: str, changed: list[str], key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            item_key: _sanitize(item, f"{pointer}/{item_key}", changed, str(item_key).lower())
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize(item, f"{pointer}/{index}", changed, key)
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if key in IDENTITY_KEYS:
            changed.append(pointer)
            return "<REDACTED_INFRASTRUCTURE_IDENTITY>"
        if ABSOLUTE_PATH.match(value):
            changed.append(pointer)
            return "<REDACTED_ABSOLUTE_PATH>"
        if "@" in value and not value.startswith(("http://", "https://")):
            changed.append(pointer)
            return "<REDACTED_CONTACT_OR_LOGIN>"
    return value


def _sanitize_json(path: Path) -> dict[str, object] | None:
    before = path.read_bytes()
    value = json.loads(before.decode("utf-8"))
    changed: list[str] = []
    clean = _sanitize(value, "", changed)
    if not changed:
        return None
    after = (json.dumps(clean, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(after)
    return {
        "file": path.as_posix(),
        "original_sha256": _sha256(before),
        "review_copy_sha256": _sha256(after),
        "redacted_json_pointer_count": len(changed),
        "redacted_json_pointers": changed,
    }


def _sanitize_jsonl(path: Path) -> dict[str, object] | None:
    before = path.read_bytes()
    output: list[str] = []
    changed_pointers: list[str] = []
    for line_number, line in enumerate(before.decode("utf-8").splitlines(), start=1):
        value = json.loads(line)
        line_changes: list[str] = []
        clean = _sanitize(value, f"line:{line_number}", line_changes)
        changed_pointers.extend(line_changes)
        output.append(json.dumps(clean, sort_keys=True, ensure_ascii=False))
    if not changed_pointers:
        return None
    after = ("\n".join(output) + "\n").encode("utf-8")
    path.write_bytes(after)
    return {
        "file": path.as_posix(),
        "original_sha256": _sha256(before),
        "review_copy_sha256": _sha256(after),
        "redacted_json_pointer_count": len(changed_pointers),
        "redacted_json_pointers": changed_pointers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/paper_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "redaction_manifest.json":
            continue
        result = (
            _sanitize_jsonl(path)
            if path.suffix == ".jsonl"
            else _sanitize_json(path)
            if path.suffix == ".json"
            else None
        )
        if result is not None:
            result["file"] = path.relative_to(root).as_posix()
            entries.append(result)
    manifest = {
        "schema_version": "el.review_redaction_manifest.v1",
        "policy": "Only local paths, contact/login strings, and infrastructure identities are redacted; scientific numbers and hashes are retained.",
        "changed_file_count": len(entries),
        "files": entries,
    }
    destination = root / "redaction_manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "changed_file_count": len(entries)}))


if __name__ == "__main__":
    main()
