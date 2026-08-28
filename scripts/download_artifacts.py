#!/usr/bin/env python3
"""Download separately hosted selected checkpoints and verify SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/paper_v1/checkpoint_index.json")
    )
    parser.add_argument("--output", type=Path, default=Path("weights/downloaded"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    if args.list:
        print(json.dumps(index, indent=2, sort_keys=True))
        return
    missing = [
        row["checkpoint_id"] for row in index["checkpoints"] if not row.get("download_url")
    ]
    if missing:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": "download URLs are not published",
                    "checkpoint_ids": missing,
                },
                indent=2,
            )
        )
        raise SystemExit(2)
    args.output.mkdir(parents=True, exist_ok=True)
    for row in index["checkpoints"]:
        destination = args.output / row["expected_filename"]
        with (
            urlopen(row["download_url"], timeout=120) as response,
            destination.open("wb") as handle,
        ):
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if _sha256(destination) != row["checkpoint_sha256"]:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"checkpoint hash mismatch: {row['checkpoint_id']}")
    print(json.dumps({"status": "PASS", "downloaded": len(index["checkpoints"])}))


if __name__ == "__main__":
    main()
