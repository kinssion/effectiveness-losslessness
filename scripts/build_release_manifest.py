#!/usr/bin/env python3
"""Create the immutable file inventory and release-tree hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scan_release import SKIP_DIRECTORIES, included_files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("release_manifest.json"))
    parser.add_argument("--sums", type=Path, default=Path("SHA256SUMS.txt"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    sums = args.sums if args.sums.is_absolute() else root / args.sums
    excluded = {output.resolve(), sums.resolve()}
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in included_files(root)
        if path.resolve() not in excluded
    ]
    tree_payload = json.dumps(
        [(row["path"], row["sha256"], row["bytes"]) for row in rows],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    manifest = {
        "schema_version": "el.release.v1",
        "release": "effectiveness-losslessness-1.0.0-rc2",
        "release_stage": "private_pre_public",
        "paper": "arXiv:2608.18025",
        "repository": "https://github.com/kinssion/effectiveness-losslessness",
        "author": "Yi Wang",
        "orcid": "0009-0004-6057-8151",
        "raw_midi_included": False,
        "model_weights_included": False,
        "ignored_directories": sorted(SKIP_DIRECTORIES),
        "file_count": len(rows),
        "total_bytes": sum(int(str(row["bytes"])) for row in rows),
        "release_tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "files": rows,
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sum_rows = [*rows, {"path": output.relative_to(root).as_posix(), "sha256": _sha256(output)}]
    sums.write_text(
        "".join(
            f"{row['sha256']}  {row['path']}\n"
            for row in sorted(sum_rows, key=lambda row: str(row["path"]))
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "file_count": len(rows),
                "release_tree_sha256": manifest["release_tree_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
