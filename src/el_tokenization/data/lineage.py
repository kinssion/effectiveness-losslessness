from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_relative_path(path: Path, data_root: Path) -> str:
    resolved = path.resolve()
    relative = resolved.relative_to(data_root.resolve())
    return relative.as_posix()
