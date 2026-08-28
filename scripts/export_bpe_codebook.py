#!/usr/bin/env python3
"""Export the frozen K tokenizer into reviewable vocabulary and merge files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/paper_v1"))
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    source = root / "run_metadata/pop1k7_jk/train_only_lossless_bpe.json"
    tokenizer = json.loads(source.read_text(encoding="utf-8"))
    model = tokenizer["model"]
    output = root / "tokenizers/pop1k7_K"
    _dump(output / "vocab.json", model["vocab"])
    _dump(output / "merges.json", model["merges"])
    _dump(
        output / "bpe_config.json",
        {
            "schema_version": "el.frozen_bpe_codebook.v1",
            "fit_split": "train",
            "source_tokenizer_sha256": _sha256(source),
            "vocabulary_size": len(model["vocab"]),
            "learned_merge_count": len(model["merges"]),
            "source_training_receipt": "../../run_metadata/pop1k7_jk/bpe_training_receipt.json",
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "vocabulary_size": len(model["vocab"]),
                "merges": len(model["merges"]),
            }
        )
    )


if __name__ == "__main__":
    main()
