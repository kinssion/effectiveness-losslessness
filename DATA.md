# Data and manifests

Raw MIDI and tokenized corpora are not distributed.

## Pop1K7

- Official record: <https://doi.org/10.5281/zenodo.13167761>
- Official code repository: <https://github.com/Dsqvival/hierarchical-structure-analysis>
- Frozen release manifest: 1,747 sources and source-level split assignments.
- Frozen windows: 22,450 eight-bar windows.
- F/G high-confidence subset: 1,263 source songs; 1,500 evaluated test windows.

The official code repository is GPL-3.0.  The dataset record must be reviewed
separately before redistributing data-derived payloads or model weights.  This
package therefore publishes hashes, relative paths, split assignments, and key
estimates, but not MIDI or caches.

Expected layout is the official archive rooted at the directory supplied to
`--root`; `relative_source_path` in `source_manifest.jsonl` is resolved beneath
that root.  Verify every byte before preprocessing:

```bash
uv run python scripts/prepare_pop1k7.py \
  --root /path/to/pop1k7 \
  --source-manifest manifests/pop1k7/source_manifest.jsonl
```

## ComMU

- Official repository: <https://github.com/POZAlabs/ComMU-code>
- Upstream dataset license: CC BY-NC-SA 4.0, non-commercial.
- Frozen 4/4 manifest: 9,299 sources, split 8,652 / 323 / 324.

```bash
uv run python scripts/prepare_commu.py \
  --root /path/to/commu \
  --source-manifest manifests/commu/source_manifest.jsonl
```

## Pipeline contract

```text
official MIDI → non-drum notes → quantized Note/REST facts → source lineage
→ exact/transposition/rhythm-interval duplicate components → song split
→ eight-bar windows (≤2,048 facts) → representation cache
```

The split occurs before windowing.  Pop1K7's frozen contract audits exact,
transposition-invariant, and rhythm/interval components across split; all are
zero.  ComMU's preregistered contract audits source lineage, exact, and
transposition-invariant components; those crossing counts are zero.  The
released ComMU manifest also reports four rhythm/interval hash collisions, but
that diagnostic was not a split constraint and is not presented as zero.

## Public test status

`pop1k7_split_v1` is the frozen test set consumed and disclosed by this paper.
Future development must not describe it as untouched.  Use a new holdout or
call it the public benchmark split.
