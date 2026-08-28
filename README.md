# Effectiveness and Losslessness in Music Tokenization

[![arXiv](https://img.shields.io/badge/arXiv-2608.18025-b31b1b.svg)](https://arxiv.org/abs/2608.18025)

Reproducibility package for **How Far Should Tokenization Go? Predictive
Effectiveness and Relational Losslessness**, by [Yi Wang](https://orcid.org/0009-0004-6057-8151).
The package studies which invertible representation operations reduce
predictive code length without confusing coordinate gain, relational excess,
and carrier gain.

- **Paper:** [arXiv:2608.18025](https://arxiv.org/abs/2608.18025)
- **Repository:** [github.com/kinssion/effectiveness-losslessness](https://github.com/kinssion/effectiveness-losslessness)

**Version status:** this repository tracks the `m4l_iclr2027_compact_v3`
content freeze. Until the arXiv replacement is visible, the stable arXiv record
may still display the earlier title and manuscript version.

## Frozen results

All values are base-2 predictive bits per declared Note/REST fact.  EOS stays in
the numerator; lower is better.  Values are mean ± sample SD over three frozen
seeds.

| Corpus / operation | Baseline | Intervention | Frozen result |
|---|---:|---:|---:|
| Pop1K7 time coordinates | A 7.11618 | D 6.26809 | −0.84809 |
| Pop1K7 relational pitch | D 6.26809 | H 6.23638 | −0.03171 |
| Pop1K7 tonal canonicalization | F 6.48686 | G 6.39631 | −0.09055 |
| Pop1K7 carrier | J 5.96305 | K 6.77367 | +0.81061 |
| ComMU time coordinates | A 12.33993 | D 12.22855 | −0.11139 |

G includes the four-bit inverse-shift code for every independently
canonicalized window.  J/K are a matched recoverable-field carrier comparison;
their serialized target counts are consequences of the intervention.

## Install and verify

Python 3.11 or 3.12 is supported.

```bash
uv sync --frozen --extra dev
make smoke
make verify-paper-artifacts
make reproduce-paper
```

Without `make`, use:

```bash
uv run el-token smoke --steps 100
uv run el-token paper verify
uv run python scripts/reproduce_paper.py --artifact-root artifacts/paper_v1
```

The first two levels need no copyrighted dataset or checkpoint.  See
`REPRODUCING.md` for checkpoint evaluation and retraining.

## Artifact status

- Code, configs, stable manifests, result JSON, receipts, curves, tokenizer
  codebook, and an exact paper-version pointer are included.
- Raw MIDI, representation caches, generated music, and model weights are not
  included.
- Selected-checkpoint hashes are indexed.  Download URLs remain `null` until
  upstream-license review and separate weight hosting are complete.
- The secondary context probe's frozen checkpoint and intervention audit are
  present, but its historical held-out window manifest was not recovered.  It
  is therefore a diagnostic artifact, not a bit-exact data-rebuild claim.
- Historical evaluators included EOS in their model numerator but did not log
  EOS NLL as a separate scalar.  The released ledger preserves that fact and
  never fabricates an EOS decomposition.

## Data boundary

Users obtain Pop1K7 and ComMU from their official sources.  This repository
redistributes neither raw MIDI nor tokenized corpora.  `pop1k7_split_v1` is the
frozen test split already consumed and disclosed by this paper; future method
development must call it a public benchmark split or create a new holdout.

Licensing and upstream restrictions are documented in `DATA.md`,
`THIRD_PARTY.md`, `NOTICE`, and `MODEL_CARD.md`.
