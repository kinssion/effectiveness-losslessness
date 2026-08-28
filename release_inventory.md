# Release inventory

## Paper-used and included

- Exact compact_v3 paper-version metadata linked to arXiv:2608.18025; the PDF
  itself is not vendored so the arXiv record remains canonical.
- Pop1K7 A–I, J/K and ComMU A/D configs, frozen protocols, model/evaluation code.
- Pop1K7 and ComMU stable source, duplicate, split, and window/subset manifests.
- Full F/G key-estimation table and K vocabulary/merge codebook.
- Validation locks, checkpoint hashes, one-shot test receipts, aggregate and
  per-seed results, J/K per-sample metrics, curves, and context-probe audit.
- Paper-ledger derivation, tables, figures, CLI, tests, CI, environment and
  mixed-license documentation.

## External to Git by design

- Raw official MIDI.
- Representation caches and tokenized corpora.
- Selected model weights and optional resume states.

Their hashes or publication status are indexed where available.
