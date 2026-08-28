# Claim-to-artifact map

The authoritative index is `artifacts/paper_v1/result_index.json`.  Model files
are external to Git; their hashes and publication state live in
`artifacts/paper_v1/checkpoint_index.json`.

| Claim | Config | Selected checkpoints | Released result | Rebuild command |
|---|---|---:|---|---|
| A→D coordinate gain | individual `A.yaml`…`D.yaml` | 12 hashes | `pop1k7_ai_paper_ledger.json` | `make table-time` |
| F→G canonicalization | individual `F.yaml`, `G.yaml` | 6 hashes | same ledger + key manifest | `make table-tonal` |
| D/H/E/I relational controls | individual `D/E/H/I.yaml` | 12 hashes | same ledger | `make table-relation` |
| J/K carrier gain | individual `J.yaml`, `K.yaml` | 6 hashes | `pop1k7_jk_paper_ledger.json` | `make table-carrier` |
| ComMU direction | `configs/commu/A,D.yaml` | 6 hashes | `commu_ad_paper_ledger.json` | `make table-commu` |
| Context dependence | `configs/context_probe/context_relation.yaml` | 1 hash | intervention audit | `make reproduce-paper` |

Paths in the table are shorthand for individual YAML files.  Exact arrays are
listed in the JSON index rather than duplicated here.

## Frozen evidence chain

Each formal family is bound by the released protocol, stable data manifest,
validation lock, checkpoint hashes, clean-test receipt, and result JSON.  J/K
add a cache audit, technical-continuation state, and train-only BPE receipt.
Copied review receipts and three source manifests were redacted only for local
paths and infrastructure identities.  Their `redaction_manifest.json` files
record pre/post hashes and affected JSON pointers.  A regular `.sha256` sidecar
checks the review copy; `.original.sha256` preserves the frozen unredacted hash.

The derived paper ledgers contain no discovered checkpoints or hard-coded
measurements.  `scripts/build_paper_ledger.py` reads per-seed frozen results and
applies only the declared denominator and side-information rules.
