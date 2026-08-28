# Reproducing the paper

The package separates evidence verification, executable synthetic checks,
checkpoint evaluation, and retraining.  A passing lower level does not imply a
higher level was run.

## Level 0 — released evidence, no data

```bash
uv sync --frozen --extra dev
uv run el-token paper verify
uv run python scripts/reproduce_paper.py --artifact-root artifacts/paper_v1
```

This verifies indexed hashes, receipt fields, paper-ledger arithmetic, frozen
means, paired deltas, reconstructed tables, and SVG figures.  Generated files
go to `paper/figure_data/`.

Accounting is explicit:

```text
paper bits / declared fact
= (model target bits including EOS + required side bits)
  / declared Note/REST count
```

For J/K the multiplier is `244248 / 242269`.  For G, four bits are charged per
one of 1,500 independent high-confidence windows.

## Level 1 — CPU synthetic smoke

```bash
uv run pytest
uv run el-token smoke --steps 100
```

The fixture is programmatically authored and contains no upstream music.  Tests
cover exact Note/REST round trips, causal same-onset ordering, duration and time
coordinates, pairwise bias determinism, pitch factorization, canonicalization,
J serialization, train-only reversible BPE, EOS/side accounting, split audits,
test-firewall behavior, checkpoint/config hashes, and parameter counts.

## Level 2 — released checkpoint evaluation

1. Obtain the official dataset and verify it:

   ```bash
   uv run python scripts/prepare_pop1k7.py \
     --root /path/to/official/pop1k7 \
     --source-manifest manifests/pop1k7/source_manifest.jsonl
   ```

2. List the selected checkpoints and their publication state:

   ```bash
   uv run python scripts/download_artifacts.py --list
   ```

3. After URLs are filled in by the public release, download and hash-check the
   separate bundle, then run the evaluator:

   ```bash
   uv run python scripts/download_artifacts.py --output weights/downloaded
   uv run python scripts/evaluate_checkpoint.py \
     --checkpoint weights/downloaded/pop1k7_D_seed20260819.safetensors \
     --config configs/pop1k7/D.yaml \
     --seed 20260819 --split test \
     --validation-lock artifacts/paper_v1/validation_locks/pop1k7_ai_validation_lock.json
   ```

This release cannot currently complete this level because checkpoint
URLs are intentionally unpublished pending upstream-license review.  The
scripts fail closed on a missing URL, wrong hash, wrong seed/arm, or absent
validation lock.

## Level 3 — full retraining

Paper configs and frozen protocols are in `configs/`.  All arms instantiate via
the same representation factory:

```bash
uv run el-token show-representation --config configs/pop1k7/D.yaml
uv run python scripts/train_arm.py --config configs/pop1k7/D.yaml \
  --seed 20260819 --prepared-root /path/to/prepared/cache --output runs/D_20260819
```

The release retains the checkpoint-compatible model, cache, exposure,
selection, and audit modules, but the clean-room public training adapter is
marked experimental until a Level-2 checkpoint evaluation has passed on a
fresh official-data rebuild.  It will not silently substitute synthetic
training for a paper run.  CI uses `--synthetic --steps 100` only.

Run one seed first, then a family, then all formal arms.  Never use the public
test split for checkpoint selection.  `sealed-test` reproduces the paper's
audit protocol; it is not an enforceable security boundary for third parties.

## Reference environment and resource logs

See `environment-reference.txt` and `BENCHMARKS.md`.  Only values found in
frozen receipts are reported.  Missing historical wall-time or cache-size
measurements remain explicitly unavailable instead of being estimated.
