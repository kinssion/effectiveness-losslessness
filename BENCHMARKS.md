# Frozen resource evidence

Only measurements present in released receipts are listed.  Missing values are
not estimated.

| Family | Reference GPU | Per-seed training wall time | Peak memory | Cache size | Status |
|---|---|---:|---:|---:|---|
| Pop1K7 A–I | frozen receipts do not retain a public hardware row | unavailable | clean-test metrics only | unavailable | not recoverable from copied logs |
| Pop1K7 J/K | NVIDIA GeForce RTX 4090 | unavailable in finalized receipts | selected test rows contain runtime diagnostics for 4/6 J/K runs | unavailable | partial diagnostics |
| ComMU A/D | NVIDIA GeForce RTX 4090 | 26.42–54.91 s in six run receipts | 128–211 MiB during clean test | unavailable | logged |
| Context probe | frozen receipt | unavailable | unavailable | unavailable | secondary diagnostic |

The very short ComMU values are reproduced exactly from `elapsed_seconds` in
the frozen run receipts and are not extrapolated into total GPU hours.  Total
paper GPU hours cannot be reconstructed reliably from the retained public logs.
