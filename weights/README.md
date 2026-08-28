# External checkpoint bundle

Model weights are intentionally absent from Git history.  The 40 selected
checkpoint hashes, expected filenames, configs, seeds, license status, and
future download URLs are in `artifacts/paper_v1/checkpoint_index.json`.

After upstream-license review, publish one immutable archive separately, fill
the URLs in that index, and run:

```bash
uv run python scripts/download_artifacts.py --output weights/downloaded
```

The downloader deletes a file whose SHA-256 does not match the frozen index.
Do not commit `weights/downloaded/`.
