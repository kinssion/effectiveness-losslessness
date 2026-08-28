from __future__ import annotations

import argparse
import json
from pathlib import Path

from el_tokenization.reporting.build_figures import build_figures
from el_tokenization.reporting.build_tables import build_tables
from el_tokenization.reporting.verify_paper_numbers import verify_paper_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/paper_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("paper/figure_data"))
    args = parser.parse_args()
    verification = verify_paper_artifacts(args.artifact_root)
    if verification["status"] != "PASS":
        raise SystemExit(json.dumps(verification, indent=2))
    result = {
        "verification": verification,
        "tables": build_tables(args.artifact_root, args.output_root),
        "figures": build_figures(args.artifact_root, args.output_root),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
