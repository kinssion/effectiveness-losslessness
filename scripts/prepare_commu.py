from __future__ import annotations

import sys

from el_tokenization.cli import main

if __name__ == "__main__":
    main(["data", "prepare", "commu", *sys.argv[1:]])
