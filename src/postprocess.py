"""Semantics-preserving output normalization.

Malformed predictions are intentionally not repaired here; evaluation is
responsible for marking them invalid.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def postprocess_amr(amr: str | None) -> str:
    if not amr:
        return ""
    return re.sub(r"\s+", " ", amr.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        "\n".join(postprocess_amr(line) for line in lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
