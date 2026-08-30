#!/usr/bin/env python3
"""Apply partial manual tire-cliff decisions to a calibration review queue."""

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliff_review import (  # noqa: E402
    CliffReviewValidationError,
    apply_review_decisions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        queue = pd.read_csv(args.queue).to_dict(orient="records")
        decisions = pd.read_csv(args.decisions).to_dict(orient="records")
        updated = apply_review_decisions(queue, decisions)
    except (
        OSError,
        pd.errors.ParserError,
        CliffReviewValidationError,
    ) as exc:
        print(f"Applying review decisions failed: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(updated).to_csv(args.output, index=False)
    pending = sum(
        row["manual_review_status"] == "pending" for row in updated
    )
    print(
        f"Applied {len(decisions)} decisions to {args.output}; "
        f"{pending} reviews remain pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
