#!/usr/bin/env python3
"""Create a balanced, calibration-only tire-cliff manual-review queue."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliff_review import (  # noqa: E402
    CliffReviewValidationError,
    build_review_queue,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--medium-limit", type=int, default=24)
    parser.add_argument("--no-cliff-limit", type=int, default=36)
    args = parser.parse_args()

    try:
        references = json.loads(args.references.read_text(encoding="utf-8"))
        queue = build_review_queue(
            references,
            medium_limit=args.medium_limit,
            no_cliff_limit=args.no_cliff_limit,
        )
    except (
        OSError,
        json.JSONDecodeError,
        CliffReviewValidationError,
    ) as exc:
        print(f"Review prioritization failed: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(queue).to_csv(args.output, index=False)
    counts = pd.Series(
        [row["review_priority_reason"] for row in queue]
    ).value_counts()
    print(f"Wrote {len(queue)} calibration reviews to {args.output}")
    for reason, count in counts.items():
        print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
