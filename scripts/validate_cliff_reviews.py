#!/usr/bin/env python3
"""Validate and freeze manually reviewed tire-cliff calibration labels."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliff_review import (  # noqa: E402
    CliffReviewValidationError,
    validate_review_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Validate a work-in-progress queue without marking it frozen.",
    )
    args = parser.parse_args()

    try:
        references = json.loads(args.references.read_text(encoding="utf-8"))
        source_stints = {
            stint["reference_id"]: stint
            for stint in references["accepted_stints"]
        }
        rows = pd.read_csv(args.reviews).to_dict(orient="records")
        labels = validate_review_rows(
            rows,
            source_stints=source_stints,
            require_complete=not args.allow_pending,
        )
    except (
        OSError,
        KeyError,
        json.JSONDecodeError,
        pd.errors.ParserError,
        CliffReviewValidationError,
    ) as exc:
        print(f"Review validation failed: {exc}", file=sys.stderr)
        return 2

    pending_count = sum(
        label["manual_review_status"] == "pending" for label in labels
    )
    output = {
        "schema_version": 1,
        "artifact_type": "reviewed_tire_cliff_calibration_labels",
        "status": "work_in_progress" if pending_count else "frozen",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_generated_at": references.get("generated_at"),
        "source_selected_race_ids": references.get("selected_race_ids"),
        "cleaning_config": references.get("cleaning_config"),
        "reference_config": references.get("reference_config"),
        "label_count": len(labels),
        "pending_count": pending_count,
        "labels": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated {len(labels)} calibration labels "
        f"({pending_count} pending); status={output['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
