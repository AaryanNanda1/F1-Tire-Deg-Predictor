#!/usr/bin/env python3
"""Evaluate captured model outputs against the curated Pirelli benchmark."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_benchmark import (  # noqa: E402
    BenchmarkValidationError,
    evaluate_suite,
    load_json,
    write_report_csv,
    write_report_json,
    write_report_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=ROOT / "benchmarks" / "pirelli_strategy_benchmarks.json",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("calibration", "holdout"),
        default="calibration",
        help="Evaluate one explicit benchmark split; defaults safely to calibration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write report.json, race_metrics.csv, and summary.md to this directory.",
    )
    args = parser.parse_args()

    try:
        benchmarks = load_json(args.benchmarks)
        benchmarks["races"] = [
            race
            for race in benchmarks["races"]
            if race.get("split") == args.split
        ]
        if not benchmarks["races"]:
            raise BenchmarkValidationError(
                f"no benchmark races are assigned to split {args.split!r}"
            )
        report = evaluate_suite(benchmarks, load_json(args.predictions))
    except (OSError, json.JSONDecodeError, BenchmarkValidationError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 2

    if args.output_dir:
        write_report_json(report, args.output_dir / "report.json")
        write_report_csv(report, args.output_dir / "race_metrics.csv")
        write_report_markdown(report, args.output_dir / "summary.md")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
