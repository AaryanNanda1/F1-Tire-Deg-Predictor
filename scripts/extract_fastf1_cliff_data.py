#!/usr/bin/env python3
"""Extract cleaned FastF1 race stints for tire-cliff manual review."""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import sys

import fastf1
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliff_reference import (  # noqa: E402
    CliffReferenceValidationError,
    extract_session_cliff_references,
    validate_cliff_manifest,
)
from data_loader import load_race_data  # noqa: E402


def _load_manifest(path):
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_cliff_manifest(manifest)
    return manifest


def _selected_races(manifest, race_ids=None, split=None, limit=None):
    races = manifest["races"]
    if race_ids:
        requested = set(race_ids)
        available = {race["id"] for race in races}
        missing = sorted(requested - available)
        if missing:
            raise CliffReferenceValidationError(
                f"unknown race ids: {', '.join(missing)}"
            )
        races = [race for race in races if race["id"] in requested]
    if split:
        races = [race for race in races if race["split"] == split]
    return races[:limit] if limit else races


def _clean_lap_export(frame):
    columns = [
        "ReferenceId",
        "RaceId",
        "Split",
        "Driver",
        "Team",
        "Stint",
        "CompoundNormalized",
        "LapNumberNumeric",
        "TyreLifeNumeric",
        "LapTimeSeconds",
        "FuelCorrectedLapTimeSeconds",
        "TrackStatus",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Rainfall",
    ]
    return frame[[column for column in columns if column in frame]]


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _write_review_plots(clean_laps, accepted_stints, output_dir):
    """Write dependency-free SVG plots for manual cliff review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        stint["reference_id"]: stint
        for stint in accepted_stints
    }
    width, height = 820, 440
    left, right, top, bottom = 70, 30, 55, 65
    plot_width = width - left - right
    plot_height = height - top - bottom

    for reference_id, group in clean_laps.groupby("ReferenceId"):
        summary = summaries[reference_id]
        ordered = group.sort_values("TyreLifeNumeric")
        x_values = ordered["TyreLifeNumeric"].astype(float).tolist()
        y_values = ordered["FuelCorrectedLapTimeSeconds"].astype(float).tolist()
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0
        y_padding = max(0.2, (y_max - y_min) * 0.1)
        y_min -= y_padding
        y_max += y_padding

        def scale_x(value):
            return left + (value - x_min) / (x_max - x_min) * plot_width

        def scale_y(value):
            return top + (y_max - value) / (y_max - y_min) * plot_height

        points = " ".join(
            f"{scale_x(x):.1f},{scale_y(y):.1f}"
            for x, y in zip(x_values, y_values)
        )
        cliff_lap = summary.get("observed_cliff_lap")
        cliff_line = ""
        if cliff_lap is not None:
            cliff_x = scale_x(float(cliff_lap))
            cliff_line = (
                f'<line x1="{cliff_x:.1f}" y1="{top}" '
                f'x2="{cliff_x:.1f}" y2="{top + plot_height}" '
                'stroke="#ff4d4d" stroke-width="2" '
                'stroke-dasharray="6 4"/>'
                f'<text x="{cliff_x + 6:.1f}" y="{top + 16}" '
                'fill="#ff4d4d" font-size="12">'
                f"candidate cliff L{cliff_lap}</text>"
            )

        title = (
            f"{summary['season']} {summary['event_name']} · "
            f"{summary['driver']} · {summary['compound']} · "
            f"stint {summary['stint']}"
        )
        subtitle = (
            f"{summary['reference_status']} · "
            f"clean laps {summary['clean_lap_count']}/"
            f"{summary['raw_lap_count']} · "
            f"confidence {summary.get('reference_confidence') or 'n/a'}"
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#111118"/>
<text x="{left}" y="25" fill="#ffffff" font-size="16" font-family="monospace">{escape(title)}</text>
<text x="{left}" y="43" fill="#9a9aaa" font-size="11" font-family="monospace">{escape(subtitle)}</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#666"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#666"/>
<polyline points="{points}" fill="none" stroke="#4dd2ff" stroke-width="2"/>
{cliff_line}
<text x="{left + plot_width / 2:.1f}" y="{height - 20}" text-anchor="middle" fill="#bbbbc8" font-size="12">Tire age (laps)</text>
<text x="18" y="{top + plot_height / 2:.1f}" transform="rotate(-90 18 {top + plot_height / 2:.1f})" text-anchor="middle" fill="#bbbbc8" font-size="12">Fuel-corrected lap time (s)</text>
<text x="{left}" y="{top + plot_height + 20}" fill="#bbbbc8" font-size="11">L{x_min:.0f}</text>
<text x="{left + plot_width}" y="{top + plot_height + 20}" text-anchor="end" fill="#bbbbc8" font-size="11">L{x_max:.0f}</text>
<text x="{left - 8}" y="{top + 4}" text-anchor="end" fill="#bbbbc8" font-size="11">{y_max:.2f}s</text>
<text x="{left - 8}" y="{top + plot_height}" text-anchor="end" fill="#bbbbc8" font-size="11">{y_min:.2f}s</text>
</svg>
"""
        path = output_dir / f"{_safe_filename(reference_id)}.svg"
        path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "benchmarks"
            / "fastf1_cliff_reference_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--race-id",
        action="append",
        help="Extract one manifest race id; repeat for multiple races.",
    )
    parser.add_argument(
        "--split",
        choices=("calibration", "holdout"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not generate SVG manual-review plots.",
    )
    parser.add_argument(
        "--offline-cache-only",
        action="store_true",
        help="Prohibit FastF1 network downloads and use local cache files only.",
    )
    args = parser.parse_args()
    if args.offline_cache_only:
        fastf1.Cache.offline_mode(True)

    try:
        manifest = _load_manifest(args.manifest)
        races = _selected_races(
            manifest,
            race_ids=args.race_id,
            split=args.split,
            limit=args.limit,
        )
    except (OSError, json.JSONDecodeError, CliffReferenceValidationError) as exc:
        print(f"Cliff extraction failed: {exc}", file=sys.stderr)
        return 2

    if not races:
        print("Cliff extraction failed: no races selected", file=sys.stderr)
        return 2

    accepted_stints = []
    rejected_stints = []
    clean_lap_frames = []
    race_summaries = []
    failures = []
    for race in races:
        key = f"{race['season']} {race['event_name']}"
        print(f"Loading {key}...")
        try:
            session = load_race_data(
                race["season"],
                race["event_name"],
                race.get("session", "R"),
            )
            result = extract_session_cliff_references(
                session.laps,
                session.weather_data,
                race,
                cleaning_config=manifest.get("cleaning_config"),
                reference_config=manifest.get("reference_config"),
            )
        except Exception as exc:
            failures.append(
                {
                    "race_id": race["id"],
                    "message": str(exc),
                }
            )
            print(f"Failed {key}: {exc}", file=sys.stderr)
            continue

        accepted_stints.extend(result["accepted_stints"])
        rejected_stints.extend(result["rejected_stints"])
        if not result["clean_laps"].empty:
            clean_lap_frames.append(_clean_lap_export(result["clean_laps"]))
        race_summaries.append(
            {
                "race_id": race["id"],
                "total_laps": result["total_laps"],
                "accepted_laps": result["accepted_laps"],
                "accepted_stints": len(result["accepted_stints"]),
                "rejected_stints": len(result["rejected_stints"]),
                "cliff_candidates": sum(
                    stint["reference_status"]
                    == "candidate_for_manual_review"
                    for stint in result["accepted_stints"]
                ),
                "lap_rejection_counts": result["lap_rejection_counts"],
            }
        )
        print(
            f"  accepted {len(result['accepted_stints'])} stints; "
            f"rejected {len(result['rejected_stints'])}"
        )

    output = {
        "schema_version": 1,
        "suite_name": manifest.get("suite_name"),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "reference_status": (
            "candidate_for_manual_review_not_ground_truth"
        ),
        "cleaning_config": manifest.get("cleaning_config"),
        "reference_config": manifest.get("reference_config"),
        "selected_race_ids": [race["id"] for race in races],
        "race_summaries": race_summaries,
        "accepted_stints": accepted_stints,
        "rejected_stints": rejected_stints,
        "failures": failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cliff_references.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    stint_rows = [
        {**stint, "accepted": True}
        for stint in accepted_stints
    ] + [
        {**stint, "accepted": False}
        for stint in rejected_stints
    ]
    if stint_rows:
        pd.json_normalize(stint_rows).to_csv(
            args.output_dir / "stint_summary.csv",
            index=False,
        )
    if accepted_stints:
        review_columns = [
            "reference_id",
            "race_id",
            "split",
            "season",
            "event_name",
            "driver",
            "team",
            "stint",
            "compound",
            "starting_tyre_age",
            "ending_tyre_age",
            "clean_lap_count",
            "reference_status",
            "observed_cliff_lap",
            "reference_confidence",
            "manual_review_status",
            "reviewed_cliff_lap",
            "review_notes",
        ]
        pd.DataFrame(accepted_stints)[review_columns].to_csv(
            args.output_dir / "review_queue.csv",
            index=False,
        )
    if clean_lap_frames:
        clean_laps = pd.concat(clean_lap_frames, ignore_index=True)
        clean_laps.to_csv(
            args.output_dir / "clean_laps.csv",
            index=False,
        )
        if not args.skip_plots:
            _write_review_plots(
                clean_laps,
                accepted_stints,
                args.output_dir / "review_plots",
            )

    print(
        f"Wrote {len(accepted_stints)} accepted stint candidates to "
        f"{args.output_dir}"
    )
    return 2 if failures and not accepted_stints else 0


if __name__ == "__main__":
    raise SystemExit(main())
