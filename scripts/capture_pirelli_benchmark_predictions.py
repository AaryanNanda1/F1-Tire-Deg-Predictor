#!/usr/bin/env python3
"""Capture current app simulation outputs for the Pirelli benchmark pilot.

Outputs from current production artifacts are diagnostic because the artifacts
may already contain data from the benchmark races.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_benchmark import load_json, validate_benchmark_suite  # noqa: E402


def _era_name(year):
    return (
        "active_aero_2026_2030"
        if int(year) >= 2026
        else "ground_effect_2022_2025"
    )


def _load_model_provenance(races, models_dir):
    metadata_path = models_dir / "era_training_metadata.json"
    if not metadata_path.exists():
        return None, None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    selected_eras = {_era_name(race["season"]) for race in races}
    artifacts = {
        name: {
            "status": values.get("status"),
            "as_of": values.get("as_of"),
            "trained_at": values.get("trained_at"),
            "loaded_event_count": len(values.get("loaded_events", [])),
        }
        for name, values in metadata.items()
        if name in selected_eras
    }
    overlap = {}
    for race in races:
        era = _era_name(race["season"])
        event_prefix = f"{race['season']}:{race['event_name']}:"
        loaded_events = metadata.get(era, {}).get("loaded_events", [])
        overlap[race["id"]] = {
            "artifact": era,
            "matched_training_events": [
                event for event in loaded_events if event.startswith(event_prefix)
            ],
        }
    return artifacts, overlap


def _payload(race):
    simulation_input = race["simulation_input"]
    return {
        "year": race["season"],
        "track_name": race["track_name"],
        "driver": simulation_input["driver"],
        "team": simulation_input["team"],
        "grid_pos": simulation_input.get("grid_pos", 1),
        "track_position": simulation_input.get(
            "track_position", simulation_input.get("grid_pos", 1)
        ),
        "race_date": simulation_input["race_date"],
        "race_time": simulation_input.get("race_time", "15:00"),
        "laps_to_complete": race["race_laps"],
        "current_lap": 0,
        "current_compound": None,
        "laps_on_current_tire": 0,
        "has_pitted": False,
        "compounds_used_count": 0,
        "sc_happened_on_tire": False,
        "sc_laps_on_tire": 0,
        "sc_currently_out": False,
        "include_strategy_diagnostics": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=ROOT / "benchmarks" / "pirelli_strategy_benchmarks.json",
    )
    parser.add_argument(
        "--split",
        choices=("calibration", "holdout"),
        default="calibration",
        help="Capture one explicit benchmark split; defaults safely to calibration.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Load model artifacts from this directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["MODEL_DIR"] = str(args.models_dir)
    from app import app

    benchmark_suite = load_json(args.benchmarks)
    validate_benchmark_suite(benchmark_suite)
    selected_races = [
        race
        for race in benchmark_suite["races"]
        if race["split"] == args.split
    ]
    if not selected_races:
        print(
            f"No benchmark races are assigned to split {args.split!r}.",
            file=sys.stderr,
        )
        return 2
    model_artifacts, benchmark_training_overlap = _load_model_provenance(
        selected_races,
        args.models_dir,
    )
    captured = {}
    failures = []
    with app.test_client() as client:
        for race in selected_races:
            response = client.post("/api/simulate", json=_payload(race))
            body = response.get_json(silent=True) or {}
            if response.status_code != 200 or body.get("status") != "success":
                failures.append(
                    {
                        "race_id": race["id"],
                        "status_code": response.status_code,
                        "message": body.get("message", "unknown simulation failure"),
                    }
                )
                continue
            captured[race["id"]] = body

    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 2

    prediction_suite = {
        "schema_version": 1,
        "benchmark_split": args.split,
        "models_dir": str(args.models_dir),
        "evaluation_mode": "production_diagnostic",
        "model_training_cutoff": None,
        "model_artifacts": model_artifacts,
        "benchmark_training_overlap": benchmark_training_overlap,
        "capture_assumptions": [
            "Historical benchmark dates use archived observed weather, not a contemporaneous pre-race forecast.",
            "Current production artifacts may contain telemetry from one or more benchmark events.",
        ],
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "races": captured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(prediction_suite, handle, indent=2)
        handle.write("\n")
    print(f"Captured {len(captured)} race predictions in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
