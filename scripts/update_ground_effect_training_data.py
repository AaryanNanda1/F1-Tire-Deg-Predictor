#!/usr/bin/env python3
"""Build a local, rate-bounded 2022-2025 Ground Effect training store."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import load_race_data
from preprocessing import preprocess_laps
from training_data_store import (
    GROUND_EFFECT_ROLE,
    PHYSICS_PRIOR_ROLE,
    ProcessedSessionStore,
    SessionSpec,
    refresh_sessions,
    session_specs_from_schedule,
)


GROUND_EFFECT_YEARS = [2022, 2023, 2024, 2025]


def _write_json(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _expected_specs() -> List[SessionSpec]:
    specs: List[SessionSpec] = []
    for year in GROUND_EFFECT_YEARS:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        specs.extend(
            session_specs_from_schedule(
                schedule,
                year=year,
                as_of=date(year, 12, 31),
                role=GROUND_EFFECT_ROLE,
            )
        )
    return specs


def _parse_years(value: str) -> List[int]:
    years = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    invalid = [year for year in years if year not in GROUND_EFFECT_YEARS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Ground Effect fetch years must be within 2022-2025: {invalid}"
        )
    return years


def _import_prior_sessions(
    target: ProcessedSessionStore,
    source_dir: str | None,
) -> List[str]:
    if not source_dir:
        return []
    source = ProcessedSessionStore(source_dir)
    source.validate(verify_hashes=True)
    imported: List[str] = []
    for key in source.session_keys(PHYSICS_PRIOR_ROLE):
        entry = source.sessions[key]
        if int(entry["year"]) not in {2024, 2025} or target.has_session(key):
            continue
        frame = pd.read_csv(source.root / entry["path"], compression="gzip")
        spec = SessionSpec(
            year=int(entry["year"]),
            round_number=int(entry["round_number"]),
            event_name=str(entry["event_name"]),
            event_date=str(entry["event_date"]),
            session_code=str(entry["session_code"]),
            role=GROUND_EFFECT_ROLE,
        )
        target.save_session(spec, frame)
        imported.append(spec.key)
    return imported


def _select_missing_specs(
    expected: Iterable[SessionSpec],
    store: ProcessedSessionStore,
    *,
    fetch_years: List[int],
    max_new_events: int,
) -> List[SessionSpec]:
    by_event: Dict[Tuple[int, int, str], List[SessionSpec]] = defaultdict(list)
    for spec in expected:
        if spec.year in fetch_years and not store.has_session(spec.key):
            by_event[(spec.year, spec.round_number, spec.event_name)].append(spec)

    events = sorted(by_event)
    if max_new_events == 0:
        events = []
    else:
        events = events[:max_new_events]
    selected: List[SessionSpec] = []
    for event in events:
        selected.extend(by_event[event])
    return selected


def _coverage(
    expected: Iterable[SessionSpec],
    store: ProcessedSessionStore,
) -> Dict[str, Any]:
    expected_specs = list(expected)
    expected_keys = [spec.key for spec in expected_specs]
    expected_races = [
        spec.key for spec in expected_specs if spec.session_code == "R"
    ]
    missing = [key for key in expected_keys if not store.has_session(key)]
    missing_races = [key for key in expected_races if not store.has_session(key)]
    return {
        "years": GROUND_EFFECT_YEARS,
        "complete": not missing,
        "expected_session_count": len(expected_keys),
        "stored_expected_session_count": len(expected_keys) - len(missing),
        "missing_session_count": len(missing),
        "missing_session_keys": missing,
        "expected_race_count": len(expected_races),
        "stored_race_count": len(expected_races) - len(missing_races),
        "missing_race_count": len(missing_races),
        "missing_race_keys": missing_races,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append a bounded number of missing 2022-2025 race weekends to the "
            "local Ground Effect training dataset. This command never trains, "
            "commits, pushes, or deploys a model."
        )
    )
    parser.add_argument(
        "--store-dir",
        default="training_data/ground_effect",
        help="Processed Ground Effect session store.",
    )
    parser.add_argument(
        "--max-new-events",
        type=int,
        default=4,
        help="Maximum missing race weekends to request in this run; 0 is coverage-only.",
    )
    parser.add_argument(
        "--fetch-years",
        type=_parse_years,
        default=GROUND_EFFECT_YEARS,
        help="Comma-separated years eligible for fetching in this run.",
    )
    parser.add_argument(
        "--import-active-prior-store",
        default=None,
        help="Import already processed 2024-2025 prior sessions from this store.",
    )
    parser.add_argument(
        "--offline-cache-only",
        action="store_true",
        help="Use existing FastF1 cache files and prohibit network downloads.",
    )
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_events < 0:
        print("--max-new-events cannot be negative", file=sys.stderr)
        return 2
    if args.offline_cache_only:
        fastf1.Cache.offline_mode(True)

    try:
        store = ProcessedSessionStore(args.store_dir)
        store.set_metadata(
            "description",
            (
                "Processed FastF1 Race, Sprint, and FP2 sessions used for "
                "reproducible 2022-2025 Ground Effect model training. Raw "
                "FastF1 response files are stored separately."
            ),
        )
        imported = _import_prior_sessions(
            store,
            args.import_active_prior_store,
        )
        expected = _expected_specs()
        selected = _select_missing_specs(
            expected,
            store,
            fetch_years=args.fetch_years,
            max_new_events=args.max_new_events,
        )
        update = refresh_sessions(
            store,
            selected,
            loader=load_race_data,
            preprocessor=preprocess_laps,
        )
        coverage = _coverage(expected, store)
        store.set_metadata("coverage", coverage)
        store.validate(verify_hashes=True)

        payload = {
            "status": "complete" if coverage["complete"] else "partial",
            "store_dir": str(args.store_dir),
            "max_new_events": args.max_new_events,
            "fetch_years": args.fetch_years,
            "selected_event_count": len(
                {
                    (spec.year, spec.round_number, spec.event_name)
                    for spec in selected
                }
            ),
            "selected_session_keys": [spec.key for spec in selected],
            "imported_session_keys": imported,
            "update": update,
            "coverage": coverage,
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "store_dir": str(args.store_dir),
            "error": str(exc),
        }
        _write_json(args.json_output, payload)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 2

    _write_json(args.json_output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
