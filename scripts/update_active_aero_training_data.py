#!/usr/bin/env python3
"""Append newly completed FastF1 sessions to the persistent training-data store."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import fastf1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import load_race_data
from preprocessing import preprocess_laps
from training_data_store import (
    ACTIVE_AERO_ROLE,
    PHYSICS_PRIOR_ROLE,
    ProcessedSessionStore,
    TrainingDataStoreError,
    refresh_sessions,
    session_specs_from_schedule,
)


def _specs_for_years(
    years: List[int],
    *,
    as_of: date,
    role: str,
) -> List:
    specs = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        specs.extend(
            session_specs_from_schedule(
                schedule,
                year=year,
                as_of=as_of,
                role=role,
            )
        )
    return specs


def _write_json(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Store processed Active Aero sessions and fetch only sessions that "
            "are missing from the repository dataset."
        )
    )
    parser.add_argument(
        "--store-dir",
        default="training_data/active_aero",
        help="Processed session store committed to the repository.",
    )
    parser.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Only include sessions completed on or before this date.",
    )
    parser.add_argument(
        "--bootstrap-prior",
        action="store_true",
        help="Also seed the frozen 2024-2025 physics-prior sessions.",
    )
    parser.add_argument(
        "--offline-cache-only",
        action="store_true",
        help="Use existing FastF1 cache files and prohibit network downloads.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional machine-readable update report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    as_of = date.fromisoformat(args.as_of_date)
    store = ProcessedSessionStore(args.store_dir)

    if args.offline_cache_only:
        fastf1.Cache.offline_mode(True)

    try:
        active_years = list(range(2026, min(as_of.year, 2030) + 1))
        active_specs = _specs_for_years(
            active_years,
            as_of=as_of,
            role=ACTIVE_AERO_ROLE,
        )
        active_result = refresh_sessions(
            store,
            active_specs,
            loader=load_race_data,
            preprocessor=preprocess_laps,
        )

        prior_result: Dict[str, Any] = {
            "added": [],
            "skipped": [],
            "failures": [],
            "mandatory_failures": [],
        }
        if args.bootstrap_prior:
            prior_specs = _specs_for_years(
                [2024, 2025],
                as_of=date(2025, 12, 31),
                role=PHYSICS_PRIOR_ROLE,
            )
            prior_result = refresh_sessions(
                store,
                prior_specs,
                loader=load_race_data,
                preprocessor=preprocess_laps,
            )
        elif not store.session_keys(PHYSICS_PRIOR_ROLE):
            raise TrainingDataStoreError(
                "The persistent physics-prior dataset is missing. Bootstrap it once "
                "with --bootstrap-prior before enabling weekly retraining."
            )

        store.validate(verify_hashes=True)
        payload = {
            "status": "failed"
            if active_result["mandatory_failures"]
            or prior_result["mandatory_failures"]
            else "ok",
            "as_of": as_of.isoformat(),
            "store_dir": str(args.store_dir),
            "active_aero": active_result,
            "physics_prior": prior_result,
            "stored_active_sessions": len(store.session_keys(ACTIVE_AERO_ROLE)),
            "stored_prior_sessions": len(store.session_keys(PHYSICS_PRIOR_ROLE)),
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "as_of": as_of.isoformat(),
            "store_dir": str(args.store_dir),
            "error": str(exc),
        }
        _write_json(args.json_output, payload)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 2

    _write_json(args.json_output, payload)
    print(json.dumps(payload, indent=2))
    return 2 if payload["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
