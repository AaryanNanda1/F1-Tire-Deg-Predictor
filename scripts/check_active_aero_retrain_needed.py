#!/usr/bin/env python3
"""Preflight check for the weekly Active Aero retraining workflow."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fastf1
import pandas as pd


ACTIVE_AERO_KEY = "active_aero_2026_2030"
ACTIVE_AERO_START_YEAR = 2026
ACTIVE_AERO_END_YEAR = 2030
RACE_SESSION_CODE = "R"


class PreflightError(RuntimeError):
    """Raised when the workflow cannot determine if retraining is needed."""


@dataclass
class CompletedRace:
    year: int
    round_number: int
    event_name: str
    event_date: str
    race_key: str


@dataclass
class PreflightDecision:
    should_retrain: bool
    reason: str
    as_of: str
    model_status: str
    loaded_race_count: int
    completed_race_count: int
    missing_race_keys: List[str]
    latest_completed_race: str


def load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Could not read training metadata at {path}: {exc}") from exc


def _as_date(value: Any) -> Optional[date]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _race_date_from_row(row: pd.Series) -> Optional[date]:
    for column in ("EventDate", "Session5Date", "Session5DateUtc"):
        if column in row:
            parsed = _as_date(row[column])
            if parsed is not None:
                return parsed
    return None


def _race_key(year: int, event_name: str) -> str:
    return f"{year}:{event_name}:{RACE_SESSION_CODE}"


def list_completed_races(as_of: date) -> List[CompletedRace]:
    if as_of < date(ACTIVE_AERO_START_YEAR, 1, 1):
        return []

    completed: List[CompletedRace] = []
    last_year = min(as_of.year, ACTIVE_AERO_END_YEAR)

    for year in range(ACTIVE_AERO_START_YEAR, last_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            raise PreflightError(f"FastF1 schedule lookup failed for {year}: {exc}") from exc

        if schedule is None or schedule.empty:
            raise PreflightError(f"FastF1 schedule for {year} was empty")
        if "RoundNumber" not in schedule.columns or "EventName" not in schedule.columns:
            raise PreflightError(
                f"FastF1 schedule for {year} did not include RoundNumber/EventName columns"
            )

        for _, row in schedule.iterrows():
            round_number = row.get("RoundNumber")
            event_name = row.get("EventName")
            event_date = _race_date_from_row(row)

            try:
                round_number_int = int(round_number)
            except (TypeError, ValueError):
                continue

            if (
                round_number_int <= 0
                or event_name is None
                or pd.isna(event_name)
                or event_date is None
            ):
                continue
            if event_date > as_of:
                continue

            event_name_str = str(event_name)
            completed.append(
                CompletedRace(
                    year=year,
                    round_number=round_number_int,
                    event_name=event_name_str,
                    event_date=event_date.isoformat(),
                    race_key=_race_key(year, event_name_str),
                )
            )

    completed.sort(key=lambda item: (item.year, item.round_number))
    return completed


def loaded_race_keys(active_metadata: Dict[str, Any]) -> List[str]:
    loaded_events = active_metadata.get("loaded_events", [])
    if not isinstance(loaded_events, list):
        return []

    return sorted(
        event
        for event in loaded_events
        if isinstance(event, str) and event.endswith(f":{RACE_SESSION_CODE}")
    )


def decide_retrain_needed(
    metadata: Dict[str, Any],
    completed_races: Iterable[CompletedRace],
    as_of: date,
) -> PreflightDecision:
    active_metadata = metadata.get(ACTIVE_AERO_KEY, {})
    if not isinstance(active_metadata, dict):
        active_metadata = {}

    completed = list(completed_races)
    expected_race_keys = [race.race_key for race in completed]
    loaded_keys = loaded_race_keys(active_metadata)
    loaded_key_set = set(loaded_keys)
    missing_keys = [key for key in expected_race_keys if key not in loaded_key_set]
    status = str(active_metadata.get("status", "missing"))
    latest_completed = completed[-1].race_key if completed else "none"

    if not active_metadata:
        reason = "active_aero_metadata_missing"
        should_retrain = True
    elif not status.startswith("trained"):
        reason = f"active_aero_status_{status}"
        should_retrain = True
    elif missing_keys:
        reason = "completed_race_data_missing_from_model_metadata"
        should_retrain = True
    else:
        reason = "no_new_completed_active_aero_races"
        should_retrain = False

    return PreflightDecision(
        should_retrain=should_retrain,
        reason=reason,
        as_of=as_of.isoformat(),
        model_status=status,
        loaded_race_count=len(loaded_keys),
        completed_race_count=len(expected_race_keys),
        missing_race_keys=missing_keys,
        latest_completed_race=latest_completed,
    )


def format_summary(decision: PreflightDecision) -> str:
    decision_text = "retrain" if decision.should_retrain else "skip"
    lines = [
        "## Active Aero retraining preflight",
        f"- Decision: {decision_text}",
        f"- Reason: `{decision.reason}`",
        f"- As-of date: `{decision.as_of}`",
        f"- Model status: `{decision.model_status}`",
        f"- Completed race count: `{decision.completed_race_count}`",
        f"- Loaded race count in metadata: `{decision.loaded_race_count}`",
        f"- Latest completed race: `{decision.latest_completed_race}`",
    ]

    if decision.missing_race_keys:
        lines.append("- Missing race data in metadata:")
        lines.extend(f"  - `{key}`" for key in decision.missing_race_keys[:12])
        if len(decision.missing_race_keys) > 12:
            remaining = len(decision.missing_race_keys) - 12
            lines.append(f"  - plus {remaining} more")

    return "\n".join(lines) + "\n"


def format_failure_summary(exc: Exception, as_of: date) -> str:
    attempted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return "\n".join(
        [
            "## Active Aero retraining preflight failed",
            f"- As-of date: `{as_of.isoformat()}`",
            f"- Attempted at: `{attempted_at}`",
            f"- Error: `{exc}`",
            "",
        ]
    )


def write_github_outputs(path: Optional[str], outputs: Dict[str, Any]) -> None:
    if not path:
        return

    with open(path, "a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            safe_value = str(value).replace("\n", " ").replace("\r", " ")
            handle.write(f"{key}={safe_value}\n")


def append_summary(path: Optional[str], summary: str) -> None:
    if not path:
        return

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(summary)


def write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the Active Aero model needs retraining."
    )
    parser.add_argument(
        "--metadata-path",
        default="models/era_training_metadata.json",
        help="Path to era training metadata JSON.",
    )
    parser.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Only consider races completed on or before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Optional GitHub Actions output file.",
    )
    parser.add_argument(
        "--summary-file",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Optional GitHub Actions step summary file.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON file for the preflight decision.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    as_of = date.fromisoformat(args.as_of_date)

    try:
        metadata = load_metadata(Path(args.metadata_path))
        completed_races = list_completed_races(as_of)
        decision = decide_retrain_needed(metadata, completed_races, as_of)
    except Exception as exc:
        write_github_outputs(
            args.github_output,
            {
                "preflight_status": "failed",
                "should_retrain": "false",
                "reason": "preflight_failed",
            },
        )
        summary = format_failure_summary(exc, as_of)
        append_summary(args.summary_file, summary)
        print(summary, file=sys.stderr)
        return 2

    payload = asdict(decision)
    payload["preflight_status"] = "passed"
    write_github_outputs(
        args.github_output,
        {
            "preflight_status": "passed",
            "should_retrain": str(decision.should_retrain).lower(),
            "reason": decision.reason,
            "missing_race_count": len(decision.missing_race_keys),
            "latest_completed_race": decision.latest_completed_race,
        },
    )
    write_json(args.json_output, payload)

    summary = format_summary(decision)
    append_summary(args.summary_file, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
