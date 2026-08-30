#!/usr/bin/env python3
"""Capture model curves matching frozen FastF1 cliff calibration labels.

The command is deliberately calibration-only. It never downloads FastF1 data,
trains a model, or reads the holdout manifest.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliff_review import validate_review_rows  # noqa: E402
from degradation_engine import TireDegradationSimulator  # noqa: E402
from mappings import (  # noqa: E402
    EVENT_NAME_TO_CIRCUIT,
    get_track_features,
    get_track_info,
    normalize_team_name,
)


ERA_NAME = "ground_effect_2022_2025"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_clean_laps(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["ReferenceId"]].append(
                {
                    "tire_age": int(float(row["TyreLifeNumeric"])),
                    "fuel_corrected_lap_time_seconds": float(
                        row["FuelCorrectedLapTimeSeconds"]
                    ),
                    "humidity_percent": (
                        float(row["Humidity"])
                        if row.get("Humidity") not in (None, "")
                        else None
                    ),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda value: value["tire_age"])
    return dict(grouped)


def _model_provenance(models_dir: Path) -> dict:
    model_path = models_dir / f"{ERA_NAME}_model.joblib"
    features_path = models_dir / f"{ERA_NAME}_features.joblib"
    degradation_model_path = (
        models_dir / f"{ERA_NAME}_degradation_model.joblib"
    )
    degradation_features_path = (
        models_dir / f"{ERA_NAME}_degradation_features.joblib"
    )
    metadata_path = models_dir / "era_training_metadata.json"
    missing = [
        str(path)
        for path in (model_path, features_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing isolated Ground Effect artifact files: "
            + ", ".join(missing)
        )
    metadata = _load_json(metadata_path)
    era_metadata = metadata.get(ERA_NAME)
    if not isinstance(era_metadata, dict):
        raise ValueError(
            f"{metadata_path} does not contain {ERA_NAME!r} provenance"
        )
    provenance = {
        "era": ERA_NAME,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "features_path": str(features_path),
        "features_sha256": _sha256(features_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "as_of": era_metadata.get("as_of"),
        "trained_at": era_metadata.get("trained_at"),
        "coverage": era_metadata.get("coverage"),
        "loaded_events": era_metadata.get("loaded_events", []),
    }
    if degradation_model_path.exists() != degradation_features_path.exists():
        raise ValueError(
            "Dedicated degradation model and feature artifacts must either "
            "both exist or both be absent"
        )
    if degradation_model_path.exists():
        provenance.update(
            {
                "degradation_model_path": str(degradation_model_path),
                "degradation_model_sha256": _sha256(
                    degradation_model_path
                ),
                "degradation_features_path": str(
                    degradation_features_path
                ),
                "degradation_features_sha256": _sha256(
                    degradation_features_path
                ),
                "degradation_target": era_metadata.get(
                    "degradation_target"
                ),
            }
        )
    return provenance


def _weather(label: dict, clean_laps: list[dict]) -> dict:
    humidities = [
        row["humidity_percent"]
        for row in clean_laps
        if row["humidity_percent"] is not None
    ]
    return {
        "air_temp": float(label["mean_air_temp_c"]),
        "track_temp": float(label["mean_track_temp_c"]),
        "humidity": round(mean(humidities), 3) if humidities else 50.0,
        "rainfall": False,
        "wind_speed": 10.0,
        "source": "fastf1_clean_stint_average",
        "fallback_fields": (
            ["wind_speed"]
            if humidities
            else ["humidity", "wind_speed"]
        ),
    }


def _training_overlap(label: dict, loaded_events: list[str]) -> list[str]:
    prefix = f"{label['season']}:{label['event_name']}:"
    return [event for event in loaded_events if event.startswith(prefix)]


def _capture_label(
    label: dict,
    clean_laps: list[dict],
    simulator: TireDegradationSimulator,
    loaded_events: list[str],
) -> dict:
    event_name = label["event_name"]
    track_name = EVENT_NAME_TO_CIRCUIT.get(event_name)
    if not track_name:
        raise ValueError(f"No circuit mapping for {event_name!r}")

    track_info = get_track_info(track_name)
    track_features = get_track_features(track_name)
    weather = _weather(label, clean_laps)
    ending_age = int(label["ending_tyre_age"])
    starting_age = int(label["starting_tyre_age"])
    compound = str(label["compound"]).upper()

    drop_off, absolute_times, raw_drop_off = simulator._simulate_compound(
        label["driver"],
        normalize_team_name(label["team"]),
        track_name,
        track_info["type"],
        track_info.get("length_km", 5.0),
        compound,
        weather,
        track_features,
        track_info.get("race_laps", 57),
        max_laps=ending_age,
        return_raw=True,
    )
    analysis = simulator._analyze_curve(
        drop_off,
        absolute_times,
        track_name=track_name,
    )
    full_predicted_curve = [
        {
            "tire_age": age,
            "predicted_lap_time_seconds": absolute_times[age - 1],
            "predicted_drop_off_seconds": round(drop_off[age - 1], 6),
            "raw_predicted_drop_off_seconds": round(
                raw_drop_off[age - 1],
                6,
            ),
        }
        for age in range(1, ending_age + 1)
    ]
    predicted_curve = [
        point
        for point in full_predicted_curve
        if starting_age <= point["tire_age"] <= ending_age
    ]
    return {
        "reference_id": label["reference_id"],
        "race_id": label["race_id"],
        "split": "calibration",
        "season": label["season"],
        "event_name": event_name,
        "track_name": track_name,
        "track_type": track_info["type"],
        "driver": label["driver"],
        "team": label["team"],
        "stint": label["stint"],
        "compound": compound,
        "starting_tire_age": starting_age,
        "ending_tire_age": ending_age,
        "manual_review_status": label["manual_review_status"],
        "reviewed_cliff_lap": label.get("reviewed_cliff_lap"),
        "weather_context": weather,
        "observed_curve": [
            {
                "tire_age": row["tire_age"],
                "fuel_corrected_lap_time_seconds": round(
                    row["fuel_corrected_lap_time_seconds"],
                    6,
                ),
            }
            for row in clean_laps
            if starting_age <= row["tire_age"] <= ending_age
        ],
        "predicted_curve": predicted_curve,
        "full_predicted_curve": full_predicted_curve,
        "predicted_performance_cliff_lap": analysis[
            "performance_cliff_lap"
        ],
        "predicted_cliff_confidence": analysis["cliff_confidence"],
        "predicted_cliff_method": analysis["cliff_method"],
        "predicted_cliff_reason": analysis["cliff_reason"],
        "predicted_strategy_useful_life_lap": analysis[
            "strategy_useful_life_lap"
        ],
        "training_overlap": _training_overlap(label, loaded_events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews",
        type=Path,
        default=(
            ROOT
            / "calibration_work"
            / "fastf1_cliff_calibration"
            / "reviewed_labels.json"
        ),
    )
    parser.add_argument(
        "--clean-laps",
        type=Path,
        default=(
            ROOT
            / "calibration_work"
            / "fastf1_cliff_calibration"
            / "clean_laps.csv"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "calibration_work" / "baseline_models",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review_artifact = _load_json(args.reviews)
    if review_artifact.get("status") != "frozen":
        raise ValueError("review artifact must be frozen before capture")
    labels = validate_review_rows(
        review_artifact.get("labels", []),
        require_complete=True,
    )
    if any(label.get("split") != "calibration" for label in labels):
        raise ValueError("capture_cliff_predictions is calibration-only")

    clean_laps_by_reference = _load_clean_laps(args.clean_laps)
    provenance = _model_provenance(args.models_dir)
    simulators: dict[int, TireDegradationSimulator] = {}
    captures = []
    skipped = []
    for label in labels:
        reference_id = label["reference_id"]
        if label["manual_review_status"] == "rejected":
            skipped.append(
                {
                    "reference_id": reference_id,
                    "reason": "manual_review_rejected",
                }
            )
            continue
        clean_laps = clean_laps_by_reference.get(reference_id)
        if not clean_laps:
            raise ValueError(f"{reference_id}: no matching clean laps")
        year = int(label["season"])
        if year not in simulators:
            simulators[year] = TireDegradationSimulator(
                year,
                models_dir=str(args.models_dir),
                force_jit_check=False,
            )
        simulator = simulators[year]
        captures.append(
            _capture_label(
                label,
                clean_laps,
                simulator,
                provenance["loaded_events"],
            )
        )

    output = {
        "schema_version": 1,
        "artifact_type": "tire_cliff_calibration_predictions",
        "split": "calibration",
        "captured_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "review_artifact": str(args.reviews),
        "review_artifact_sha256": _sha256(args.reviews),
        "clean_laps_artifact": str(args.clean_laps),
        "clean_laps_sha256": _sha256(args.clean_laps),
        "model_provenance": provenance,
        "capture_count": len(captures),
        "skipped_count": len(skipped),
        "capture_assumptions": [
            "FastF1 clean-stint mean air, track, and humidity values are used.",
            "Wind speed defaults to 10 km/h because it is absent from the frozen clean-lap artifact.",
            "Rejected manual reviews are excluded.",
            "Matching predicted_curve values are limited to the reviewed tire-age interval; full_predicted_curve preserves ages 1 through the stint endpoint for detector evaluation.",
            "Each predicted point preserves both the raw degradation-model drop-off and the monotonic drop-off used by the UI and strategy engine.",
            "Training overlap is reported per race and is not hidden.",
        ],
        "captures": captures,
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(
        f"Captured {len(captures)} calibration curves; "
        f"skipped {len(skipped)} rejected labels; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
