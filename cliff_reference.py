"""FastF1 stint cleaning and observed tire-cliff reference construction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
REQUIRED_LAP_COLUMNS = {
    "Driver",
    "Stint",
    "Compound",
    "TyreLife",
    "LapNumber",
    "LapTime",
    "Time",
    "TrackStatus",
    "PitInTime",
    "PitOutTime",
    "IsAccurate",
}

DEFAULT_CLEANING_CONFIG = {
    "minimum_clean_laps": 8,
    "maximum_removed_fraction": 0.4,
    "fuel_burn_sec_per_race_lap": 0.07,
    "outlier_minimum_slow_sec": 6.0,
    "outlier_minimum_fast_sec": 3.0,
    "outlier_mad_multiplier": 4.0,
}

DEFAULT_REFERENCE_CONFIG = {
    "minimum_segment_laps": 5,
    "minimum_pre_cliff_slope_sec_per_lap": -0.05,
    "minimum_post_cliff_slope_sec_per_lap": 0.08,
    "minimum_slope_increase_sec_per_lap": 0.08,
    "minimum_fit_improvement_ratio": 0.2,
}


class CliffReferenceValidationError(ValueError):
    """Raised when a manifest or FastF1 frame cannot support extraction."""


def validate_cliff_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise CliffReferenceValidationError("manifest schema_version must be 1")
    races = manifest.get("races")
    if not isinstance(races, list) or not races:
        raise CliffReferenceValidationError("manifest races must be non-empty")

    seen_ids = set()
    for index, race in enumerate(races):
        context = f"races[{index}]"
        race_id = race.get("id")
        if not isinstance(race_id, str) or not race_id:
            raise CliffReferenceValidationError(f"{context}.id is required")
        if race_id in seen_ids:
            raise CliffReferenceValidationError(f"duplicate race id: {race_id}")
        seen_ids.add(race_id)
        if race.get("split") not in {"calibration", "holdout"}:
            raise CliffReferenceValidationError(
                f"{context}.split must be calibration or holdout"
            )
        if not isinstance(race.get("season"), int):
            raise CliffReferenceValidationError(
                f"{context}.season must be an integer"
            )
        if not isinstance(race.get("event_name"), str):
            raise CliffReferenceValidationError(
                f"{context}.event_name is required"
            )
        if race.get("session", "R") != "R":
            raise CliffReferenceValidationError(
                f"{context}.session must be R for the first reference suite"
            )


def _lap_time_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _merge_weather(
    laps: pd.DataFrame,
    weather: Optional[pd.DataFrame],
) -> pd.DataFrame:
    output = laps.copy()
    weather_columns = ("Rainfall", "AirTemp", "TrackTemp", "Humidity")
    if (
        weather is None
        or weather.empty
        or "Time" not in output.columns
        or "Time" not in weather.columns
        or "Rainfall" not in weather.columns
    ):
        output["WeatherAvailable"] = False
        for column in weather_columns:
            if column not in output.columns:
                output[column] = False if column == "Rainfall" else np.nan
        return output

    available = [column for column in weather_columns if column in weather]
    weather_frame = weather[["Time", *available]].sort_values("Time")
    output = pd.merge_asof(
        output.sort_values("Time"),
        weather_frame,
        on="Time",
        direction="backward",
        suffixes=("", "_weather"),
    )
    for column in weather_columns:
        weather_column = f"{column}_weather"
        if weather_column in output:
            output[column] = output[weather_column]
            output.drop(columns=[weather_column], inplace=True)
        elif column not in output:
            output[column] = False if column == "Rainfall" else np.nan
    output["WeatherAvailable"] = True
    return output


def _is_true(value: Any) -> bool:
    return False if pd.isna(value) else bool(value)


def _base_rejection_reasons(row: pd.Series) -> list[str]:
    reasons = []
    if not np.isfinite(row["LapTimeSeconds"]) or row["LapTimeSeconds"] <= 0:
        reasons.append("invalid_lap_time")
    if not np.isfinite(row["TyreLifeNumeric"]) or row["TyreLifeNumeric"] <= 0:
        reasons.append("invalid_tyre_life")
    if not np.isfinite(row["LapNumberNumeric"]):
        reasons.append("invalid_lap_number")
    if row["CompoundNormalized"] not in DRY_COMPOUNDS:
        reasons.append("non_dry_compound")
    if "IsAccurate" in row and not _is_true(row["IsAccurate"]):
        reasons.append("inaccurate_lap")
    if "Deleted" in row and _is_true(row["Deleted"]):
        reasons.append("deleted_lap")
    if "TrackStatus" in row and str(row["TrackStatus"]) != "1":
        reasons.append("non_green_track_status")
    if "PitInTime" in row and pd.notna(row["PitInTime"]):
        reasons.append("pit_in_lap")
    if "PitOutTime" in row and pd.notna(row["PitOutTime"]):
        reasons.append("pit_out_lap")
    if _is_true(row.get("Rainfall", False)):
        reasons.append("rain_affected")
    if not _is_true(row.get("WeatherAvailable", False)):
        reasons.append("weather_unavailable")
    return reasons


def _add_outlier_reasons(
    frame: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
) -> None:
    eligible = frame[frame["RejectionReasons"].map(len) == 0]
    group_columns = ["Driver", "Stint", "CompoundNormalized"]
    for _, group in eligible.groupby(group_columns, dropna=False):
        values = group["FuelCorrectedLapTimeSeconds"]
        median = float(values.median())
        mad = float(np.median(np.abs(values - median)))
        robust_scale = 1.4826 * mad
        slow_limit = max(
            float(cleaning_config["outlier_minimum_slow_sec"]),
            float(cleaning_config["outlier_mad_multiplier"]) * robust_scale,
        )
        fast_limit = max(
            float(cleaning_config["outlier_minimum_fast_sec"]),
            float(cleaning_config["outlier_mad_multiplier"]) * robust_scale,
        )
        for row_index, value in values.items():
            if value > median + slow_limit:
                frame.at[row_index, "RejectionReasons"].append(
                    "slow_pace_outlier"
                )
            elif value < median - fast_limit:
                frame.at[row_index, "RejectionReasons"].append(
                    "fast_pace_outlier"
                )


def prepare_fastf1_laps(
    laps: pd.DataFrame,
    weather: Optional[pd.DataFrame] = None,
    cleaning_config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Annotate FastF1 laps with fuel correction and conservative exclusions."""
    cfg = {**DEFAULT_CLEANING_CONFIG, **(cleaning_config or {})}
    frame = pd.DataFrame(laps).copy()
    missing = sorted(REQUIRED_LAP_COLUMNS - set(frame.columns))
    if missing:
        raise CliffReferenceValidationError(
            f"FastF1 laps missing required columns: {', '.join(missing)}"
        )

    frame["SourceRow"] = np.arange(len(frame))
    frame["LapTimeSeconds"] = _lap_time_seconds(frame["LapTime"])
    frame["TyreLifeNumeric"] = pd.to_numeric(
        frame["TyreLife"], errors="coerce"
    )
    frame["LapNumberNumeric"] = pd.to_numeric(
        frame["LapNumber"], errors="coerce"
    )
    frame["CompoundNormalized"] = (
        frame["Compound"].astype("string").str.upper()
    )
    frame = _merge_weather(frame, weather)
    frame["FuelCorrectedLapTimeSeconds"] = (
        frame["LapTimeSeconds"]
        + (
            frame["LapNumberNumeric"] - 1.0
        ) * float(cfg["fuel_burn_sec_per_race_lap"])
    )
    frame["RejectionReasons"] = frame.apply(
        _base_rejection_reasons,
        axis=1,
    )
    _add_outlier_reasons(frame, cfg)
    frame["AcceptedLap"] = frame["RejectionReasons"].map(len) == 0
    return frame.sort_values(["Driver", "Stint", "TyreLifeNumeric"])


def construct_observed_cliff_reference(
    cleaned_stint: pd.DataFrame,
    reference_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Construct a review candidate using piecewise linear fuel-corrected pace."""
    cfg = {**DEFAULT_REFERENCE_CONFIG, **(reference_config or {})}
    stint = cleaned_stint.sort_values("TyreLifeNumeric")
    x = stint["TyreLifeNumeric"].to_numpy(dtype=float)
    y = stint["FuelCorrectedLapTimeSeconds"].to_numpy(dtype=float)
    min_segment = int(cfg["minimum_segment_laps"])

    no_cliff = {
        "reference_status": "no_cliff_candidate",
        "observed_cliff_lap": None,
        "reference_confidence": None,
        "pre_cliff_slope_sec_per_lap": None,
        "post_cliff_slope_sec_per_lap": None,
        "slope_increase_sec_per_lap": None,
        "fit_improvement_ratio": None,
    }
    if len(stint) < min_segment * 2:
        return {**no_cliff, "reference_reason": "insufficient segment data"}

    single_fit = np.polyfit(x, y, 1)
    single_sse = float(np.sum((y - np.polyval(single_fit, x)) ** 2))
    best = None
    for breakpoint in range(min_segment, len(stint) - min_segment + 1):
        pre_x, post_x = x[:breakpoint], x[breakpoint:]
        pre_y, post_y = y[:breakpoint], y[breakpoint:]
        pre_fit = np.polyfit(pre_x, pre_y, 1)
        post_fit = np.polyfit(post_x, post_y, 1)
        pre_slope = float(pre_fit[0])
        post_slope = float(post_fit[0])
        slope_increase = post_slope - pre_slope
        if (
            pre_slope
            < float(cfg["minimum_pre_cliff_slope_sec_per_lap"])
            or post_slope
            < float(cfg["minimum_post_cliff_slope_sec_per_lap"])
            or slope_increase
            < float(cfg["minimum_slope_increase_sec_per_lap"])
        ):
            continue
        combined_sse = float(
            np.sum((pre_y - np.polyval(pre_fit, pre_x)) ** 2)
            + np.sum((post_y - np.polyval(post_fit, post_x)) ** 2)
        )
        if best is None or combined_sse < best["combined_sse"]:
            best = {
                "breakpoint": breakpoint,
                "pre_slope": pre_slope,
                "post_slope": post_slope,
                "slope_increase": slope_increase,
                "combined_sse": combined_sse,
            }

    if best is None or single_sse <= 1e-12:
        return {
            **no_cliff,
            "reference_reason": "no material piecewise slope increase",
        }

    improvement = max(
        0.0,
        min(1.0, 1.0 - best["combined_sse"] / single_sse),
    )
    if improvement < float(cfg["minimum_fit_improvement_ratio"]):
        return {
            **no_cliff,
            "fit_improvement_ratio": round(improvement, 4),
            "reference_reason": "piecewise fit improvement below threshold",
        }

    confidence = (
        "high"
        if (
            improvement
            >= min(1.0, 2 * float(cfg["minimum_fit_improvement_ratio"]))
            and best["slope_increase"]
            >= 2 * float(cfg["minimum_slope_increase_sec_per_lap"])
        )
        else "medium"
    )
    return {
        "reference_status": "candidate_for_manual_review",
        "observed_cliff_lap": int(round(x[best["breakpoint"]])),
        "reference_confidence": confidence,
        "pre_cliff_slope_sec_per_lap": round(best["pre_slope"], 4),
        "post_cliff_slope_sec_per_lap": round(best["post_slope"], 4),
        "slope_increase_sec_per_lap": round(best["slope_increase"], 4),
        "fit_improvement_ratio": round(improvement, 4),
        "reference_reason": "piecewise fuel-corrected slope increase",
    }


def extract_session_cliff_references(
    laps: pd.DataFrame,
    weather: Optional[pd.DataFrame],
    race: Mapping[str, Any],
    cleaning_config: Optional[Mapping[str, Any]] = None,
    reference_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return accepted/rejected stints and cleaned lap rows for one race."""
    cleaning_cfg = {**DEFAULT_CLEANING_CONFIG, **(cleaning_config or {})}
    prepared = prepare_fastf1_laps(laps, weather, cleaning_cfg)
    group_columns = ["Driver", "Stint", "CompoundNormalized"]
    accepted_stints = []
    rejected_stints = []
    clean_frames = []

    relevant = prepared[
        prepared["CompoundNormalized"].isin(DRY_COMPOUNDS)
        & prepared["Driver"].notna()
        & prepared["Stint"].notna()
    ]
    for (driver, stint_number, compound), group in relevant.groupby(
        group_columns,
        dropna=False,
    ):
        clean = group[group["AcceptedLap"]].copy()
        raw_count = len(group)
        clean_count = len(clean)
        removed_fraction = (
            (raw_count - clean_count) / raw_count if raw_count else 1.0
        )
        rejection_counts = Counter(
            reason
            for reasons in group["RejectionReasons"]
            for reason in reasons
        )
        base_summary = {
            "race_id": race["id"],
            "split": race["split"],
            "season": race["season"],
            "event_name": race["event_name"],
            "driver": str(driver),
            "team": (
                str(clean["Team"].dropna().iloc[0])
                if "Team" in clean and not clean["Team"].dropna().empty
                else None
            ),
            "stint": int(stint_number),
            "compound": str(compound),
            "raw_lap_count": raw_count,
            "clean_lap_count": clean_count,
            "removed_fraction": round(removed_fraction, 4),
            "lap_rejection_counts": dict(sorted(rejection_counts.items())),
        }

        rejection_reason = None
        if clean_count < int(cleaning_cfg["minimum_clean_laps"]):
            rejection_reason = "insufficient_clean_laps"
        elif removed_fraction > float(
            cleaning_cfg["maximum_removed_fraction"]
        ):
            rejection_reason = "excessive_removed_fraction"
        elif (
            clean["TyreLifeNumeric"].max()
            - clean["TyreLifeNumeric"].min()
            + 1
            < int(cleaning_cfg["minimum_clean_laps"])
        ):
            rejection_reason = "insufficient_tyre_age_span"

        if rejection_reason:
            rejected_stints.append(
                {**base_summary, "stint_rejection_reason": rejection_reason}
            )
            continue

        reference = construct_observed_cliff_reference(
            clean,
            reference_config,
        )
        reference_id = (
            f"{race['id']}:{driver}:stint-{int(stint_number)}:{compound}"
        )
        clean["ReferenceId"] = reference_id
        clean["RaceId"] = race["id"]
        clean["Split"] = race["split"]
        clean_frames.append(clean)
        accepted_stints.append(
            {
                **base_summary,
                "reference_id": reference_id,
                "starting_tyre_age": int(clean["TyreLifeNumeric"].min()),
                "ending_tyre_age": int(clean["TyreLifeNumeric"].max()),
                "mean_air_temp_c": (
                    round(float(clean["AirTemp"].mean()), 2)
                    if clean["AirTemp"].notna().any()
                    else None
                ),
                "mean_track_temp_c": (
                    round(float(clean["TrackTemp"].mean()), 2)
                    if clean["TrackTemp"].notna().any()
                    else None
                ),
                "manual_review_status": "pending",
                "reviewed_cliff_lap": None,
                "review_notes": None,
                **reference,
            }
        )

    clean_laps = (
        pd.concat(clean_frames, ignore_index=True)
        if clean_frames
        else pd.DataFrame()
    )
    lap_rejection_counts = Counter(
        reason
        for reasons in prepared["RejectionReasons"]
        for reason in reasons
    )
    return {
        "race_id": race["id"],
        "accepted_stints": accepted_stints,
        "rejected_stints": rejected_stints,
        "clean_laps": clean_laps,
        "lap_rejection_counts": dict(sorted(lap_rejection_counts.items())),
        "total_laps": len(prepared),
        "accepted_laps": int(prepared["AcceptedLap"].sum()),
    }
