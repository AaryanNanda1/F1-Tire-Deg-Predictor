from dataclasses import dataclass
from typing import List, Tuple

import fastf1
import pandas as pd

from data_loader import load_race_data
from mappings import get_track_info, normalize_team_name


VALID_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


@dataclass
class RaceSlice:
    year: int
    event_name: str
    weight: float
    source: str


def _resolve_target_event(target_year: int, target_grand_prix: str) -> Tuple[int, str]:
    session = fastf1.get_session(target_year, target_grand_prix, "R")
    event = session.event
    return int(event["RoundNumber"]), str(event["EventName"])


def resolve_target_context(target_year: int, target_grand_prix: str) -> dict:
    target_round, target_event_name = _resolve_target_event(target_year, target_grand_prix)
    track_info = get_track_info(target_event_name)
    return {
        "round_number": target_round,
        "event_name": target_event_name,
        "track_type": track_info["type"],
        "track_length_km": float(track_info.get("length_km", 5.0)),
    }


def _build_slices(target_year: int, target_grand_prix: str) -> List[RaceSlice]:
    target_round, target_event_name = _resolve_target_event(target_year, target_grand_prix)
    schedule = fastf1.get_event_schedule(target_year, include_testing=False)
    prior = schedule[schedule["RoundNumber"] < target_round].sort_values("RoundNumber")

    slices: List[RaceSlice] = []
    recent_three = prior.tail(3)
    recent_weights = [3.0, 2.5, 2.0]

    for idx, (_, row) in enumerate(recent_three.iloc[::-1].iterrows()):
        slices.append(
            RaceSlice(
                year=target_year,
                event_name=str(row["EventName"]),
                weight=recent_weights[idx],
                source=f"prev_{idx + 1}_race",
            )
        )

    older = prior.iloc[: max(0, len(prior) - len(recent_three))]
    for _, row in older.iterrows():
        slices.append(
            RaceSlice(
                year=target_year,
                event_name=str(row["EventName"]),
                weight=1.0,
                source="older_current_season",
            )
        )

    slices.append(
        RaceSlice(
            year=target_year - 1,
            event_name=target_event_name,
            weight=2.5,
            source="same_race_prev_year",
        )
    )
    return slices


def _extract_session_laps(
    session, year: int, source_weight: float, source_name: str
) -> pd.DataFrame:
    laps = session.laps.pick_accurate().pick_track_status("1").copy()
    laps = laps[laps["Compound"].isin(VALID_COMPOUNDS)].copy()
    if laps.empty:
        return pd.DataFrame()

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["Team"] = laps["Team"].apply(normalize_team_name)
    laps["Driver"] = laps["Driver"].astype(str)
    laps["Year"] = year
    laps["EventName"] = str(session.event["EventName"])
    laps["RoundNumber"] = int(session.event["RoundNumber"])

    track_info = get_track_info(str(session.event["EventName"]))
    track_type = track_info["type"]
    track_length_km = float(track_info.get("length_km", 5.0))
    laps["TrackType"] = track_type
    laps["TrackLengthKM"] = track_length_km
    laps["TyreLifeKM"] = laps["TyreLife"] * track_length_km

    weather = session.weather_data.copy()
    if "Time" in weather.columns and not weather.empty:
        weather_cols = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed", "WindDirection"]
        available = [c for c in weather_cols if c in weather.columns]
        laps = laps.sort_values("Time")
        weather = weather.sort_values("Time")
        laps = pd.merge_asof(laps, weather[["Time"] + available], on="Time", direction="backward")

    if "Rainfall" not in laps.columns:
        laps["Rainfall"] = False
    laps["Rainfall"] = laps["Rainfall"].fillna(False).astype(int)
    laps["IsWet"] = ((laps["Compound"].isin(["INTERMEDIATE", "WET"])) | (laps["Rainfall"] > 0)).astype(int)
    laps["DataWeight"] = source_weight
    laps["DataSource"] = source_name

    base_cols = [
        "Year",
        "RoundNumber",
        "EventName",
        "Driver",
        "Team",
        "LapNumber",
        "TyreLife",
        "TyreLifeKM",
        "TrackLengthKM",
        "Compound",
        "Stint",
        "TrackType",
        "IsWet",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Rainfall",
        "LapTimeSeconds",
        "DataWeight",
        "DataSource",
    ]
    available_cols = [c for c in base_cols if c in laps.columns]
    out = laps[available_cols].copy()
    out.dropna(subset=["LapTimeSeconds", "TyreLifeKM", "Compound", "Driver", "Team"], inplace=True)
    return out


def build_weighted_history(target_year: int, target_grand_prix: str) -> pd.DataFrame:
    """
    Build a weighted pre-race training dataset with emphasis on:
    - Previous 3 races in current season
    - Same race one year prior
    - Older races in current season (lower weight)
    """
    slices = _build_slices(target_year, target_grand_prix)
    frames: List[pd.DataFrame] = []

    for slc in slices:
        try:
            session = load_race_data(slc.year, slc.event_name, "R")
            frame = _extract_session_laps(session, slc.year, slc.weight, slc.source)
            if not frame.empty:
                frames.append(frame)
        except Exception:
            # Keep planning robust when an event cannot be loaded.
            continue

    # Fallback for early season / sparse cache:
    # take the tail of previous season when requested slices are unavailable.
    if not frames:
        try:
            prev_schedule = fastf1.get_event_schedule(target_year - 1, include_testing=False).sort_values("RoundNumber")
            fallback_events = prev_schedule.tail(5)
            for _, row in fallback_events.iterrows():
                session = load_race_data(target_year - 1, str(row["EventName"]), "R")
                frame = _extract_session_laps(session, target_year - 1, 1.2, "fallback_prev_season_tail")
                if not frame.empty:
                    frames.append(frame)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
