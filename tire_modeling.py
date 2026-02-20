from typing import Dict, Optional

import numpy as np
import pandas as pd


VALID_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = weights.sum() / 2.0
    return float(values[np.searchsorted(cdf, cutoff)])


def _apply_lap_delta(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["Year", "EventName", "Driver", "Team", "Stint", "Compound"]
    out = df.copy()
    out["StintBaseline"] = out.groupby(keys)["LapTimeSeconds"].transform("min")
    out["LapDelta"] = out["LapTimeSeconds"] - out["StintBaseline"]
    return out


def compute_wet_experience_km(
    history_df: pd.DataFrame,
    target_year: int,
    driver: str,
    team: str,
    target_track_type: str,
) -> float:
    season = history_df[history_df["Year"] == target_year]
    season = season[(season["Driver"] == driver) & (season["Team"] == team)]
    season = season[(season["IsWet"] == 1) & (season["TrackType"] == target_track_type)]
    if season.empty:
        return 0.0
    return float((season["TrackLengthKM"]).sum())


def build_compound_models(
    history_df: pd.DataFrame,
    driver: str,
    team: str,
    track_type: str,
    track_length_km: float,
    wet_experience_km: float = 0.0,
    lap_delta_window_sec: float = 1.2,
) -> Dict[str, Dict[str, float]]:
    """
    Fit compound-specific degradation models and derive stint windows.
    """
    if history_df.empty:
        return {}

    scoped = history_df[(history_df["Driver"] == driver) & (history_df["Team"] == team)].copy()
    if scoped.empty:
        scoped = history_df[history_df["Team"] == team].copy()
    if scoped.empty:
        scoped = history_df.copy()

    scoped = _apply_lap_delta(scoped)
    models: Dict[str, Dict[str, float]] = {}

    for compound in VALID_COMPOUNDS:
        cdf = scoped[scoped["Compound"] == compound].copy()
        if cdf.empty:
            continue

        in_track = cdf[cdf["TrackType"] == track_type]
        if len(in_track) >= 12:
            cdf = in_track

        x = cdf["TyreLifeKM"].to_numpy(dtype=float)
        y = cdf["LapDelta"].to_numpy(dtype=float)
        w = cdf["DataWeight"].fillna(1.0).to_numpy(dtype=float)

        if len(cdf) >= 6 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1, w=w)
        else:
            slope, intercept = 0.03, 0.0

        slope = max(0.0, float(slope))
        intercept = max(0.0, float(intercept))

        # Wet experience on similar-speed tracks lowers expected wet degradation.
        if compound in {"INTERMEDIATE", "WET"} and wet_experience_km > 0:
            reduction = min(0.2, wet_experience_km / 2000.0)
            slope = slope * (1.0 - reduction)

        if slope > 1e-6:
            window_km = lap_delta_window_sec / slope
        else:
            window_km = float(cdf["TyreLifeKM"].quantile(0.75))
        window_km = min(window_km, float(cdf["TyreLifeKM"].quantile(0.9)))
        window_laps = max(1, int(round(window_km / track_length_km)))

        fresh_slice = cdf[cdf["TyreLife"] <= 2]
        if fresh_slice.empty:
            fresh_slice = cdf
        fresh_lap = _weighted_median(
            fresh_slice["LapTimeSeconds"].to_numpy(dtype=float),
            fresh_slice["DataWeight"].fillna(1.0).to_numpy(dtype=float),
        )

        models[compound] = {
            "slope_sec_per_km": slope,
            "intercept_sec": intercept,
            "window_km": float(window_km),
            "window_laps": float(window_laps),
            "fresh_lap_time_sec": float(fresh_lap),
            "sample_size": float(len(cdf)),
        }

    return models


def build_overstay_table(
    compound_models: Dict[str, Dict[str, float]],
    track_length_km: float,
    max_extra_laps: int = 10,
) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for compound, model in compound_models.items():
        slope_per_lap = model["slope_sec_per_km"] * track_length_km
        rows = []
        cumulative = 0.0
        for extra in range(1, max_extra_laps + 1):
            incremental = slope_per_lap * extra
            cumulative += incremental
            rows.append(
                {
                    "extra_lap": extra,
                    "incremental_delta_sec": round(float(incremental), 3),
                    "cumulative_delta_sec": round(float(cumulative), 3),
                }
            )
        out[compound] = rows
    return out
