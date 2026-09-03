"""Canonical feature groups used by the feature-engineering experiments."""

from __future__ import annotations

TRACK_FEATURES = [
    "traction", "tyre_stress", "asphalt_grip", "corner_speed_energy",
    "abrasiveness", "braking_severity", "lateral_load",
]
TYRE_AGE_FEATURES = ["TyreLife", "TyreLifeKM", "TyreLifeSquared", "TireAgeRatio"]
PROGRESS_FEATURES = ["LapNumber", "NormalizedLap", "LapsRemaining", "FuelLoad", "Stint"]
WEATHER_FEATURES = [
    "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed", "IsWet", "FuelLoadMissing",
]
TRACK_AGE_INTERACTIONS = [
    "tire_age_x_abrasiveness", "track_temp_x_tyre_stress", "tire_age_x_traction",
    "tire_age_x_lateral_load", "soft_abrasiveness_interaction", "soft_traction_interaction",
]
COMPOUND_AGE_INTERACTIONS = ["soft_age_interaction", "medium_age_interaction", "hard_age_interaction"]
CATEGORICAL_FEATURES = ["Driver", "Team", "Compound", "TrackType", "SessionCode"]
TARGET = "LapTimeDelta"
METADATA = ["EventDate", "EventName", "SessionKey", "TrainingRole", "Season", "SampleWeight"]

ALL_RAW_FEATURES = (
    CATEGORICAL_FEATURES + TYRE_AGE_FEATURES + PROGRESS_FEATURES + WEATHER_FEATURES
    + TRACK_FEATURES + TRACK_AGE_INTERACTIONS + COMPOUND_AGE_INTERACTIONS
)

LEAKAGE_FEATURES = {
    "StintLength", "NormalizedTyreLife", "normalized_life_x_tyre_stress",
    "TeamBaselinePace", "FieldBaselinePace", "RelativePace", "EventName",
    "EventDate", "LapTimeDelta",
}

AGE_REDUNDANCY_REMOVALS = ["TyreLifeKM", "TyreLifeSquared"]
PROGRESS_REDUNDANCY_REMOVALS = ["LapsRemaining"]

FEATURE_GROUPS = {
    "circuit": TRACK_FEATURES,
    "tyre_age": TYRE_AGE_FEATURES,
    "race_progress": PROGRESS_FEATURES,
    "weather": WEATHER_FEATURES,
    "track_age_interactions": TRACK_AGE_INTERACTIONS,
    "compound_age_interactions": COMPOUND_AGE_INTERACTIONS,
    "categorical_context": CATEGORICAL_FEATURES,
}


def assert_safe_features(features: list[str]) -> None:
    prohibited = sorted(set(features) & LEAKAGE_FEATURES)
    if prohibited:
        raise ValueError(f"Prohibited leakage/metadata features entered a training matrix: {prohibited}")

