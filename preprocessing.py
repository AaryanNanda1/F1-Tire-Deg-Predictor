import pandas as pd
import numpy as np
from mappings import (
    TEAM_MAPPING,
    TRACK_BASE_PACE,
    TRACK_CONFIG,
    get_track_info,
    get_track_features,
    normalize_team_name,
    resolve_circuit_key,
)


# Increment this whenever the persisted processed-session feature schema or
# feature-generation logic changes.  The training-data store records it in
# each manifest so immutable session keys do not hide stale preprocessing.
PREPROCESSING_VERSION = "track-features-v8-observed-stints-missing-weather"
ROBUST_FILTER_MAD_MULTIPLIER = 6.0
ROBUST_FILTER_MIN_THRESHOLD_SEC = 3.0
FUTURE_INFORMATION_FEATURES = frozenset(
    {
        "StintLength",
        "NormalizedTyreLife",
        "normalized_life_x_tyre_stress",
    }
)
SESSION_DERIVED_BASELINE_FEATURES = frozenset(
    {
        "TeamBaselinePace",
        "FieldBaselinePace",
        "RelativePace",
    }
)


def infer_session_code(session) -> str:
    """Map a FastF1 session object to the training session categories."""
    candidates = [
        getattr(session, "name", None),
        getattr(session, "session_name", None),
    ]
    session_info = getattr(session, "session_info", None)
    if hasattr(session_info, "get"):
        candidates.extend(
            session_info.get(key)
            for key in ("Name", "name", "Type", "type")
        )
    label = " ".join(str(value) for value in candidates if value).lower()
    if "practice 2" in label or "fp2" in label:
        return "FP2"
    if "sprint" in label:
        return "S"
    return "R"

def preprocess_laps(session):
    """
    Cleans and processes lap data for tire degradation modeling with advanced features.
    
    Features include:
    - Core tire metrics (TyreLife, TyreLifeSquared, and tire-age interactions)
    - Weather data (AirTemp, TrackTemp, Humidity, Rainfall, WindSpeed)
    - 7 source-backed track characteristic features
    - 3 race-distance normalization features
    - 5 interaction features for cross-domain learning
    
    Args:
        session (fastf1.core.Session): Loaded session object.
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for training.
    """
    audit = {
        "input_rows": int(len(session.laps)),
        "pit_in_out_removed": 0,
        "short_stint_removed": 0,
        "robust_outlier_removed": 0,
        "final_rows": 0,
        "baseline_groups": 0,
    }

    # 1. Filter for accurate laps and Green flag conditions
    # We remove laps with Safety Car (SC), VSC, or yellow flags as they don't represent true tire perf.
    laps = session.laps.pick_accurate().pick_track_status('1')
    
    # 2. Filter for valid tire compounds (Slicks only for now, unless wet is requested)
    # The user asked for "Wet/dry indicator" so we should keep Wets/Inters if they exist, 
    # but for consistent *tire deg* modeling, mixing wet/dry laps in one model is tricky.
    # However, the user explicitly asked for "is_wet" feature, so we KEEP them and mark them.
    valid_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    laps = laps[laps['Compound'].isin(valid_compounds)].copy()

    # Explicitly remove pit-in and pit-out laps. These can pass the green-flag
    # filter but are not representative of tire performance.
    pit_columns = [
        column
        for column in ('PitInTime', 'PitOutTime')
        if column in laps.columns
    ]
    if pit_columns:
        pit_lap_mask = laps[pit_columns].notna().any(axis=1)
        audit["pit_in_out_removed"] = int(pit_lap_mask.sum())
        laps = laps.loc[~pit_lap_mask].copy()
    
    # 3. Basic Lap Features
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['Driver'] = laps['Driver'] # e.g., 'VER', 'HAM'
    
    # 4. Team Name Normalization
    laps['Team'] = laps['Team'].apply(normalize_team_name)
    
    # 4b. Metadata for chronological splitting
    laps['EventDate'] = pd.to_datetime(session.event['EventDate']).date()
    laps['EventName'] = session.event['EventName']
    
    # 5. Track Information
    circuit_name = session.event['EventName']
    track_info = get_track_info(circuit_name)
    laps['TrackType'] = track_info['type']  # Slow, Medium, Fast
    track_length_km = track_info.get('length_km', 5.0)
    race_laps = track_info.get('race_laps', 57)
    
    # 5b. Track Characteristics — 7 source-backed numeric features
    track_feats = get_track_features(circuit_name)
    for feat_name, feat_val in track_feats.items():
        laps[feat_name] = feat_val
    
    # 6. Distance-based Tyre Life
    laps['TyreLifeKM'] = laps['TyreLife'] * track_length_km
    
    # 6b. Filter out very short observed stints (< 3 laps). TyreLife can start
    # at 7 or 8 when a driver takes over a used set, so its maximum is not a
    # valid count of how many laps were actually observed in this session.
    observed_laps = laps.groupby(['Driver', 'Stint'])['LapNumber'].transform('count')
    audit["short_stint_removed"] = int((observed_laps < 3).sum())
    laps = laps.loc[observed_laps >= 3].copy()
    
    # 7b. TyreLifeSquared: Captures exponential/non-linear late-stint degradation cliffs
    laps['TyreLifeSquared'] = laps['TyreLife'] ** 2
    
    # 7c. FuelLoad proxy for race and Sprint sessions only. FP2 lap number
    # does not identify starting fuel, so it receives a neutral sentinel plus
    # an explicit missingness indicator.
    session_code = infer_session_code(session)
    laps['SessionCode'] = session_code
    total_laps = laps['LapNumber'].max()
    if session_code == "FP2":
        laps['FuelLoad'] = np.nan
        laps['FuelLoadMissing'] = 1
    else:
        laps['FuelLoad'] = (1.0 - (laps['LapNumber'] / total_laps)).clip(0, 1)
        laps['FuelLoadMissing'] = 0
    
    # 7d. Race-distance normalization features
    # These help the model understand that lap 20 at Monaco is different from lap 20 at Spa
    laps['NormalizedLap'] = (laps['LapNumber'] / race_laps).clip(0, 1)
    laps['LapsRemaining'] = (race_laps - laps['LapNumber']).clip(0)
    laps['TireAgeRatio'] = (laps['TyreLife'] / race_laps).clip(0, 1)
    
    # 7. Weather Data Integration
    # Weather data is time-series. We need to merge it with laps based on 'Time'.
    # default fastf1 weather data usually has: AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed
    weather = session.weather_data.copy()
    
    # We use merge_asof to find the closest weather data point to the END of the lap (Time)
    # Ensure both are sorted and have the same time type
    laps = laps.sort_values('Time')
    weather = weather.sort_values('Time')
    
    # Select relevant weather columns
    weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall', 'WindSpeed', 'WindDirection']
    # Filter only available columns
    available_weather_cols = [c for c in weather_cols if c in weather.columns]
    
    # Merge
    laps = pd.merge_asof(laps, weather[['Time'] + available_weather_cols], on='Time', direction='backward')
    
    # 8. Wet/Dry Indicator
    # If Rainfall > 0 OR Compound is Wet/Inter -> IsWet = True
    # Note: 'Rainfall' in fastf1 is a boolean flag (True/False) or binary
    laps['IsWet'] = (laps['Compound'].isin(['INTERMEDIATE', 'WET'])) | (laps['Rainfall'] == True)
    
    # 9. Session-local robust pace filtering
    # The session itself is the outer baseline scope; splitting by team and
    # wet/dry regime prevents a wet lap or a slower team from contaminating the
    # baseline used to audit another group. The baseline is filtering-only.
    baseline_groups = laps.groupby(
        ['Team', 'IsWet'],
        observed=True,
    )['LapTimeSeconds']
    group_medians = baseline_groups.transform('median')
    group_mads = baseline_groups.transform(
        lambda values: float(np.median(np.abs(values - np.median(values))))
    )
    thresholds = np.maximum(
        ROBUST_FILTER_MIN_THRESHOLD_SEC,
        ROBUST_FILTER_MAD_MULTIPLIER * group_mads,
    )
    robust_keep = (
        (laps['LapTimeSeconds'] >= group_medians - thresholds)
        & (laps['LapTimeSeconds'] <= group_medians + thresholds)
    )
    audit["robust_outlier_removed"] = int((~robust_keep).sum())
    audit["baseline_groups"] = int(
        laps.loc[robust_keep, ['Team', 'IsWet']].drop_duplicates().shape[0]
    )
    laps = laps.loc[robust_keep].copy()

    # Recalculate clean baselines after filtering for auditability and future
    # filtering extensions. They are deliberately not persisted as predictors.
    clean_baselines = (
        laps.groupby(['Team', 'IsWet'], observed=True)['LapTimeSeconds']
        .median()
        .to_dict()
    )
    audit["clean_baseline_count"] = len(clean_baselines)
    
    # 10. Interaction Features
    # These help the model learn cross-domain relationships:
    # - tire_age × abrasiveness: tire age costs more at abrasive tracks
    # - TrackTemp × tyre_stress: temperature matters more at high-stress circuits
    # - TyreLife × traction: rear-limited degradation at traction-heavy circuits
    # - TyreLife × lateral_load: sustained cornering wears tires differently
    laps['tire_age_x_abrasiveness'] = laps['TyreLife'] * laps['abrasiveness']
    if 'TrackTemp' in laps.columns:
        laps['track_temp_x_tyre_stress'] = laps['TrackTemp'] * laps['tyre_stress']
    else:
        laps['track_temp_x_tyre_stress'] = 0.0
    laps['tire_age_x_traction'] = laps['TyreLife'] * laps['traction']
    laps['tire_age_x_lateral_load'] = laps['TyreLife'] * laps['lateral_load']
    
    # --- New Compound-Specific Interactions ---
    # These help the model learn that different compounds have different degradation slopes
    laps['is_soft'] = (laps['Compound'] == 'SOFT').astype(int)
    laps['is_medium'] = (laps['Compound'] == 'MEDIUM').astype(int)
    laps['is_hard'] = (laps['Compound'] == 'HARD').astype(int)
    
    laps['soft_age_interaction'] = laps['TyreLife'] * laps['is_soft']
    laps['medium_age_interaction'] = laps['TyreLife'] * laps['is_medium']
    laps['hard_age_interaction'] = laps['TyreLife'] * laps['is_hard']
    
    # Compound sensitivity to track characteristics
    laps['soft_abrasiveness_interaction'] = laps['is_soft'] * laps['abrasiveness']
    laps['soft_traction_interaction'] = laps['is_soft'] * laps['traction']
    
    # 11. Final Target Creation
    # Use a known circuit baseline shared with inference. This avoids using
    # the held-out session's complete lap history to construct either X or y.
    circuit_key = resolve_circuit_key(circuit_name)
    target_baseline = TRACK_BASE_PACE.get(circuit_key, 90.0)
    laps['LapTimeDelta'] = laps['LapTimeSeconds'] - target_baseline
    
    # 12. Final Feature Selection
    features = [
        'Driver',
        'Team',
        'LapNumber',
        'TyreLife',
        'TyreLifeKM',
        'TyreLifeSquared',
        'FuelLoad',
        'FuelLoadMissing',
        'Compound',
        'Stint',
        'SessionCode',
        'TrackType',
        'IsWet',
        'AirTemp',
        'TrackTemp',
        'Humidity',
        'Rainfall',
        'WindSpeed',
        # Source-backed track characteristic features (7)
        'traction',
        'tyre_stress',
        'asphalt_grip',
        'corner_speed_energy',
        'abrasiveness',
        'braking_severity',
        'lateral_load',
        # Race-distance normalization (3)
        'NormalizedLap',
        'LapsRemaining',
        'TireAgeRatio',
        # Interaction features (5)
        'tire_age_x_abrasiveness',
        'track_temp_x_tyre_stress',
        'tire_age_x_traction',
        'tire_age_x_lateral_load',
        'soft_age_interaction',
        'medium_age_interaction',
        'hard_age_interaction',
        'soft_abrasiveness_interaction',
        'soft_traction_interaction',
        # Metadata
        'EventDate',
        'EventName',
        'LapTimeDelta'  # Target
    ]

    if FUTURE_INFORMATION_FEATURES.intersection(features):
        raise RuntimeError("Future-information features leaked into the training schema")
    
    # Filter standard numerical/categorical columns
    # We might have missing values from the merge or calculations
    df = laps[features].copy()
    
    # Preserve categorical values in their canonical raw form. Encoding is
    # fitted later inside each chronological training fold so held-out events
    # cannot alter the feature schema and unknown categories are safe.
    categorical_cols = ['Driver', 'Team', 'Compound', 'TrackType', 'SessionCode']
    for column in categorical_cols:
        df[column] = df[column].astype(str)

    # Boolean to Int. Categorical columns remain strings for the fitted
    # ColumnTransformer/OneHotEncoder.
    df['IsWet'] = df['IsWet'].astype(int)
    if 'Rainfall' in df.columns:
        df['Rainfall'] = df['Rainfall'].astype(int)

    # Keep the FP2 missing-fuel state explicit while giving estimators a
    # finite numeric value. FuelLoadMissing tells the model that zero is not
    # an observed empty tank.
    df['FuelLoad'] = df['FuelLoad'].fillna(0.0)
        
    essential_columns = [
        'Driver', 'Team', 'Compound', 'TrackType', 'SessionCode',
        'LapNumber', 'TyreLife', 'LapTimeDelta', 'EventDate', 'EventName',
    ]
    df.dropna(subset=essential_columns, inplace=True)
    audit["final_rows"] = int(len(df))
    df.attrs["filter_audit"] = audit
    
    return df

if __name__ == "__main__":
    from data_loader import load_race_data
    # Use a bigger session to test normalization
    print("Loading data for validation...")
    session = load_race_data(2023, 'Bahrain')
    data = preprocess_laps(session)
    print("Processed Data Head:")
    print(data.head())
    print("\nColumns:", data.columns.tolist())
    print("\nShape:", data.shape)
