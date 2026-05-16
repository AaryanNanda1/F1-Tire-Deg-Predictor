import pandas as pd
import numpy as np
from mappings import TEAM_MAPPING, get_track_info, normalize_team_name, get_track_features, TRACK_CONFIG

def preprocess_laps(session):
    """
    Cleans and processes lap data for tire degradation modeling with advanced features.
    
    Features include:
    - Core tire metrics (TyreLife, NormalizedTyreLife, TyreLifeSquared, etc.)
    - Weather data (AirTemp, TrackTemp, Humidity, Rainfall, WindSpeed)
    - 10 track characteristic features (7 raw + 3 derived)
    - 3 race-distance normalization features
    - 5 interaction features for cross-domain learning
    
    Args:
        session (fastf1.core.Session): Loaded session object.
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for training.
    """
    # 1. Filter for accurate laps and Green flag conditions
    # We remove laps with Safety Car (SC), VSC, or yellow flags as they don't represent true tire perf.
    laps = session.laps.pick_accurate().pick_track_status('1')
    
    # 2. Filter for valid tire compounds (Slicks only for now, unless wet is requested)
    # The user asked for "Wet/dry indicator" so we should keep Wets/Inters if they exist, 
    # but for consistent *tire deg* modeling, mixing wet/dry laps in one model is tricky.
    # However, the user explicitly asked for "is_wet" feature, so we KEEP them and mark them.
    valid_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    laps = laps[laps['Compound'].isin(valid_compounds)]
    
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
    
    # 5b. Track Characteristics — 10 numeric features (7 raw + 3 derived)
    track_feats = get_track_features(circuit_name)
    for feat_name, feat_val in track_feats.items():
        laps[feat_name] = feat_val
    
    # 6. Distance-based Tyre Life
    laps['TyreLifeKM'] = laps['TyreLife'] * track_length_km
    
    # 6b. Filter out very short stints (< 3 laps): removes installation laps, in/out laps, SC anomalies
    stint_len_map = laps.groupby(['Driver', 'Stint'])['TyreLife'].transform('max')
    laps['StintLength'] = stint_len_map.clip(lower=1)
    laps = laps[laps['StintLength'] >= 3].copy()
    
    # 7a. NormalizedTyreLife: Where is this tire in its life? (0 = fresh, 1 = end of stint)
    # Soft at TyreLife=10/StintLength=12 -> 0.83 (nearly done)
    # Hard at TyreLife=10/StintLength=38 -> 0.26 (still early)
    # This single feature gives the model the compound-differentiation signal without hardcoding.
    laps['NormalizedTyreLife'] = (laps['TyreLife'] / laps['StintLength']).clip(0, 1)
    
    # 7b. TyreLifeSquared: Captures exponential/non-linear late-stint degradation cliffs
    laps['TyreLifeSquared'] = laps['TyreLife'] ** 2
    
    # 7c. FuelLoad proxy: 1.0 = full tank (lap 1), 0.0 = empty (last lap)
    # Standard F1 estimate: 0.07s per lap of fuel burn removes ~0.07s of lap time per lap
    total_laps = laps['LapNumber'].max()
    laps['FuelLoad'] = (1.0 - (laps['LapNumber'] / total_laps)).clip(0, 1)
    
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
    
    # 9. Pace Features (Team & Field Baseline)
    # Calculate baseline pace for this session to contextulize performance.
    # We'll take the median of the top 50% accurate laps as 'FieldBaseline'
    field_baseline = laps['LapTimeSeconds'].median()
    laps['FieldBaselinePace'] = field_baseline
    
    # Team Baseline: Median pace per team
    team_baselines = laps.groupby('Team')['LapTimeSeconds'].median().to_dict()
    laps['TeamBaselinePace'] = laps['Team'].map(team_baselines)
    
    # Relative Pace: How much faster/slower is the team compared to field?
    # Negative = Faster, Positive = Slower
    laps['RelativePace'] = laps['TeamBaselinePace'] - laps['FieldBaselinePace']
    
    # 10. Interaction Features
    # These help the model learn cross-domain relationships:
    # - tire_age × abrasiveness: tire age costs more at abrasive tracks
    # - TrackTemp × track_temp_sensitivity: temperature matters more at sensitive circuits
    # - TyreLife × traction: rear-limited degradation at traction-heavy circuits
    # - TyreLife × lateral_load: sustained cornering wears tires differently
    # - NormalizedTyreLife × thermal_stress: thermal degradation accelerates with tire age
    laps['tire_age_x_abrasiveness'] = laps['TyreLife'] * laps['abrasiveness']
    if 'TrackTemp' in laps.columns:
        laps['track_temp_x_sensitivity'] = laps['TrackTemp'] * laps['track_temp_sensitivity']
    else:
        laps['track_temp_x_sensitivity'] = 0.0
    laps['tire_age_x_traction'] = laps['TyreLife'] * laps['traction']
    laps['tire_age_x_lateral_load'] = laps['TyreLife'] * laps['lateral_load']
    laps['normalized_life_x_thermal'] = laps['NormalizedTyreLife'] * laps['thermal_stress']
    
    # 11. Final Feature Selection
    features = [
        'Driver',
        'Team',
        'LapNumber',
        'TyreLife',
        'TyreLifeKM',
        'StintLength',
        'NormalizedTyreLife',
        'TyreLifeSquared',
        'FuelLoad',
        'Compound',
        'Stint',
        'TrackType',
        'IsWet',
        'AirTemp',
        'TrackTemp',
        'Humidity',
        'Rainfall',
        'WindSpeed',
        'TeamBaselinePace',
        'FieldBaselinePace',
        'RelativePace',
        # Track characteristic features (10)
        'traction',
        'high_speed_load',
        'abrasiveness',
        'surface_roughness',
        'braking_severity',
        'lateral_load',
        'track_temp_sensitivity',
        'thermal_stress',
        'surface_wear',
        'energy_load',
        # Race-distance normalization (3)
        'NormalizedLap',
        'LapsRemaining',
        'TireAgeRatio',
        # Interaction features (5)
        'tire_age_x_abrasiveness',
        'track_temp_x_sensitivity',
        'tire_age_x_traction',
        'tire_age_x_lateral_load',
        'normalized_life_x_thermal',
        # Metadata
        'EventDate',
        'EventName',
        'LapTimeSeconds'  # Target
    ]
    
    # Filter standard numerical/categorical columns
    # We might have missing values from the merge or calculations
    df = laps[features].copy()
    
    # Handle categoricals: Enforce fixed categories to ensure feature alignment across models
    # This prevents 'Medium' only models from failing on 'Slow' speed track simulations.
    ALL_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    ALL_TRACK_TYPES = ['Slow', 'Medium', 'Fast']
    
    df['Compound'] = pd.Categorical(df['Compound'], categories=ALL_COMPOUNDS)
    df['TrackType'] = pd.Categorical(df['TrackType'], categories=ALL_TRACK_TYPES)
    
    categorical_cols = ['Driver', 'Team', 'Compound', 'TrackType']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Boolean to Int
    df['IsWet'] = df['IsWet'].astype(int)
    if 'Rainfall' in df.columns:
        df['Rainfall'] = df['Rainfall'].astype(int)
        
    df.dropna(inplace=True)
    
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

