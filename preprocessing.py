import pandas as pd
import numpy as np
from mappings import TEAM_MAPPING, get_track_info, normalize_team_name

def preprocess_laps(session):
    """
    Cleans and processes lap data for tire degradation modeling with advanced features.
    
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
    
    # 5. Track Information
    circuit_name = session.event['EventName']
    track_info = get_track_info(circuit_name)
    laps['TrackType'] = track_info['type'] # High, Medium, Low
    track_length_km = track_info.get('length_km', 5.0)
    
    # 6. Distance-based Tyre Life
    # TyreLife is in Laps. User wants TyreLife in KM.
    laps['TyreLifeKM'] = laps['TyreLife'] * track_length_km
    
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
    
    # 10. Final Feature Selection
    features = [
        'Driver',
        'Team',
        'LapNumber',
        'TyreLife',
        'TyreLifeKM',
        'Compound',
        'Stint',
        'TrackType',
        'IsWet',
        'AirTemp',
        'TrackTemp',
        'Humidity',
        'Rainfall',
        'TeamBaselinePace',
        'FieldBaselinePace',
        'RelativePace',
        'LapTimeSeconds' # Target
    ]
    
    # Filter standard numerical/categorical columns
    # We might have missing values from the merge or calculations
    df = laps[features].copy()
    
    # Handle categoricals: One-Hot Encoding
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
