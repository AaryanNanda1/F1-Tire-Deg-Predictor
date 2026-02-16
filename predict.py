import pandas as pd
import joblib
import numpy as np
import sys
import os
from mappings import normalize_team_name, get_track_info

def predict_lap_time(
    lap_number, 
    tyre_life_laps, 
    compound, 
    stint,
    team,
    driver,
    track_name,
    air_temp=30.0,
    track_temp=40.0,
    humidity=50.0,
    rainfall=False,
    wind_speed=2.0,
    team_baseline_pace=100.0,
    field_baseline_pace=100.0
):
    """
    Predicts lap time using the advanced feature model.
    """
    # 1. Load Model and Features
    if not os.path.exists("tire_deg_model.joblib") or not os.path.exists("model_features.joblib"):
        print("Model or features not found. Run train_model.py first.")
        return None
        
    model = joblib.load("tire_deg_model.joblib")
    feature_names = joblib.load("model_features.joblib")
    
    # 2. Normalize and Prepare Inputs
    norm_team = normalize_team_name(team)
    track_info = get_track_info(track_name)
    track_type = track_info['type']
    track_length_km = track_info.get('length_km', 5.0)
    
    tyre_life_km = tyre_life_laps * track_length_km
    
    # Wet logic matching preprocessing
    is_wet = 1 if (compound in ['INTERMEDIATE', 'WET']) or rainfall else 0
    relative_pace = team_baseline_pace - field_baseline_pace
    
    # 3. Create Input DataFrame
    # Must match the columns before get_dummies in preprocessing (conceptually), 
    # but since we need to align with *trained* features (which are already dummied), 
    # we create the raw row first.
    
    input_dict = {
        'Driver': driver,
        'Team': norm_team,
        'LapNumber': lap_number,
        'TyreLife': tyre_life_laps,
        'TyreLifeKM': tyre_life_km,
        'Compound': compound,
        'Stint': stint,
        'TrackType': track_type,
        'IsWet': is_wet,
        'AirTemp': air_temp,
        'TrackTemp': track_temp,
        'Humidity': humidity,
        'Rainfall': int(rainfall),
        'TeamBaselinePace': team_baseline_pace,
        'FieldBaselinePace': field_baseline_pace,
        'RelativePace': relative_pace
    }
    
    raw_df = pd.DataFrame([input_dict])
    
    # 4. One-Hot Encoding Alignment
    # We apply get_dummies to the raw categorical columns we know of.
    categorical_cols = ['Driver', 'Team', 'Compound', 'TrackType']
    df_dummies = pd.get_dummies(raw_df, columns=categorical_cols, drop_first=False)
    
    # 5. Reindex to match training columns
    # This adds 0 for missing columns (e.g. Team_Ferrari if we passed Team_Mercedes)
    # and drops extra columns if any (unlikely).
    final_input = df_dummies.reindex(columns=feature_names, fill_value=0)
    
    # 6. Predict
    prediction = model.predict(final_input)[0]
    return prediction

if __name__ == "__main__":
    if not os.path.exists("tire_deg_model.joblib"):
        print("Please train the model first.")
    else:
        print("Predicting for Max Verstappen (Red Bull) at Bahrain...")
        # Example: Lap 15, Softs (5 laps old), Bahrain (Medium Speed)
        pred = predict_lap_time(
            lap_number=15,
            tyre_life_laps=5,
            compound='SOFT',
            stint=1,
            team='Red Bull Racing', # Will be normalized to 'Red Bull'
            driver='1', # Max Verstappen's number is usually 1 or 33, fastf1 uses string numbers often
            track_name='Bahrain International Circuit',
            air_temp=28.5,
            track_temp=35.2,
            team_baseline_pace=96.5, # Example value
            field_baseline_pace=97.0
        )
        print(f"Predicted Lap Time: {pred:.3f} seconds")
