import pandas as pd
import joblib
import os
import json
import argparse
from mappings import normalize_team_name, get_track_info
from historical_data import build_weighted_history, resolve_target_context
from tire_modeling import build_compound_models, build_overstay_table, compute_wet_experience_km
from strategy_optimizer import optimize_strategy

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


def _infer_condition(history_df, threshold=0.25):
    if history_df.empty:
        return "dry"
    wet_share = float((history_df["IsWet"] == 1).mean())
    return "wet" if wet_share >= threshold else "dry"


def predict_with_strategy(
    year,
    gp,
    driver,
    team,
    race_laps,
    lap_number,
    tyre_life_laps,
    compound,
    stint,
    air_temp=30.0,
    track_temp=40.0,
    humidity=50.0,
    rainfall=False,
    wind_speed=2.0,
    team_baseline_pace=100.0,
    field_baseline_pace=100.0,
    race_condition="auto",
    pit_loss_sec=21.0,
):
    lap_time_sec = predict_lap_time(
        lap_number=lap_number,
        tyre_life_laps=tyre_life_laps,
        compound=compound,
        stint=stint,
        team=team,
        driver=driver,
        track_name=gp,
        air_temp=air_temp,
        track_temp=track_temp,
        humidity=humidity,
        rainfall=rainfall,
        wind_speed=wind_speed,
        team_baseline_pace=team_baseline_pace,
        field_baseline_pace=field_baseline_pace,
    )
    if lap_time_sec is None:
        return None

    norm_team = normalize_team_name(team)
    target_context = resolve_target_context(year, gp)
    track_type = target_context["track_type"]
    track_length_km = target_context["track_length_km"]

    history = build_weighted_history(year, gp)
    wet_experience_km = compute_wet_experience_km(
        history_df=history,
        target_year=year,
        driver=driver,
        team=norm_team,
        target_track_type=track_type,
    )
    compound_models = build_compound_models(
        history_df=history,
        driver=driver,
        team=norm_team,
        track_type=track_type,
        track_length_km=track_length_km,
        wet_experience_km=wet_experience_km,
    )

    condition = race_condition
    if condition == "auto":
        condition = _infer_condition(history)

    best = optimize_strategy(
        compound_models=compound_models,
        race_laps=race_laps,
        track_length_km=track_length_km,
        race_condition=condition,
        pit_loss_sec=pit_loss_sec,
    )
    overstay = build_overstay_table(compound_models, track_length_km)

    return {
        "lap_time_prediction_sec": float(lap_time_sec),
        "strategy": {
            "target": {
                "year": year,
                "grand_prix": gp,
                "event_name": target_context["event_name"],
                "driver": driver,
                "team": norm_team,
                "track_type": track_type,
                "track_length_km": track_length_km,
                "race_laps": race_laps,
                "race_condition": condition,
            },
            "phase_1_history_rows": int(len(history)),
            "phase_2_compound_models": compound_models,
            "phase_2_wet_experience_km": round(wet_experience_km, 3),
            "phase_3_best_strategies": best,
            "phase_3_overstay_delta": overstay,
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lap time prediction with optional strategy output")
    parser.add_argument("--lap-number", type=int, default=15)
    parser.add_argument("--tyre-life-laps", type=int, default=5)
    parser.add_argument("--compound", type=str, default="SOFT")
    parser.add_argument("--stint", type=int, default=1)
    parser.add_argument("--team", type=str, default="Red Bull Racing")
    parser.add_argument("--driver", type=str, default="VER")
    parser.add_argument("--track-name", type=str, default="Bahrain International Circuit")
    parser.add_argument("--air-temp", type=float, default=28.5)
    parser.add_argument("--track-temp", type=float, default=35.2)
    parser.add_argument("--humidity", type=float, default=50.0)
    parser.add_argument("--rainfall", action="store_true")
    parser.add_argument("--wind-speed", type=float, default=2.0)
    parser.add_argument("--team-baseline-pace", type=float, default=96.5)
    parser.add_argument("--field-baseline-pace", type=float, default=97.0)
    parser.add_argument("--with-strategy", action="store_true")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", type=str, default="Bahrain")
    parser.add_argument("--race-laps", type=int, default=57)
    parser.add_argument("--condition", choices=["auto", "dry", "wet", "mixed"], default="auto")
    parser.add_argument("--pit-loss-sec", type=float, default=21.0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists("tire_deg_model.joblib") or not os.path.exists("model_features.joblib"):
        raise SystemExit("Model or features not found. Run train_model.py first.")

    if args.with_strategy:
        result = predict_with_strategy(
            year=args.year,
            gp=args.gp,
            driver=args.driver,
            team=args.team,
            race_laps=args.race_laps,
            lap_number=args.lap_number,
            tyre_life_laps=args.tyre_life_laps,
            compound=args.compound,
            stint=args.stint,
            air_temp=args.air_temp,
            track_temp=args.track_temp,
            humidity=args.humidity,
            rainfall=args.rainfall,
            wind_speed=args.wind_speed,
            team_baseline_pace=args.team_baseline_pace,
            field_baseline_pace=args.field_baseline_pace,
            race_condition=args.condition,
            pit_loss_sec=args.pit_loss_sec,
        )
        print(json.dumps(result, indent=2))
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
    else:
        pred = predict_lap_time(
            lap_number=args.lap_number,
            tyre_life_laps=args.tyre_life_laps,
            compound=args.compound,
            stint=args.stint,
            team=args.team,
            driver=args.driver,
            track_name=args.track_name,
            air_temp=args.air_temp,
            track_temp=args.track_temp,
            humidity=args.humidity,
            rainfall=args.rainfall,
            wind_speed=args.wind_speed,
            team_baseline_pace=args.team_baseline_pace,
            field_baseline_pace=args.field_baseline_pace,
        )
        print(f"Predicted Lap Time: {pred:.3f} seconds")
