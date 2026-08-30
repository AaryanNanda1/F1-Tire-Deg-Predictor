import os
import joblib
import numpy as np
import pandas as pd
from mappings import (
    EVENT_NAME_TO_CIRCUIT,
    TRACK_BASE_PACE,
    TRACK_PIT_LOSS,
    get_legacy_track_feature_aliases,
    get_track_characteristic_source,
    get_track_features,
    get_track_info,
    normalize_team_name,
)
from weather_api import get_track_weather
from tire_life_analysis import analyze_tire_life

REVERSE_EVENT_NAME_MAP = {v: k for k, v in EVENT_NAME_TO_CIRCUIT.items()}

class TireDegradationSimulator:
    def __init__(self, year: int, models_dir: str = "models", force_jit_check: bool = True):
        self.year = year
        self.models_dir = models_dir
        
        # Determine correct era model
        if 2022 <= year <= 2025:
            prefix = "ground_effect_2022_2025"
            start_y, end_y = 2022, 2025
        elif 2026 <= year <= 2030:
            prefix = "active_aero_2026_2030"
            start_y, end_y = 2026, 2030
        else:
            raise ValueError(f"Year {year} falls outside supported era models (2022-2030)")
            
        model_path = os.path.join(models_dir, f"{prefix}_model.joblib")
        features_path = os.path.join(models_dir, f"{prefix}_features.joblib")
        metadata_path = os.path.join(models_dir, "era_training_metadata.json")
        
        # JIT Training check: if models missing or older than 5 days, retrain
        needs_retrain = False
        import json, datetime
        if not os.path.exists(model_path) or not os.path.exists(features_path) or not os.path.exists(metadata_path):
            needs_retrain = True
        else:
            with open(metadata_path, "r") as f:
                try:
                    meta = json.load(f)
                    as_of_str = meta.get(prefix, {}).get("as_of")
                    if not as_of_str:
                        needs_retrain = True
                    else:
                        last_train_date = datetime.date.fromisoformat(as_of_str)
                        if (datetime.date.today() - last_train_date).days > 5:
                            needs_retrain = True
                except json.JSONDecodeError:
                    needs_retrain = True
                    
        # Skip JIT training if explicitly told to (useful for fast local testing)
        if not force_jit_check:
            needs_retrain = False
            
        if needs_retrain:
            print(f"JIT Update Triggered: Pulling the latest API data and training the {prefix} model...")
            from train_era_models import train_era
            import shutil
            from pathlib import Path
            
            out_dir = Path(models_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Perform JIT training
            result = train_era(start_y, end_y, prefix, datetime.date.today(), out_dir)
            
            # Update metadata explicitly
            existing_meta = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    try:
                        existing_meta = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            existing_meta[prefix] = result
            with open(metadata_path, "w") as f:
                json.dump(existing_meta, f, indent=2)
                
            # Clear raw fastf1 cache
            cache_path = Path("cache")
            if cache_path.exists():
                shutil.rmtree(cache_path)
            print("JIT Training Complete!")
            
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        self.valid_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
        
        # Realistic typical stint lengths per compound used as simulation inputs only.
        # These set the NormalizedTyreLife context so the model knows how far into its
        # life the tire is.  The model learned actual physics; these are what-if parameters.
        self.compound_sim_stints = {
            "SOFT":         15,
            "MEDIUM":       28,
            "HARD":         42,
            "INTERMEDIATE": 20,
            "WET":          25,
        }

    def _simulate_compound(self, driver, norm_team, track_name, track_type, track_length_km,
                           compound, weather_data, track_features, race_laps, max_laps=50):
        """Simulates lap times for a single compound from lap 1 to max_laps.
        
        Args:
            track_features: dict with 7 source-backed track characteristic values
            race_laps: int, official race lap count for this circuit
        
        Returns:
            drop_off_curve: list of floats representing time lost vs. fresh tire (for cliff analysis)
            predictions: list of absolute predicted lap times in seconds (for UI graph)
        """
        is_wet = 1 if compound in ["INTERMEDIATE", "WET"] or weather_data["rainfall"] else 0
        
        # Compound-specific simulation context
        sim_stint_length = self.compound_sim_stints.get(compound, max_laps)
        fuel_load = max(0.0, 1.0 - (15 / max_laps))  # LapNumber=15 held constant

        rows = []
        for age in range(1, max_laps + 1):
            normalized_life = min(1.0, age / sim_stint_length)
            normalized_lap = min(1.0, 15 / race_laps)  # LapNumber held at 15
            laps_remaining = max(0, race_laps - 15)
            tire_age_ratio = min(1.0, age / race_laps)
            
            row = {
                'Driver': driver,
                'Team': norm_team,
                'LapNumber': 15,  # Held constant to isolate pure tire degradation from fuel burn
                'TyreLife': age,
                'TyreLifeKM': age * track_length_km,
                'StintLength': sim_stint_length,
                'NormalizedTyreLife': normalized_life,
                'TyreLifeSquared': age ** 2,
                'FuelLoad': fuel_load,
                'Compound': compound,
                'Stint': 1,
                'TrackType': track_type,
                'IsWet': is_wet,
                'AirTemp': weather_data['air_temp'],
                'TrackTemp': weather_data['track_temp'],
                'Humidity': weather_data['humidity'],
                'Rainfall': int(weather_data['rainfall']),
                'WindSpeed': weather_data.get('wind_speed', 10.0),
                'TeamBaselinePace': 100.0,
                'FieldBaselinePace': 100.0,
                'RelativePace': 0.0,
                # Race-distance normalization
                'NormalizedLap': normalized_lap,
                'LapsRemaining': laps_remaining,
                'TireAgeRatio': tire_age_ratio,
            }
            
            # Add the seven source-backed track characteristic features.
            for feat_name, feat_val in track_features.items():
                row[feat_name] = feat_val

            # Keep committed pre-migration model artifacts usable.  A newly
            # trained model ignores these aliases because they are not in its
            # persisted feature schema.
            row.update(get_legacy_track_feature_aliases(track_features))
            
            # Add interaction features
            row['tire_age_x_abrasiveness'] = age * track_features.get('abrasiveness', 0.5)
            row['track_temp_x_tyre_stress'] = weather_data['track_temp'] * track_features.get('tyre_stress', 0.5)
            row['tire_age_x_traction'] = age * track_features.get('traction', 0.5)
            row['tire_age_x_lateral_load'] = age * track_features.get('lateral_load', 0.5)
            row['normalized_life_x_tyre_stress'] = normalized_life * track_features.get('tyre_stress', 0.5)

            # Interaction aliases for pre-migration model artifacts.
            row['track_temp_x_sensitivity'] = row['track_temp_x_tyre_stress']
            row['normalized_life_x_thermal'] = row['normalized_life_x_tyre_stress']
            
            # --- New Compound-Specific Interactions ---
            row['soft_age_interaction'] = age if compound == 'SOFT' else 0
            row['medium_age_interaction'] = age if compound == 'MEDIUM' else 0
            row['hard_age_interaction'] = age if compound == 'HARD' else 0
            row['soft_abrasiveness_interaction'] = track_features.get('abrasiveness', 0.5) if compound == 'SOFT' else 0
            row['soft_traction_interaction'] = track_features.get('traction', 0.5) if compound == 'SOFT' else 0
            
            event_name = REVERSE_EVENT_NAME_MAP.get(track_name, track_name)
            row['EventName'] = event_name
            row['EventDate'] = pd.Timestamp.now().date()
            row['LapTimeDelta'] = 0
            
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Handle categoricals: Enforce fixed categories to ensure feature alignment with models
        ALL_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        ALL_TRACK_TYPES = ['Slow', 'Medium', 'Fast']
        
        df['Compound'] = pd.Categorical(df['Compound'], categories=ALL_COMPOUNDS)
        df['TrackType'] = pd.Categorical(df['TrackType'], categories=ALL_TRACK_TYPES)
        
        # Dummy variables matching training
        categorical_cols = ['Driver', 'Team', 'Compound', 'TrackType', 'EventName']
        df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        final_input = df_dummies.reindex(columns=self.feature_names, fill_value=0)
        
        raw_predictions = self.model.predict(final_input)
        
        # Since the model predicts LapTimeDelta, reconstruct absolute lap time using track's base pace
        base_pace = TRACK_BASE_PACE.get(track_name, 90.0)
        absolute_lap_times = [round(float(p) + base_pace, 3) for p in raw_predictions]
        
        # Calculate drop-off curve relative to the first lap on this tire
        base_lap = raw_predictions[0]
        drop_off_curve = [float(p - base_lap) for p in raw_predictions]
        
        # Enforce monotonic increase on drop-off (suppress model noise)
        for i in range(1, len(drop_off_curve)):
            drop_off_curve[i] = max(drop_off_curve[i-1], drop_off_curve[i])
            
        return drop_off_curve, absolute_lap_times

    def _analyze_curve(self, drop_off_curve, absolute_lap_times, track_name=None):
        """
        Analyzes a degradation curve using the dual-output tire_life_analysis engine.

        Produces TWO independent outputs:
          1. performance_cliff_lap — Where the tire's lap-time curve begins to
             noticeably worsen. Based purely on tire physics (sustained acceleration
             rule). May be None if no clear cliff exists.

          2. strategy_useful_life_lap — The last lap where staying out is still
             better than pitting. Based on cumulative degradation cost vs pit loss.
             A tire past its cliff may still be strategically worth using.

        All existing API fields are preserved for frontend compatibility.
        """
        # Look up track-specific pit loss; fall back to 22s default
        pit_loss = TRACK_PIT_LOSS.get(track_name, 22.0) if track_name else 22.0

        # Run the full dual-output analysis pipeline
        result = analyze_tire_life(
            raw_times=absolute_lap_times,
            pit_loss=pit_loss,
        )

        useful_life = result["strategy_useful_life_lap"]
        cliff_lap = result["performance_cliff_lap"]
        suggested_lo = max(1, useful_life - 2)
        suggested_hi = useful_life + 2

        return {
            # --- Existing fields (frontend backward compatibility) ---
            # cliff_point_lap is mapped to strategy_useful_life_lap for the
            # ReferenceLine and strategy optimizer. The frontend will also
            # get the new separate fields below.
            "cliff_point_lap": useful_life,
            "drop_off_per_lap_sec": result["drop_off_per_lap_sec"],
            "suggested_lifespan": f"{suggested_lo}-{suggested_hi} laps",
            "graph_data": {lap: val for lap, val in enumerate(absolute_lap_times, start=1)},

            # --- Smoothed data for charting ---
            "smoothed_graph_data": {
                lap: round(val, 3)
                for lap, val in enumerate(result["smoothed_times"], start=1)
            },

            # --- Performance Cliff (tire physics) ---
            "performance_cliff_lap": cliff_lap,
            "cliff_confidence": result["cliff_confidence"],
            "cliff_reason": result["cliff_reason"],

            # --- Strategy Useful Life (race strategy) ---
            "strategy_useful_life_lap": useful_life,
            "strategy_confidence": result["strategy_confidence"],
            "strategy_reason": result["strategy_reason"],
        }

    def simulate(self, driver: str, team: str, track_name: str, race_date: str, race_time: str = None):
        """
        Main entry point. Fetches weather and simulates all compounds.
        """
        norm_team = normalize_team_name(team)
        track_info = get_track_info(track_name)
        track_type = track_info['type']
        track_length_km = track_info.get('length_km', 5.0)
        race_laps = track_info.get('race_laps', 57)
        track_features = get_track_features(track_name)
        track_feature_source = get_track_characteristic_source(track_name)
        
        # 1. Fetch live or historical weather based on date and exact time (B)
        weather_data = get_track_weather(track_name, race_date, race_time)
        
        results = {
            "input_context": {
                "driver": driver,
                "team": norm_team,
                "year": self.year,
                "track": track_name,
                "track_category": track_type,
                "track_features": track_features,
                "track_feature_source": track_feature_source,
                "race_time": race_time,
                "weather": weather_data
            },
            "compounds": {}
        }
        
        # 2. Simulate for each compound
        for compound in self.valid_compounds:
            # Wets don't make sense if dry, and slicks don't make sense if heavy rain,
            # but we'll generate all to provide the full "what-if" mapping.
            max_laps = race_laps + 2  # Full race distance + 2
            
            drop_off_curve, absolute_lap_times = self._simulate_compound(
                driver, norm_team, track_name, track_type, track_length_km, compound,
                weather_data, track_features, race_laps, max_laps
            )
            
            analysis = self._analyze_curve(drop_off_curve, absolute_lap_times, track_name=track_name)
            results["compounds"][compound] = analysis
            
        return results

if __name__ == "__main__":
    # Quick Test
    import json
    sim = TireDegradationSimulator(year=2026)
    out = sim.simulate("VER", "Red Bull Racing", "Bahrain International Circuit", "2026-03-05", "18:00")
    print(json.dumps(out["input_context"]["weather"], indent=2))
    print(json.dumps(out["compounds"]["SOFT"], indent=2))
