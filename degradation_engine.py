import os
import joblib
import numpy as np
import pandas as pd
from mappings import normalize_team_name, get_track_info, TRACK_PIT_LOSS
from weather_api import get_track_weather
from tire_life_analysis import recommend_tire_life

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

    def _simulate_compound(self, driver, norm_team, track_type, track_length_km, compound, weather_data, max_laps=50):
        """Simulates lap times for a single compound from lap 1 to max_laps.
        
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
            rows.append({
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
                'RelativePace': 0.0
            })
            
        df = pd.DataFrame(rows)
        
        # Handle categoricals: Enforce fixed categories to ensure feature alignment with models
        ALL_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        ALL_TRACK_TYPES = ['Low', 'Medium', 'High']
        
        df['Compound'] = pd.Categorical(df['Compound'], categories=ALL_COMPOUNDS)
        df['TrackType'] = pd.Categorical(df['TrackType'], categories=ALL_TRACK_TYPES)
        
        # Dummy variables matching training
        categorical_cols = ['Driver', 'Team', 'Compound', 'TrackType']
        df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        final_input = df_dummies.reindex(columns=self.feature_names, fill_value=0)
        
        raw_predictions = self.model.predict(final_input)
        absolute_lap_times = [round(float(p), 3) for p in raw_predictions]
        
        # Calculate drop-off curve relative to the first lap on this tire
        base_lap = raw_predictions[0]
        drop_off_curve = [float(p - base_lap) for p in raw_predictions]
        
        # Enforce monotonic increase on drop-off (suppress model noise)
        for i in range(1, len(drop_off_curve)):
            drop_off_curve[i] = max(drop_off_curve[i-1], drop_off_curve[i])
            
        return drop_off_curve, absolute_lap_times

    def _analyze_curve(self, drop_off_curve, absolute_lap_times, track_name=None):
        """
        Analyzes a degradation curve using the robust tire_life_analysis engine.

        Combines:
          1. Savitzky-Golay smoothing to remove lap-to-lap noise
          2. ruptures PELT change-point detection for regime shifts
          3. Cumulative degradation cost vs pit-loss optimization

        The recommended tire life is the EARLIEST lap where staying out costs
        more than pitting, OR where a statistical change point is detected.

        All existing API fields are preserved for frontend compatibility.
        New diagnostic fields are added alongside them.
        """
        # Look up track-specific pit loss; fall back to 22s default
        pit_loss = TRACK_PIT_LOSS.get(track_name, 22.0) if track_name else 22.0

        # Run the full analysis pipeline
        result = recommend_tire_life(
            raw_times=absolute_lap_times,
            pit_loss=pit_loss,
        )

        recommended = result["recommended_max_life"]
        suggested_lifespan_lo = max(1, recommended - 2)
        suggested_lifespan_hi = recommended + 2

        return {
            # --- Existing fields (frontend compatibility) ---
            "drop_off_per_lap_sec": result["drop_off_per_lap_sec"],
            "cliff_point_lap": recommended,  # Now powered by the smarter engine
            "suggested_lifespan": f"{suggested_lifespan_lo}-{suggested_lifespan_hi} laps",
            "graph_data": {lap: val for lap, val in enumerate(absolute_lap_times, start=1)},

            # --- New diagnostic fields ---
            "smoothed_graph_data": {
                lap: round(val, 3)
                for lap, val in enumerate(result["smoothed_times"], start=1)
            },
            "change_point_lap": result["change_point_lap"],
            "cost_crossover_lap": result["cost_crossover_lap"],
            "recommended_max_life": recommended,
            "recommendation_reason": result["recommendation_reason"],
            "confidence": result["confidence"],
        }

    def simulate(self, driver: str, team: str, track_name: str, race_date: str, race_time: str = None):
        """
        Main entry point. Fetches weather and simulates all compounds.
        """
        norm_team = normalize_team_name(team)
        track_info = get_track_info(track_name)
        track_type = track_info['type']
        track_length_km = track_info.get('length_km', 5.0)
        
        # 1. Fetch live or historical weather based on date and exact time (B)
        weather_data = get_track_weather(track_name, race_date, race_time)
        
        results = {
            "input_context": {
                "driver": driver,
                "team": norm_team,
                "year": self.year,
                "track": track_name,
                "track_category": track_type,
                "race_time": race_time,
                "weather": weather_data
            },
            "compounds": {}
        }
        
        # 2. Simulate for each compound
        for compound in self.valid_compounds:
            # Wets don't make sense if dry, and slicks don't make sense if heavy rain,
            # but we'll generate all to provide the full "what-if" mapping.
            max_laps = int(305 / track_length_km) + 2  # Full race distance + 2
            
            drop_off_curve, absolute_lap_times = self._simulate_compound(
                driver, norm_team, track_type, track_length_km, compound, weather_data, max_laps
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
