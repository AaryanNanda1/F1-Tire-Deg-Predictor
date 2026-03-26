import os
import joblib
import numpy as np
import pandas as pd
from mappings import normalize_team_name, get_track_info
from weather_api import get_track_weather

class TireDegradationSimulator:
    def __init__(self, year: int, models_dir: str = "models"):
        self.year = year
        self.models_dir = models_dir
        
        # Determine correct era model
        if 2022 <= year <= 2025:
            prefix = "ground_effect_2022_2025"
        elif 2026 <= year <= 2030:
            prefix = "active_aero_2026_2030"
        else:
            raise ValueError(f"Year {year} falls outside supported era models (2022-2030)")
            
        model_path = os.path.join(models_dir, f"{prefix}_model.joblib")
        features_path = os.path.join(models_dir, f"{prefix}_features.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            raise FileNotFoundError(f"Missing model files for {prefix}. Run train_era_models.py.")
            
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        self.valid_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

    def _simulate_compound(self, driver, norm_team, track_type, track_length_km, compound, weather_data, max_laps=50):
        """Simulates lap times for a single compound from lap 1 to max_laps."""
        is_wet = 1 if compound in ["INTERMEDIATE", "WET"] or weather_data["rainfall"] else 0
        
        rows = []
        for age in range(1, max_laps + 1):
            rows.append({
                'Driver': driver,
                'Team': norm_team,
                'LapNumber': 15,  # Held constant to isolate pure tire degradation from fuel burn
                'TyreLife': age,
                'TyreLifeKM': age * track_length_km,
                'Compound': compound,
                'Stint': 1,
                'TrackType': track_type,
                'IsWet': is_wet,
                'AirTemp': weather_data['air_temp'],
                'TrackTemp': weather_data['track_temp'],
                'Humidity': weather_data['humidity'],
                'Rainfall': int(weather_data['rainfall']),
                'TeamBaselinePace': 100.0,
                'FieldBaselinePace': 100.0,
                'RelativePace': 0.0
            })
            
        df = pd.DataFrame(rows)
        # Dummy variables matching training
        categorical_cols = ['Driver', 'Team', 'Compound', 'TrackType']
        df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        final_input = df_dummies.reindex(columns=self.feature_names, fill_value=0)
        
        predictions = self.model.predict(final_input)
        
        # Calculate drop-off curve relative to the first lap on this tire
        base_lap = predictions[0]
        drop_off_curve = [float(p - base_lap) for p in predictions]
        
        # Prevent negative dropoffs (model noise) by making it monotonically increasing (or at least non-negative)
        for i in range(1, len(drop_off_curve)):
            drop_off_curve[i] = max(drop_off_curve[i-1], drop_off_curve[i])
            
        return drop_off_curve

    def _analyze_curve(self, drop_off_curve):
        """Extracts slope, cliff point, and lifespan from a degradation curve using the Kneedle algorithm."""
        n = len(drop_off_curve)
        
        # Expected lap time drop-off per lap (slope of linear part, approx laps 3 to 15 or end)
        eval_end = min(15, n - 1)
        if eval_end > 3:
            slope = (drop_off_curve[eval_end] - drop_off_curve[3]) / (eval_end - 3)
        else:
            slope = drop_off_curve[-1] / n
            
        slope = max(0.01, slope) # Prevent zero/negative slope
        
        # Cliff Point: point of maximum distance from a line connecting start and end
        # (Simplified Kneedle algorithm)
        x1, y1 = 1.0, float(drop_off_curve[0])
        x2, y2 = float(n), float(drop_off_curve[-1])
        denom = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        max_dist = -1
        cliff_point = n
        
        for i in range(2, n - 2): # Exclude very early or very late cliffs
            x0, y0 = float(i + 1), float(drop_off_curve[i])
            # Perpendicular distance from (x0, y0) to line (x1, y1)-(x2, y2)
            if denom > 0:
                dist = np.abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) / denom
            else:
                dist = 0.0

            
            # We want the point where the curve sags the most *below* the line (meaning rapid rise after)
            # For degradation, the curve usually bows downwards then shoots up.
            if dist > max_dist:
                max_dist = dist
                cliff_point = i + 1
                
        # Fallbacks if cliff is unrealistic
        if cliff_point >= n - 1:
            cliff_point = int(n * 0.8)
            
        # Suggested lifespan (safely pit before cliff, delta up to 5 laps)
        suggested_lifespan = max(1, cliff_point - 3)
        
        return {
            "drop_off_per_lap_sec": round(float(slope), 3),
            "cliff_point_lap": cliff_point,
            "suggested_lifespan": f"{suggested_lifespan}-{suggested_lifespan + 4} laps",
            "graph_data": {lap: round(val, 3) for lap, val in enumerate(drop_off_curve, start=1)}
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
            max_laps = int(305 / track_length_km) + 2 # Full race distance + 2
            
            curve = self._simulate_compound(
                driver, norm_team, track_type, track_length_km, compound, weather_data, max_laps
            )
            
            analysis = self._analyze_curve(curve)
            results["compounds"][compound] = analysis
            
        return results

if __name__ == "__main__":
    # Quick Test
    import json
    sim = TireDegradationSimulator(year=2026)
    out = sim.simulate("VER", "Red Bull Racing", "Bahrain International Circuit", "2026-03-05", "18:00")
    print(json.dumps(out["input_context"]["weather"], indent=2))
    print(json.dumps(out["compounds"]["SOFT"], indent=2))
