import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
from datetime import date

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": os.environ.get(
                "CORS_ORIGINS",
                "https://f1-tire-deg.netlify.app,http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
        },
        r"/health": {"origins": "*"},
    },
)

_DEPLOY_METADATA = None

# FastF1 provides historical seasons as well as the active season, but future
# event data is not available. Keep the simulator aligned with that boundary
# instead of exposing placeholder future-season calendars.
MODEL_FIRST_YEAR = 2022
MODEL_LAST_YEAR = 2030


def get_available_simulation_years():
    """Return supported historical seasons through the active FastF1 season."""
    current_year = date.today().year
    last_available_year = min(current_year, MODEL_LAST_YEAR)
    if last_available_year < MODEL_FIRST_YEAR:
        return []
    return list(range(MODEL_FIRST_YEAR, last_available_year + 1))


def get_deploy_metadata():
    global _DEPLOY_METADATA
    if _DEPLOY_METADATA is not None:
        return _DEPLOY_METADATA

    commit = os.environ.get("RENDER_GIT_COMMIT")
    source = "render" if commit else "unknown"

    if not commit:
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            )
            commit = result.stdout.strip()
            source = "local-git"
        except Exception:
            commit = "unknown"

    _DEPLOY_METADATA = {
        "commit": commit,
        "commit_short": commit[:7] if commit != "unknown" else commit,
        "source": source,
    }
    return _DEPLOY_METADATA


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "f1-tire-deg-predictor",
        "version": get_deploy_metadata(),
    })

@app.route('/api/options', methods=['GET'])
def get_options():
    from mappings import TRACK_PIT_LOSS, TEAM_MAPPING, TRACK_CONFIG, get_roster_map, get_yearly_tracks

    available_years = get_available_simulation_years()

    # Provide the dynamic options for the UI dropdowns
    drivers = ["VER", "PER", "LEC", "SAI", "RUS", "HAM", "NOR", "RIC", "OCO", "ALO", "BOT", "ZHO", "VET", "STR", "HUL", "MAG", "MSC", "GAS", "TSU", "ALB", "LAT", "DEV", "PIA", "SAR", "LAW", "DOO", "BEA", "COL", "ANT", "BOR", "HAD", "LIN"]
    
    track_laps = {}
    for track, info in TRACK_CONFIG.items():
        track_laps[track] = info.get('race_laps', 57)

    # Load model metadata if it exists
    model_metadata = {}
    metadata_path = "models/era_training_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception as e:
            print(f"Error reading metadata: {e}")

    return jsonify({
        "tracks": list(TRACK_PIT_LOSS.keys()),
        "teams": list(set(TEAM_MAPPING.values())),
        "drivers": drivers,
        "years": available_years,
        "current_year": date.today().year,
        "compounds": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
        "track_laps": track_laps,
        "driver_roster": get_roster_map(available_years),
        "yearly_tracks": get_yearly_tracks(available_years),
        "model_metadata": model_metadata
    })

# Singleton engines to avoid reloading models on every request
_ENGINE_CACHE = {}
_ENGINE_MTIMES = {}

def get_engine(year):
    from degradation_engine import TireDegradationSimulator

    # Determine which era model to use
    era = "active_aero" if year >= 2026 else "ground_effect"
    prefix = "active_aero_2026_2030" if year >= 2026 else "ground_effect_2022_2025"
    model_file = f"models/{prefix}_model.joblib"
    
    current_mtime = os.path.getmtime(model_file) if os.path.exists(model_file) else 0
    
    if era not in _ENGINE_CACHE or _ENGINE_MTIMES.get(era, 0) < current_mtime:
        _ENGINE_CACHE[era] = TireDegradationSimulator(year=year, force_jit_check=False)
        _ENGINE_MTIMES[era] = current_mtime
        
    return _ENGINE_CACHE[era]

def _derive_weather_condition(weather_data: dict) -> str:
    """Derives weather condition category from degradation engine's weather output."""
    if not weather_data.get("rainfall", False):
        return "dry"
    
    # Check hourly forecasts for precipitation intensity
    hourly = weather_data.get("hourly_forecasts", [])
    if hourly:
        max_rain = max((h.get("rainfall_mm", 0) for h in hourly), default=0)
        if max_rain >= 5.0:
            return "heavy_wet"
        return "light_wet"
    
    # Fallback: check synopsis for hints
    synopsis = weather_data.get("synopsis", "").lower()
    if "heavy" in synopsis:
        return "heavy_wet"
    return "light_wet"

@app.route('/api/simulate', methods=['POST'])
def simulate_strategy():
    data = request.json
    try:
        year = int(data.get("year", 2026))
        available_years = get_available_simulation_years()
        if year not in available_years:
            current_year = date.today().year
            return jsonify({
                "status": "error",
                "message": (
                    f"Only seasons through {min(current_year, MODEL_LAST_YEAR)} can be simulated. "
                    "Future-season FastF1 race data is not available."
                ),
            }), 400
        driver = data.get("driver", "VER")
        team = data.get("team", "Red Bull Racing")
        track_name = data.get("track_name", "Circuit Zandvoort")
        
        # Race context
        current_lap = int(data.get("current_lap", 0))
        total_laps = int(data.get("laps_to_complete", 72))
        grid_pos = int(data.get("grid_pos", 1))
        
        # New: Current tire state
        current_compound = data.get("current_compound", None)
        laps_on_current_tire = int(data.get("laps_on_current_tire", 0))
        
        # New: Safety car context
        sc_happened_on_tire = bool(data.get("sc_happened_on_tire", False))
        sc_laps_on_tire = int(data.get("sc_laps_on_tire", 0))
        sc_currently_out = bool(data.get("sc_currently_out", False))
        
        # New: Pit history
        has_pitted = bool(data.get("has_pitted", False))
        compounds_used_count = max(0, int(data.get("compounds_used_count", 0) or 0))
        
        # New: Track position (distinct from grid_pos for mid-race sims)
        track_position = int(data.get("track_position", grid_pos))
        
        # New: Compounds already used (for 2-compound rule tracking)
        compounds_used = data.get("compounds_used", [])
        if not isinstance(compounds_used, list):
            compounds_used = []
        if current_compound and current_compound not in compounds_used:
            compounds_used.append(current_compound)
        compounds_used_count = max(compounds_used_count, len(set(compounds_used)))
        
        # Mocking Date & Time based on input year (Normally we'd lookup true F1 calendar dates)
        race_date = f"{year}-08-30" 
        race_time = "15:00"

        # 1. Get Cached ML Engine (degradation engine output consumed as-is)
        deg_sim = get_engine(year)
        out_degradation = deg_sim.simulate(
            driver=driver, 
            team=team, 
            track_name=track_name, 
            race_date=race_date, 
            race_time=race_time
        )
        
        # 2. Derive weather condition from degradation engine's weather data
        weather_data = out_degradation["input_context"].get("weather", {})
        weather_condition = data.get("weather_override", None) or _derive_weather_condition(weather_data)
        
        # 3. Boot Strategy Simulator (consumes degradation output unchanged)
        from sim_engine import StrategySimulator

        sim = StrategySimulator(out_degradation)
        strategies = sim.generate_strategies(
            total_laps=total_laps,
            current_lap=current_lap,
            current_compound=current_compound,
            laps_on_current_tire=laps_on_current_tire,
            sc_happened_on_tire=sc_happened_on_tire,
            sc_laps_on_tire=sc_laps_on_tire,
            sc_currently_out=sc_currently_out,
            has_pitted=has_pitted,
            track_position=track_position,
            grid_pos=grid_pos,
            weather_condition=weather_condition,
            compounds_used=compounds_used,
            compounds_used_count=compounds_used_count,
        )
        
        # Merge the outputs to send back to the UI
        # We need the curve mappings (out_degradation) for the Line Chart
        # We need the strategies (strategies) for the Bar Chart
        return jsonify({
            "status": "success",
            "degradation_graphs": out_degradation["compounds"],
            "strategies": strategies,
            "weather_condition": weather_condition,
            "weather_forecast": weather_data,
            "input_context": out_degradation["input_context"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
