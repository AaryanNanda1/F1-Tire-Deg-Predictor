from flask import Flask, request, jsonify
import sys
import traceback
from mappings import TRACK_PIT_LOSS, TEAM_MAPPING, TRACK_CONFIG, get_roster_map
import math
from degradation_engine import TireDegradationSimulator
from sim_engine import StrategySimulator

app = Flask(__name__)

@app.route('/api/options', methods=['GET'])
def get_options():
    # Provide the dynamic options for the UI dropdowns
    drivers = ["VER", "PER", "LEC", "SAI", "RUS", "HAM", "NOR", "RIC", "OCO", "ALO", "BOT", "ZHO", "VET", "STR", "HUL", "MAG", "MSC", "GAS", "TSU", "ALB", "LAT", "DEV", "PIA", "SAR", "LAW", "DOO", "BEA", "COL", "ANT", "BOR", "HAD", "LIN"]
    
    track_laps = {}
    for track, info in TRACK_CONFIG.items():
        length = info.get("length_km", 5.0)
        target_km = 260.0 if "Monaco" in track else 305.0
        track_laps[track] = math.ceil(target_km / length)

    return jsonify({
        "tracks": list(TRACK_PIT_LOSS.keys()),
        "teams": list(set(TEAM_MAPPING.values())),
        "drivers": drivers,
        "years": [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030],
        "compounds": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
        "track_laps": track_laps,
        "driver_roster": get_roster_map()
    })

# Singleton engines to avoid reloading models on every request
_ENGINE_CACHE = {}

def get_engine(year):
    # Determine which era model to use
    era = "active_aero" if year >= 2026 else "ground_effect"
    if era not in _ENGINE_CACHE:
        _ENGINE_CACHE[era] = TireDegradationSimulator(year=year, force_jit_check=False)
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
        
        # New: Track position (distinct from grid_pos for mid-race sims)
        track_position = int(data.get("track_position", grid_pos))
        
        # New: Compounds already used (for 2-compound rule tracking)
        compounds_used = data.get("compounds_used", [])
        if current_compound and current_compound not in compounds_used:
            compounds_used.append(current_compound)
        
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
            compounds_used=compounds_used
        )
        
        # Merge the outputs to send back to the UI
        # We need the curve mappings (out_degradation) for the Line Chart
        # We need the strategies (strategies) for the Bar Chart
        return jsonify({
            "status": "success",
            "degradation_graphs": out_degradation["compounds"],
            "strategies": strategies,
            "weather_condition": weather_condition,
            "weather_forecast": weather_data
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Flask runs on 5001 to avoid macOS AirPlay conflict on 5000
    app.run(port=5001, debug=True)
