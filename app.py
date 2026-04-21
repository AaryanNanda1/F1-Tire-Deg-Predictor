from flask import Flask, request, jsonify
import sys
import traceback
from mappings import TRACK_PIT_LOSS, TEAM_MAPPING
from degradation_engine import TireDegradationSimulator
from sim_engine import StrategySimulator

app = Flask(__name__)

@app.route('/api/options', methods=['GET'])
def get_options():
    # Provide the dynamic options for the UI dropdowns
    drivers = ["VER", "NOR", "HAM", "RUS", "LEC", "SAI", "PIA", "PER", "ALO", "STR", "TSU", "RIC", "HUL", "MAG", "ALB", "SAR", "BOT", "ZHO", "GAS", "OCO"]
    
    return jsonify({
        "tracks": list(TRACK_PIT_LOSS.keys()),
        "teams": list(set(TEAM_MAPPING.values())),
        "drivers": drivers,
        "years": [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
    })

# Singleton engines to avoid reloading models on every request
_ENGINE_CACHE = {}

def get_engine(year):
    # Determine which era model to use
    era = "active_aero" if year >= 2026 else "ground_effect"
    if era not in _ENGINE_CACHE:
        _ENGINE_CACHE[era] = TireDegradationSimulator(year=year, force_jit_check=False)
    return _ENGINE_CACHE[era]

@app.route('/api/simulate', methods=['POST'])
def simulate_strategy():
    data = request.json
    try:
        year = int(data.get("year", 2026))
        driver = data.get("driver", "VER")
        team = data.get("team", "Red Bull Racing")
        track_name = data.get("track_name", "Circuit Zandvoort")
        grid_pos = int(data.get("grid_pos", 1))
        
        # Weather / Race Context overrides
        current_lap = int(data.get("current_lap", 0))
        # Total laps for the track (just defaulting to 72 if missing from UI for now, normally track specific)
        laps_to_complete = int(data.get("laps_to_complete", 72)) - current_lap
        if laps_to_complete <= 0:
            laps_to_complete = 1
            
        # Safety car params
        sc_lap = data.get("sc_lap", None)
        sc_duration = data.get("sc_duration", None)
        if sc_lap: sc_lap = int(sc_lap)
        if sc_duration: sc_duration = int(sc_duration)
        
        # Mocking Date & Time based on input year (Normally we'd lookup true F1 calendar dates)
        race_date = f"{year}-08-30" 
        race_time = "15:00"

        # 1. Get Cached ML Engine
        deg_sim = get_engine(year)
        out_degradation = deg_sim.simulate(
            driver=driver, 
            team=team, 
            track_name=track_name, 
            race_date=race_date, 
            race_time=race_time
        )
        
        # 2. Boot Math Strategy Simulator
        sim = StrategySimulator(out_degradation)
        strategies = sim.generate_strategies(
            laps_to_complete=laps_to_complete, 
            grid_pos=grid_pos, 
            sc_lap=sc_lap, 
            sc_duration=sc_duration
        )
        
        # Merge the outputs to send back to the UI
        # We need the curve mappings (out_degradation) for the Line Chart
        # We need the strategies (strategies) for the Bar Chart
        return jsonify({
            "status": "success",
            "degradation_graphs": out_degradation["compounds"],
            "strategies": strategies
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Flask runs on 5001 to avoid macOS AirPlay conflict on 5000
    app.run(port=5001, debug=True)
