import json
from degradation_engine import TireDegradationSimulator
from sim_engine import StrategySimulator

def run_test_suite():
    print("========================================")
    print(" F1 TIRE STRATEGY SIMULATOR - TEST SUITE")
    print("========================================")
    
    # ---------------------------------------------------------
    # SCENARIO 1: Max Verstappen, Clean Air (P1) [ACTIVE AERO ERA]
    # ---------------------------------------------------------
    print("\n--- SCENARIO 1: Zandvoort 2026 / P1 / Clean Air ---")
    print("Booting ML Degradation Engine (Year: 2026 - Active Aero)...")
    deg_sim_2026 = TireDegradationSimulator(year=2026, force_jit_check=False) 
    
    print("Generating base degradation profiles...")
    out_s1 = deg_sim_2026.simulate(
        driver="VER", 
        team="Red Bull Racing", 
        track_name="Circuit Zandvoort", 
        race_date="2026-08-30", 
        race_time="15:00"
    )
    sim1 = StrategySimulator(out_s1)
    # 72 Laps, P1 (No dirty air multiplier), No Safety Car
    result_1 = sim1.generate_strategies(laps_to_complete=72, grid_pos=1)
    print("Best Strategy:", result_1["best_strategy"]["sequence"], "| Delta:", result_1["best_strategy"]["total_optimal_delta"])
    
    # ---------------------------------------------------------
    # SCENARIO 2: Lewis Hamilton, Dirty Air (P8) [GROUND EFFECT ERA]
    # ---------------------------------------------------------
    print("\n--- SCENARIO 2: Zandvoort 2024 / P8 / Dirty Air (1.05x wear) ---")
    print("Booting ML Degradation Engine (Year: 2024 - Ground Effect)...")
    deg_sim_2024 = TireDegradationSimulator(year=2024, force_jit_check=False) 
    
    out_s2 = deg_sim_2024.simulate(
        driver="HAM", 
        team="Scuderia Ferrari", 
        track_name="Circuit Zandvoort", 
        race_date="2024-08-25", 
        race_time="15:00"
    )
    sim2 = StrategySimulator(out_s2)
    # P8 means dirty air multiplier applies 
    result_2 = sim2.generate_strategies(laps_to_complete=72, grid_pos=8)
    print("Best Strategy:", result_2["best_strategy"]["sequence"], "| Delta:", result_2["best_strategy"]["total_optimal_delta"])

    # ---------------------------------------------------------
    # SCENARIO 3: Lando Norris, Safety Car Intervention [ACTIVE AERO ERA]
    # ---------------------------------------------------------
    print("\n--- SCENARIO 3: Zandvoort 2026 / P4 / Early Safety Car (Laps 10-15) ---")
    out_s3 = deg_sim_2026.simulate(
        driver="NOR", 
        team="McLaren", 
        track_name="Circuit Zandvoort", 
        race_date="2026-08-30", 
        race_time="15:00"
    )
    sim3 = StrategySimulator(out_s3)
    # P4, and a Safety Car on lap 10 lasting 5 laps (Wear reduced by 30% for those laps)
    result_3 = sim3.generate_strategies(laps_to_complete=72, grid_pos=4, sc_lap=10, sc_duration=5)
    print("Best Strategy:", result_3["best_strategy"]["sequence"], "| Delta:", result_3["best_strategy"]["total_optimal_delta"])
    
    print("\nDone! Feel free to run this file repeatedly or modify the parameters in `test_scenarios.py` to see how the engine responds.")

if __name__ == "__main__":
    run_test_suite()
