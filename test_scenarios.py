import json
from degradation_engine import TireDegradationSimulator
from sim_engine import StrategySimulator

def run_test_suite():
    print("========================================")
    print(" F1 TIRE STRATEGY SIMULATOR - TEST SUITE")
    print("========================================")
    
    # ---------------------------------------------------------
    # SCENARIO 1: Max Verstappen, Clean Air (P1) [ACTIVE AERO ERA]
    # Pre-race start — backward compatibility
    # ---------------------------------------------------------
    print("\n--- SCENARIO 1: Zandvoort 2026 / P1 / Clean Air (Pre-Race) ---")
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
    # 72 Laps, P1 (No dirty air), Pre-race
    result_1 = sim1.generate_strategies(total_laps=72, grid_pos=1)
    print("Optimal:", result_1["best_strategy"]["sequence"], "| Delta:", result_1["best_strategy"]["total_optimal_delta"])
    print("Safe:   ", result_1["safe_strategy"]["sequence"] if result_1["safe_strategy"] else "N/A")
    print("Risky:  ", result_1["risky_strategy"]["sequence"] if result_1["risky_strategy"] else "N/A")
    
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
    result_2 = sim2.generate_strategies(total_laps=72, grid_pos=8)
    print("Optimal:", result_2["best_strategy"]["sequence"], "| Delta:", result_2["best_strategy"]["total_optimal_delta"])
    print("Safe:   ", result_2["safe_strategy"]["sequence"] if result_2["safe_strategy"] else "N/A")
    print("Risky:  ", result_2["risky_strategy"]["sequence"] if result_2["risky_strategy"] else "N/A")

    # ---------------------------------------------------------
    # SCENARIO 3: Lando Norris, Safety Car Intervention [ACTIVE AERO ERA]
    # Mid-race: Lap 25, on Medium for 20 laps, SC happened on this tire
    # ---------------------------------------------------------
    print("\n--- SCENARIO 3: Zandvoort 2026 / Mid-Race (L25) / SC on Current Tire ---")
    out_s3 = deg_sim_2026.simulate(
        driver="NOR", 
        team="McLaren", 
        track_name="Circuit Zandvoort", 
        race_date="2026-08-30", 
        race_time="15:00"
    )
    sim3 = StrategySimulator(out_s3)
    result_3 = sim3.generate_strategies(
        total_laps=72, 
        current_lap=25,
        current_compound="MEDIUM",
        laps_on_current_tire=20,
        sc_happened_on_tire=True,
        sc_laps_on_tire=3,
        has_pitted=True,
        track_position=4,
        compounds_used=["SOFT", "MEDIUM"]
    )
    print("Optimal:", result_3["best_strategy"]["sequence"] if result_3["best_strategy"] else "N/A")
    print("Safe:   ", result_3["safe_strategy"]["sequence"] if result_3["safe_strategy"] else "N/A")
    print("Risky:  ", result_3["risky_strategy"]["sequence"] if result_3["risky_strategy"] else "N/A")
    
    # ---------------------------------------------------------
    # SCENARIO 4: SC Currently Active — Risky pitting opportunity
    # Lap 30, on Hard for 25 laps, P12 (outside points), SC just deployed
    # ---------------------------------------------------------
    print("\n--- SCENARIO 4: Zandvoort 2026 / SC Active / Risky Pit (P12) ---")
    out_s4 = deg_sim_2026.simulate(
        driver="PER", 
        team="Red Bull Racing", 
        track_name="Circuit Zandvoort", 
        race_date="2026-08-30", 
        race_time="15:00"
    )
    sim4 = StrategySimulator(out_s4)
    result_4 = sim4.generate_strategies(
        total_laps=72,
        current_lap=30,
        current_compound="HARD",
        laps_on_current_tire=25,
        sc_currently_out=True,
        has_pitted=True,
        track_position=12,
        compounds_used=["MEDIUM", "HARD"]
    )
    print("Optimal:", result_4["best_strategy"]["sequence"] if result_4["best_strategy"] else "N/A")
    print("Safe:   ", result_4["safe_strategy"]["sequence"] if result_4["safe_strategy"] else "N/A")
    print("Risky:  ", result_4["risky_strategy"]["sequence"] if result_4["risky_strategy"] else "N/A")
    
    # ---------------------------------------------------------
    # SCENARIO 5: Has NOT pitted yet — must pit at least once
    # Lap 15 on Soft tire, 15 laps old, P3
    # ---------------------------------------------------------
    print("\n--- SCENARIO 5: Zandvoort 2026 / Must Pit (No Stop Yet) ---")
    result_5 = sim1.generate_strategies(
        total_laps=72,
        current_lap=15,
        current_compound="SOFT",
        laps_on_current_tire=15,
        has_pitted=False,
        track_position=3,
        compounds_used=["SOFT"]
    )
    print("Optimal:", result_5["best_strategy"]["sequence"] if result_5["best_strategy"] else "N/A")
    print("Safe:   ", result_5["safe_strategy"]["sequence"] if result_5["safe_strategy"] else "N/A")
    print("Risky:  ", result_5["risky_strategy"]["sequence"] if result_5["risky_strategy"] else "N/A")
    
    print("\nDone! All 5 scenarios completed.")

if __name__ == "__main__":
    run_test_suite()
