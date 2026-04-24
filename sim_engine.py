import itertools
import math

from mappings import TRACK_PIT_LOSS

# Default Constants
FUEL_BURN_PER_LAP = 0.07
DIRTY_AIR_WEAR_MULTIPLIER = 1.05
SC_WEAR_MULTIPLIER = 0.70

class StrategySimulator:
    def __init__(self, degradation_profiles: dict):
        """
        Takes the output dictionary from TireDegradationSimulator.simulate()
        """
        self.profiles = degradation_profiles["compounds"]
        self.context = degradation_profiles["input_context"]
        
    def _get_deg_time(self, compound: str, effective_age: float):
        """Linearly interpolates degradation time from the ML curve based on effective tire age."""
        graph = self.profiles[compound]["graph_data"]
        # Convert string keys from JSON back to int
        graph = {int(k): v for k, v in graph.items()}
        
        max_age = max(graph.keys())
        if effective_age <= 1.0:
            return graph[1]
        if effective_age >= max_age:
            # If we blow past the curve data, extrapolate aggressively using the cliff slope
            overshoot = effective_age - max_age
            return graph[max_age] + (overshoot * self.profiles[compound]["drop_off_per_lap_sec"] * 2.0)
            
        lower_age = math.floor(effective_age)
        upper_age = math.ceil(effective_age)
        
        if lower_age == upper_age:
            return graph.get(lower_age, 0.0)
            
        # Interpolate
        lower_val = graph.get(lower_age, 0.0)
        upper_val = graph.get(upper_age, 0.0)
        fraction = effective_age - lower_age
        
        return lower_val + (upper_val - lower_val) * fraction

    def _eval_strategy(self, compounds: list, stint_lengths: list, grid_pos: int, laps_to_complete: int, start_lap: int=1, start_age: float=0.0, sc_lap: int=None, sc_duration: int=None):
        """Evaluates total race delta time for a specific combination of stints."""
        total_deg_delta = 0.0
        max_cliff_overshoot = 0
        
        current_lap = start_lap
        
        for idx, (compound, length) in enumerate(zip(compounds, stint_lengths)):
            # If this is the starting tire, carries over its previous age
            effective_age = start_age if idx == 0 else 0.0
            # Prefer the robust recommended_max_life from tire_life_analysis;
            # fall back to cliff_point_lap for backward compatibility.
            cliff_point = self.profiles[compound].get("recommended_max_life",
                          self.profiles[compound]["cliff_point_lap"])
            
            for _ in range(length):
                wear_factor = DIRTY_AIR_WEAR_MULTIPLIER if grid_pos > 1 else 1.0
                
                # Apply 0.70x wear modifier if laps fall under the Safety Car window
                if sc_lap and sc_duration:
                    if sc_lap <= current_lap < (sc_lap + sc_duration):
                        wear_factor *= SC_WEAR_MULTIPLIER
                
                effective_age += wear_factor
                total_deg_delta += self._get_deg_time(compound, effective_age)
                current_lap += 1
                
            overshoot = max(0, effective_age - cliff_point)
            if overshoot > max_cliff_overshoot:
                max_cliff_overshoot = overshoot
                
        # Total optimization time = Deg + Pit losses
        # (Fuel burn is constant across all strategies, so we ignore it for ranking purposes)
        num_stops = len(compounds) - 1
        track_name = self.context["track"]
        pit_loss_sec = TRACK_PIT_LOSS.get(track_name, 24.0)
        total_time_delta = total_deg_delta + (num_stops * pit_loss_sec)
        
        return {
            "compounds": compounds,
            "stints": stint_lengths,
            "stops": num_stops,
            "total_delta": total_time_delta,
            "max_cliff_overshoot": max_cliff_overshoot
        }

    def generate_strategies(self, laps_to_complete: int, grid_pos: int, start_compound: str = None, start_age: float = 0.0, sc_lap: int=None, sc_duration: int=None):
        """
        Brute forces 1, 2, and 3 stop strategies to find the mathematical optimum.
        Categorizes them into Best, Safe, and Aggressive.
        """
        valid_compounds = ["SOFT", "MEDIUM", "HARD"]
        all_evaluations = []
        
        # Helper to generate stint length combinations that sum to laps_to_complete
        def get_stint_combos(num_stints, total_laps, min_laps=8, step=2):
            if num_stints == 1:
                yield [total_laps]
            elif num_stints == 2:
                for i in range(min_laps, total_laps - min_laps + 1, step):
                    yield [i, total_laps - i]
            elif num_stints == 3:
                for i in range(min_laps, total_laps - (min_laps*2) + 1, step):
                    for j in range(min_laps, total_laps - i - min_laps + 1, step):
                        yield [i, j, total_laps - i - j]
            elif num_stints == 4: # 3 stops
                # For 4 stints, even larger steps can be used to keep UI snappy
                s = step if total_laps < 50 else step * 2 
                for i in range(min_laps, total_laps - (min_laps*3) + 1, s):
                    for j in range(min_laps, total_laps - i - (min_laps*2) + 1, s):
                        for k in range(min_laps, total_laps - i - j - min_laps + 1, s):
                            yield [i, j, k, total_laps - i - j - k]
                            
        # Test 1, 2, and 3 stop setups
        for stops in [1, 2, 3]:
            num_stints = stops + 1
            
            # Generate all compound permutations (e.g. SOFT->HARD, or if start_compound is forced)
            if start_compound:
                compound_pools = [[start_compound]] + [valid_compounds] * (stops)
                compound_combos = list(itertools.product(*compound_pools))
            else:
                compound_combos = list(itertools.product(valid_compounds, repeat=num_stints))
                
            for combo in compound_combos:
                # F1 Rule: Must use at least 2 different compounds in a dry race
                if len(set(combo)) < 2:
                    continue
                    
                for lengths in get_stint_combos(num_stints, laps_to_complete):
                    eval_result = self._eval_strategy(combo, lengths, grid_pos, laps_to_complete, start_age=start_age, sc_lap=sc_lap, sc_duration=sc_duration)
                    all_evaluations.append(eval_result)
                    
        # Sort all by total race time delta (fastest first)
        all_evaluations.sort(key=lambda x: x["total_delta"])
        
        # Categorize
        best_strategy = all_evaluations[0] if all_evaluations else None
        
        safe_strategies = [s for s in all_evaluations if s["max_cliff_overshoot"] <= 0]
        safer_alternative = safe_strategies[0] if safe_strategies else None
        
        # Aggressive: Faster pace but pushes a tire up to 5 laps past the cliff point
        agg_strategies = [s for s in all_evaluations if 1 <= s["max_cliff_overshoot"] <= 5]
        aggressive_alternative = agg_strategies[0] if agg_strategies else None

        return {
            "best_strategy": self._format_output(best_strategy),
            "safer_alternative": self._format_output(safer_alternative),
            "aggressive_alternative": self._format_output(aggressive_alternative)
        }
        
    def _format_output(self, strat):
        if not strat:
            return None
            
        sequence_labels = []
        stints_data = []
        current_lap = 1
        for comp, length in zip(strat["compounds"], strat["stints"]):
            end_lap = current_lap + length - 1
            sequence_labels.append(f"{comp} [L{current_lap} - L{end_lap}]")
            stints_data.append({
                "compound": comp,
                "laps": length,
                "start": current_lap,
                "end": end_lap
            })
            current_lap = end_lap + 1
            
        return {
            "stops": strat["stops"],
            "sequence": " -> ".join(sequence_labels),
            "stints_data": stints_data,
            "total_optimal_delta": round(strat["total_delta"], 2),
            "risk_cliff_overshoot": round(strat["max_cliff_overshoot"], 1)
        }

if __name__ == "__main__":
    import json
    from degradation_engine import TireDegradationSimulator
    
    # Quick Test
    print("Initializing Degradation Engine...")
    deg_sim = TireDegradationSimulator(year=2026) 
    
    print("Generating profiles for Zandvoort (72 laps) for Max Verstappen (P3)...")
    # Zandvoort 2026 race date placeholder, using 15:00 local time
    out = deg_sim.simulate("VER", "Red Bull Racing", "Circuit Zandvoort", "2026-08-30", "15:00")
    
    sim = StrategySimulator(out)
    result = sim.generate_strategies(laps_to_complete=72, grid_pos=3)
    
    print("\n=== STRATEGY RESULTS ===")
    print(json.dumps(result, indent=2))
