import unittest

from sim_engine import StrategySimulator


class StrategySimulatorWetToDryTest(unittest.TestCase):
    def _build_profiles(self):
        return {
            "compounds": {
                "WET": {
                    "graph_data": {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
                "INTERMEDIATE": {
                    "graph_data": {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
                "SOFT": {
                    "graph_data": {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
                "MEDIUM": {
                    "graph_data": {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
                "HARD": {
                    "graph_data": {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
            },
            "input_context": {"track": "Monaco"},
        }

    def test_penalizes_wet_tires_on_dry_track(self):
        simulator = StrategySimulator(self._build_profiles())

        wet_delta, _ = simulator._eval_stint("WET", 2, 1, weather_condition="heavy_wet")
        dry_delta, _ = simulator._eval_stint("WET", 2, 1, weather_condition="dry")

        self.assertAlmostEqual(wet_delta, 200.0)
        self.assertAlmostEqual(dry_delta, 212.25)

    def test_penalizes_intermediates_less_than_full_wets_on_dry_track(self):
        simulator = StrategySimulator(self._build_profiles())

        wet_delta, _ = simulator._eval_stint("WET", 2, 1, weather_condition="dry")
        intermediate_delta, _ = simulator._eval_stint("INTERMEDIATE", 2, 1, weather_condition="dry")

        self.assertAlmostEqual(intermediate_delta, 206.12)
        self.assertLess(intermediate_delta, wet_delta)

    def test_allows_switch_from_wet_to_single_dry_compound_in_dry_conditions(self):
        profiles = {
            "compounds": {
                "WET": {
                    "graph_data": {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
                "SOFT": {
                    "graph_data": {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
                "MEDIUM": {
                    "graph_data": {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
                "HARD": {
                    "graph_data": {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 4,
                    "strategy_useful_life_lap": 4,
                },
            },
            "input_context": {"track": "Monaco"},
        }

        simulator = StrategySimulator(profiles)
        result = simulator.generate_strategies(
            total_laps=20,
            current_lap=10,
            current_compound="WET",
            laps_on_current_tire=10,
            sc_happened_on_tire=True,
            sc_laps_on_tire=1,
            sc_currently_out=True,
            has_pitted=False,
            track_position=5,
            grid_pos=5,
            weather_condition="dry",
            compounds_used=["WET"],
        )

        self.assertIsNotNone(result["best_strategy"])


if __name__ == "__main__":
    unittest.main()
