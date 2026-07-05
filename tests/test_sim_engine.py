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

    def _build_ranked_profiles(self):
        return {
            "compounds": {
                "WET": {
                    "graph_data": {lap: 200.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "INTERMEDIATE": {
                    "graph_data": {lap: 0.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "SOFT": {
                    "graph_data": {lap: 50.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "MEDIUM": {
                    "graph_data": {lap: 0.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "HARD": {
                    "graph_data": {lap: 60.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
            },
            "input_context": {"track": "Monaco"},
        }

    def _build_full_wet_preferred_profiles(self):
        return {
            "compounds": {
                "WET": {
                    "graph_data": {lap: 0.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "INTERMEDIATE": {
                    "graph_data": {lap: 100.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "SOFT": {
                    "graph_data": {lap: 120.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "MEDIUM": {
                    "graph_data": {lap: 120.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
                },
                "HARD": {
                    "graph_data": {lap: 120.0 for lap in range(1, 21)},
                    "drop_off_per_lap_sec": 1.0,
                    "cliff_point_lap": 20,
                    "strategy_useful_life_lap": 20,
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

    def test_penalizes_dry_tires_on_light_wet_track_by_compound_hardness(self):
        simulator = StrategySimulator(self._build_profiles())

        soft_delta, _ = simulator._eval_stint("SOFT", 2, 1, weather_condition="light_wet")
        medium_delta, _ = simulator._eval_stint("MEDIUM", 2, 1, weather_condition="light_wet")
        hard_delta, _ = simulator._eval_stint("HARD", 2, 1, weather_condition="light_wet")

        self.assertAlmostEqual(soft_delta, 200.7)
        self.assertAlmostEqual(medium_delta, 200.9)
        self.assertAlmostEqual(hard_delta, 201.1)
        self.assertLess(soft_delta, medium_delta)
        self.assertLess(medium_delta, hard_delta)

    def test_heavy_wet_dry_tire_penalty_is_larger_than_light_wet(self):
        simulator = StrategySimulator(self._build_profiles())

        light_wet_delta, _ = simulator._eval_stint("HARD", 2, 1, weather_condition="light_wet")
        heavy_wet_delta, _ = simulator._eval_stint("HARD", 2, 1, weather_condition="heavy_wet")

        self.assertAlmostEqual(heavy_wet_delta - light_wet_delta, 2.2)

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

    def test_rejects_same_dry_compound_without_wet_tires(self):
        simulator = StrategySimulator(self._build_profiles())

        self.assertFalse(simulator._satisfies_compound_rule([], ["MEDIUM", "MEDIUM"]))
        self.assertFalse(simulator._satisfies_compound_rule(["SOFT"], ["SOFT"]))

    def test_allows_same_compound_when_all_wet_or_intermediate(self):
        simulator = StrategySimulator(self._build_profiles())

        self.assertTrue(simulator._satisfies_compound_rule([], ["INTERMEDIATE", "INTERMEDIATE"]))
        self.assertTrue(simulator._satisfies_compound_rule([], ["WET", "WET"]))

    def test_allows_two_dry_compounds_without_wet_tires(self):
        simulator = StrategySimulator(self._build_profiles())

        self.assertTrue(simulator._satisfies_compound_rule([], ["MEDIUM", "HARD"]))

    def test_light_wet_strategy_does_not_return_same_dry_compound_only(self):
        simulator = StrategySimulator(self._build_ranked_profiles())

        result = simulator.generate_strategies(
            total_laps=10,
            current_lap=0,
            grid_pos=1,
            weather_condition="light_wet",
        )

        best_compounds = [s["compound"] for s in result["best_strategy"]["stints_data"]]
        self.assertFalse(
            len(set(best_compounds)) == 1
            and best_compounds[0] in {"SOFT", "MEDIUM", "HARD"}
        )

    def test_light_wet_allows_controlled_full_wet_transitions(self):
        simulator = StrategySimulator(self._build_profiles())

        self.assertTrue(
            simulator._is_light_wet_compound_sequence_allowed(["WET", "INTERMEDIATE", "SOFT"])
        )
        self.assertTrue(
            simulator._is_light_wet_compound_sequence_allowed(["INTERMEDIATE", "WET", "SOFT"])
        )
        self.assertTrue(
            simulator._is_light_wet_compound_sequence_allowed(["SOFT", "INTERMEDIATE", "WET"])
        )

    def test_light_wet_rejects_arbitrary_full_wet_transitions(self):
        simulator = StrategySimulator(self._build_profiles())

        self.assertFalse(
            simulator._is_light_wet_compound_sequence_allowed(["WET", "SOFT"])
        )
        self.assertFalse(
            simulator._is_light_wet_compound_sequence_allowed(["SOFT", "WET", "HARD"])
        )
        self.assertFalse(
            simulator._is_light_wet_compound_sequence_allowed(["WET", "WET"])
        )

    def test_light_wet_strategy_can_return_controlled_full_wet_transition(self):
        simulator = StrategySimulator(self._build_full_wet_preferred_profiles())

        result = simulator.generate_strategies(
            total_laps=10,
            current_lap=0,
            grid_pos=1,
            weather_condition="light_wet",
        )

        best_compounds = [s["compound"] for s in result["best_strategy"]["stints_data"]]
        self.assertIn("WET", best_compounds)
        self.assertTrue(simulator._is_light_wet_compound_sequence_allowed(best_compounds))

    def test_all_intermediate_race_strategy_is_valid_with_physical_stop(self):
        simulator = StrategySimulator(self._build_ranked_profiles())

        result = simulator.generate_strategies(
            total_laps=10,
            current_lap=0,
            grid_pos=1,
            weather_condition="heavy_wet",
        )

        best = result["best_strategy"]
        best_compounds = [s["compound"] for s in best["stints_data"]]
        self.assertEqual(best["stops"], 1)
        self.assertEqual(set(best_compounds), {"INTERMEDIATE"})


if __name__ == "__main__":
    unittest.main()
