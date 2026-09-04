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

        self.assertAlmostEqual(soft_delta, 201.5)
        self.assertAlmostEqual(medium_delta, 202.0)
        self.assertAlmostEqual(hard_delta, 202.5)
        self.assertLess(soft_delta, medium_delta)
        self.assertLess(medium_delta, hard_delta)

    def test_heavy_wet_dry_tire_penalty_is_larger_than_light_wet(self):
        simulator = StrategySimulator(self._build_profiles())

        light_wet_delta, _ = simulator._eval_stint("HARD", 2, 1, weather_condition="light_wet")
        heavy_wet_delta, _ = simulator._eval_stint("HARD", 2, 1, weather_condition="heavy_wet")

        self.assertAlmostEqual(heavy_wet_delta - light_wet_delta, 5.0)

    def test_strategy_cost_components_sum_to_expected_total(self):
        profiles = self._build_profiles()
        for profile in profiles["compounds"].values():
            profile["performance_cliff_lap"] = 2
            profile["cliff_confidence"] = "high"
            profile["strategy_confidence"] = "high"
        simulator = StrategySimulator(profiles)

        strategy = simulator._eval_strategy(
            ["SOFT", "MEDIUM"],
            [5, 5],
            position=1,
            laps_to_complete=10,
            weather_condition="dry",
        )

        component_total = sum(
            strategy[field]
            for field in (
                "base_lap_time_cost_sec",
                "degradation_cost_sec",
                "weather_mismatch_cost_sec",
                "pit_loss_cost_sec",
                "traffic_cost_sec",
                "cliff_risk_cost_sec",
            )
        )
        self.assertAlmostEqual(component_total, strategy["total_delta"])
        self.assertEqual(
            strategy["expected_total_time_sec"],
            strategy["risk_adjusted_total_time_sec"],
        )
        self.assertGreater(
            strategy["max_performance_cliff_overshoot_laps"], 0
        )
        self.assertGreater(strategy["max_useful_life_overshoot_laps"], 0)

    def test_formatted_strategy_exposes_stint_diagnostics(self):
        profiles = self._build_ranked_profiles()
        for profile in profiles["compounds"].values():
            profile["performance_cliff_lap"] = 6
            profile["cliff_confidence"] = "medium"
            profile["strategy_confidence"] = "high"
        simulator = StrategySimulator(profiles)

        result = simulator.generate_strategies(
            total_laps=10,
            grid_pos=1,
            weather_condition="dry",
            include_candidate_diagnostics=True,
        )

        best = result["best_strategy"]
        self.assertEqual(best["strategy_role"], "mathematical_fastest")
        self.assertEqual(best["score_rank"], 1)
        self.assertIn("cost_breakdown", best)
        self.assertIn("performance_cliff_lap", best["stints_data"][0])
        self.assertEqual(len(result["candidate_diagnostics"]), 5)

    def test_missing_performance_cliff_is_reported_as_unknown(self):
        simulator = StrategySimulator(self._build_profiles())

        strategy = simulator._eval_strategy(
            ["SOFT", "MEDIUM"],
            [5, 5],
            position=1,
            laps_to_complete=10,
            weather_condition="dry",
        )

        self.assertIsNone(
            strategy["max_performance_cliff_overshoot_laps"]
        )

    def test_useful_life_bounds_drive_safe_and_risky_caps(self):
        profiles = self._build_profiles()
        for profile in profiles["compounds"].values():
            profile.update(
                strategy_useful_life_lap=10,
                strategy_useful_life_lower=8,
                strategy_useful_life_upper=12,
                strategy_useful_life_uncertainty_laps=2,
            )
        simulator = StrategySimulator(profiles)
        self.assertEqual(simulator._get_stint_cap("SOFT", "safe"), 8)
        self.assertEqual(simulator._get_stint_cap("SOFT", "risky"), 12)
        self.assertLessEqual(simulator._get_stint_cap("SOFT", "risky"), 25)

    def test_formatted_output_contains_useful_life_window_and_pit_window(self):
        profiles = self._build_ranked_profiles()
        for profile in profiles["compounds"].values():
            profile.update(
                strategy_useful_life_lap=8,
                strategy_useful_life_lower=7,
                strategy_useful_life_upper=9,
                strategy_useful_life_uncertainty_laps=1,
                strategy_useful_life_confidence="high",
            )
        simulator = StrategySimulator(profiles)
        strategy = simulator._eval_strategy(
            ["SOFT", "MEDIUM"], [10, 10], position=1,
            laps_to_complete=20,
        )
        formatted = simulator._format_output(strategy, start_lap=0, strategy_role="safe")
        stint = formatted["stints_data"][0]
        self.assertEqual(stint["strategy_useful_life_lower"], 7)
        self.assertEqual(stint["strategy_useful_life_lap"], 8)
        self.assertEqual(stint["strategy_useful_life_upper"], 9)
        self.assertEqual(stint["recommended_pit_window_start"], 7)
        self.assertEqual(stint["recommended_pit_window_end"], 9)

    def test_interval_does_not_multiply_candidate_search(self):
        before = self._build_ranked_profiles()
        after = self._build_ranked_profiles()
        for profile in after["compounds"].values():
            profile.update(strategy_useful_life_lower=17, strategy_useful_life_upper=20, strategy_useful_life_uncertainty_laps=3)
        first = StrategySimulator(before)
        second = StrategySimulator(after)
        first.generate_strategies(total_laps=20, grid_pos=1)
        second.generate_strategies(total_laps=20, grid_pos=1)
        self.assertEqual(first.last_strategy_searches, second.last_strategy_searches)
        self.assertEqual(first.last_candidate_count, second.last_candidate_count)

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

    def test_distinct_compound_count_satisfies_dry_compound_rule(self):
        simulator = StrategySimulator(self._build_profiles())

        self.assertFalse(
            simulator._satisfies_compound_rule(
                ["MEDIUM"],
                ["MEDIUM"],
                compounds_used_count=1,
            )
        )
        self.assertTrue(
            simulator._satisfies_compound_rule(
                ["MEDIUM"],
                ["MEDIUM"],
                compounds_used_count=2,
            )
        )

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
