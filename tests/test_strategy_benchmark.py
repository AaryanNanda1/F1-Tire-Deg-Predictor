import copy
import unittest

from strategy_benchmark import (
    BenchmarkValidationError,
    aggregate_reports,
    evaluate_race,
    evaluate_suite,
    validate_benchmark_suite,
)


def _benchmark_race():
    return {
        "id": "test-race",
        "split": "calibration",
        "season": 2024,
        "event_name": "Test Grand Prix",
        "track_name": "Test Circuit",
        "race_laps": 57,
        "simulation_input": {
            "driver": "TST",
            "team": "Test Team",
            "grid_pos": 1,
            "race_date": "2024-01-01",
            "race_time": "15:00",
        },
        "pirelli": {
            "compound_mapping": {
                "HARD": "C1",
                "MEDIUM": "C2",
                "SOFT": "C3",
            },
            "recommended_strategies": [
                {
                    "label": "fastest",
                    "stop_count": 2,
                    "compounds": ["SOFT", "SOFT", "MEDIUM"],
                    "stint_laps": [12, 20, 25],
                    "pit_windows": [[10, 14], [30, 35]],
                }
            ],
            "source_url": "https://press.pirelli.com/test-prerace/",
            "evidence": "Synthetic test fixture.",
        },
        "observed": {
            "reference_strategies": [
                {
                    "driver": "TST",
                    "stop_count": 2,
                    "compounds": ["SOFT", "SOFT", "MEDIUM"],
                    "stint_laps": [12, 20, 25],
                    "pit_windows": [[11, 13], [31, 33]],
                }
            ],
            "longest_stints": {
                "SOFT": {"laps": 15, "driver": "AAA"},
                "MEDIUM": {"laps": 30, "driver": "BBB"},
                "HARD": {"laps": 40, "driver": "CCC"},
            },
            "race_interruptions": [],
            "source_url": "https://press.pirelli.com/test-race/",
            "evidence": "Synthetic test fixture.",
        },
    }


def _model_output():
    return {
        "status": "success",
        "strategies": {
            "best_strategy": {
                "stops": 2,
                "expected_total_time_sec": 5000.0,
                "risk_adjusted_total_time_sec": 5000.0,
                "cost_breakdown": {
                    "degradation_cost_sec": 12.0,
                    "pit_loss_cost_sec": 44.0,
                },
                "max_performance_cliff_overshoot_laps": 2,
                "max_useful_life_overshoot_laps": 0,
                "stints_data": [
                    {"compound": "SOFT", "laps": 12, "start": 1, "end": 12},
                    {"compound": "SOFT", "laps": 20, "start": 13, "end": 32},
                    {"compound": "MEDIUM", "laps": 25, "start": 33, "end": 57},
                ],
            },
            "safe_strategy": {
                "stops": 1,
                "stints_data": [
                    {"compound": "MEDIUM", "laps": 25},
                    {"compound": "HARD", "laps": 32},
                ],
            },
            "risky_strategy": None,
        },
        "degradation_graphs": {
            "SOFT": {
                "strategy_useful_life_lap": 16,
                "performance_cliff_lap": 14,
                "cliff_confidence": "high",
                "cliff_method": "hybrid",
            },
            "MEDIUM": {
                "strategy_useful_life_lap": 28,
                "performance_cliff_lap": 20,
                "cliff_confidence": "medium",
                "cliff_method": "piecewise",
            },
            "HARD": {
                "strategy_useful_life_lap": 40,
                "performance_cliff_lap": None,
                "cliff_confidence": None,
                "cliff_method": "sustained",
            },
        },
    }


class StrategyBenchmarkTest(unittest.TestCase):
    def test_scores_strategy_and_useful_life(self):
        report = evaluate_race(_benchmark_race(), _model_output())

        self.assertEqual(report["split"], "calibration")
        self.assertTrue(report["strategy_metrics"]["best_pirelli_stop_count_match"])
        self.assertTrue(report["strategy_metrics"]["best_pirelli_exact_sequence_match"])
        self.assertTrue(report["strategy_metrics"]["best_pirelli_compound_set_match"])
        self.assertEqual(
            report["strategy_metrics"]["best_pirelli_pit_window_mae_laps"],
            0,
        )
        self.assertTrue(report["strategy_metrics"]["best_actual_exact_sequence_match"])
        self.assertEqual(
            report["model_strategies"][0]["cost_breakdown"][
                "degradation_cost_sec"
            ],
            12.0,
        )

        comparisons = {
            row["compound"]: row for row in report["tire_life_comparisons"]
        }
        self.assertEqual(
            comparisons["SOFT"]["useful_life_margin_vs_demonstrated_laps"],
            1,
        )
        self.assertEqual(
            comparisons["MEDIUM"]["useful_life_margin_vs_demonstrated_laps"],
            -2,
        )
        self.assertEqual(
            comparisons["HARD"]["useful_life_margin_vs_demonstrated_laps"],
            0,
        )
        self.assertEqual(
            comparisons["MEDIUM"]["useful_life_below_demonstrated_laps"],
            2,
        )
        self.assertFalse(
            comparisons["MEDIUM"]["useful_life_at_or_above_demonstrated"]
        )
        self.assertEqual(
            comparisons["SOFT"]["observed_stint_margin_beyond_cliff_laps"],
            1,
        )

        aggregate = aggregate_reports([report])
        self.assertEqual(aggregate["best_pirelli_exact_sequence_match_rate"], 1.0)
        self.assertEqual(aggregate["best_actual_compound_set_match_rate"], 1.0)
        self.assertEqual(
            aggregate["useful_life_at_or_above_demonstrated_rate"],
            0.667,
        )
        self.assertEqual(
            aggregate["useful_life_below_demonstrated_rate"],
            0.333,
        )
        self.assertEqual(
            aggregate["useful_life_below_demonstrated_mean_shortfall_laps"],
            2.0,
        )
        self.assertEqual(
            aggregate["useful_life_margin_vs_demonstrated_mean_laps"],
            -0.333,
        )
        self.assertEqual(aggregate["performance_cliff_context_observations"], 2)
        self.assertEqual(
            aggregate["observed_stint_margin_beyond_cliff_mean_laps"], 5.5
        )
        self.assertEqual(
            aggregate["predicted_cliff_before_demonstrated_stint_rate"],
            1.0,
        )

    def test_top_three_match_can_succeed_when_best_strategy_misses(self):
        model_output = _model_output()
        model_output["strategies"]["best_strategy"] = {
            "stops": 1,
            "stints_data": [
                {"compound": "MEDIUM", "laps": 25},
                {"compound": "HARD", "laps": 32},
            ],
        }
        model_output["strategies"]["safe_strategy"] = {
            "stops": 2,
            "stints_data": [
                {"compound": "SOFT", "laps": 12},
                {"compound": "SOFT", "laps": 20},
                {"compound": "MEDIUM", "laps": 25},
            ],
        }

        report = evaluate_race(_benchmark_race(), model_output)

        self.assertFalse(report["strategy_metrics"]["best_pirelli_stop_count_match"])
        self.assertTrue(report["strategy_metrics"]["top_three_pirelli_stop_count_match"])
        self.assertFalse(report["strategy_metrics"]["best_pirelli_exact_sequence_match"])
        self.assertTrue(report["strategy_metrics"]["top_three_pirelli_exact_sequence_match"])

    def test_partial_references_return_null_metrics(self):
        race = _benchmark_race()
        race["pirelli"]["recommended_strategies"] = [
            {
                "label": "stop-count-only",
                "stop_count": 2,
                "compounds": None,
                "stint_laps": None,
                "pit_windows": None,
            }
        ]

        report = evaluate_race(race, _model_output())

        self.assertTrue(report["strategy_metrics"]["best_pirelli_stop_count_match"])
        self.assertIsNone(
            report["strategy_metrics"]["best_pirelli_exact_sequence_match"]
        )
        self.assertIsNone(
            report["strategy_metrics"]["best_pirelli_pit_window_mae_laps"]
        )

    def test_rejects_duplicate_race_ids(self):
        race = _benchmark_race()
        suite = {
            "schema_version": 1,
            "races": [race, copy.deepcopy(race)],
        }

        with self.assertRaisesRegex(BenchmarkValidationError, "duplicate race id"):
            validate_benchmark_suite(suite)

    def test_rejects_invalid_absolute_compound(self):
        race = _benchmark_race()
        race["pirelli"]["compound_mapping"]["SOFT"] = "C6"
        suite = {"schema_version": 1, "races": [race]}

        with self.assertRaisesRegex(
            BenchmarkValidationError, "invalid Pirelli specification"
        ):
            validate_benchmark_suite(suite)

    def test_rejects_race_without_explicit_split(self):
        race = _benchmark_race()
        race.pop("split")

        with self.assertRaisesRegex(
            BenchmarkValidationError,
            "split must be calibration or holdout",
        ):
            validate_benchmark_suite({"schema_version": 1, "races": [race]})

    def test_evaluate_suite_requires_every_prediction(self):
        race = _benchmark_race()
        benchmark_suite = {
            "schema_version": 1,
            "suite_name": "Test",
            "races": [race],
        }
        predictions = {
            "schema_version": 1,
            "evaluation_mode": "production_diagnostic",
            "races": {},
        }

        with self.assertRaisesRegex(BenchmarkValidationError, "missing predictions"):
            evaluate_suite(benchmark_suite, predictions)


if __name__ == "__main__":
    unittest.main()
