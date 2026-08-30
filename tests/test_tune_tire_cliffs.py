import unittest

from scripts.tune_tire_cliffs import (
    MINIMUM_CLIFF_RECALL,
    _objective,
    parameter_grids,
    run_search,
)
from tire_life_analysis import DEFAULT_CONFIG


def _capture(reference_id, status, reviewed_cliff_lap=None):
    return {
        "reference_id": reference_id,
        "split": "calibration",
        "starting_tire_age": 1,
        "ending_tire_age": 12,
        "manual_review_status": status,
        "reviewed_cliff_lap": reviewed_cliff_lap,
        "predicted_curve": [
            {
                "tire_age": age,
                "predicted_lap_time_seconds": 90.0 + 0.01 * age,
            }
            for age in range(1, 13)
        ],
    }


class TuneTireCliffsTest(unittest.TestCase):
    def test_objective_prefers_better_classification_and_timing(self):
        strong = _objective(
            {
                "balanced_accuracy": 0.8,
                "mean_absolute_cliff_lap_error": 2.0,
                "mean_cliff_lap_error_bias": -1.0,
            }
        )
        weak = _objective(
            {
                "balanced_accuracy": 0.5,
                "mean_absolute_cliff_lap_error": 7.0,
                "mean_cliff_lap_error_bias": -5.0,
            }
        )

        self.assertLess(
            strong["objective_loss"],
            weak["objective_loss"],
        )

    def test_grids_are_bounded_and_include_production_defaults(self):
        grids = parameter_grids()

        self.assertEqual(len(grids["sustained"]), 432)
        self.assertEqual(len(grids["rolling_sustained"]), 648)
        self.assertEqual(len(grids["piecewise"]), 108)
        self.assertEqual(len(grids["hybrid"]), 324)
        self.assertIn(
            {
                "cliff_slope_threshold": DEFAULT_CONFIG[
                    "cliff_slope_threshold"
                ],
                "cliff_curvature_threshold": DEFAULT_CONFIG[
                    "cliff_curvature_threshold"
                ],
                "cliff_baseline_delta": DEFAULT_CONFIG[
                    "cliff_baseline_delta"
                ],
                "cliff_persistence_laps": DEFAULT_CONFIG[
                    "cliff_persistence_laps"
                ],
                "cliff_min_lap": DEFAULT_CONFIG["cliff_min_lap"],
            },
            grids["sustained"],
        )
        self.assertIn(
            {
                "rolling_trend_window": DEFAULT_CONFIG[
                    "rolling_trend_window"
                ],
                "cliff_slope_threshold": DEFAULT_CONFIG[
                    "cliff_slope_threshold"
                ],
                "rolling_min_slope_increase": DEFAULT_CONFIG[
                    "rolling_min_slope_increase"
                ],
                "cliff_baseline_delta": DEFAULT_CONFIG[
                    "cliff_baseline_delta"
                ],
                "rolling_min_fit_improvement_ratio": DEFAULT_CONFIG[
                    "rolling_min_fit_improvement_ratio"
                ],
                "cliff_persistence_laps": DEFAULT_CONFIG[
                    "cliff_persistence_laps"
                ],
            },
            grids["rolling_sustained"],
        )
        self.assertEqual(MINIMUM_CLIFF_RECALL, 0.5)

    def test_search_rejects_holdout_capture(self):
        capture = _capture("holdout", "confirmed_no_cliff")
        capture["split"] = "holdout"

        with self.assertRaisesRegex(ValueError, "holdout"):
            run_search(
                {
                    "artifact_type": (
                        "tire_cliff_calibration_predictions"
                    ),
                    "split": "calibration",
                    "captures": [capture],
                },
                grids={
                    "sustained": [{}],
                    "rolling_sustained": [{}],
                    "piecewise": [{}],
                    "hybrid": [{}],
                },
            )

    def test_rejects_unknown_curve_source(self):
        with self.assertRaisesRegex(ValueError, "curve_source"):
            run_search(
                {
                    "artifact_type": (
                        "tire_cliff_calibration_predictions"
                    ),
                    "split": "calibration",
                    "captures": [
                        _capture(
                            "calibration",
                            "confirmed_no_cliff",
                        )
                    ],
                },
                grids={
                    "sustained": [{}],
                    "rolling_sustained": [{}],
                    "piecewise": [{}],
                    "hybrid": [{}],
                },
                curve_source="unknown",
            )

    def test_can_limit_search_to_one_detector_method(self):
        report = run_search(
            {
                "artifact_type": (
                    "tire_cliff_calibration_predictions"
                ),
                "split": "calibration",
                "captures": [
                    _capture(
                        "calibration",
                        "confirmed_no_cliff",
                    )
                ],
            },
            grids={
                "sustained": [{}],
                "rolling_sustained": [{}],
                "piecewise": [{}],
                "hybrid": [{}],
            },
            method_names=("rolling_sustained",),
        )

        self.assertEqual(
            list(report["methods"]),
            ["rolling_sustained"],
        )


if __name__ == "__main__":
    unittest.main()
