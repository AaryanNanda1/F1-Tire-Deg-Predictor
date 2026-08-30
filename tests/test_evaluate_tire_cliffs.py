import unittest

from scripts.evaluate_tire_cliffs import (
    _captured_curve,
    _observed_curve,
    _raw_captured_curve,
    evaluate_calibration_predictions,
)


def _capture(
    reference_id,
    baseline,
    status,
    reviewed_cliff_lap=None,
    *,
    first_age=1,
):
    return {
        "reference_id": reference_id,
        "race_id": "calibration-2023-test",
        "split": "calibration",
        "season": 2023,
        "event_name": "Test Grand Prix",
        "track_name": "Test Circuit",
        "track_type": "balanced",
        "driver": "DRV",
        "team": "Test Team",
        "compound": "MEDIUM",
        "starting_tire_age": first_age,
        "ending_tire_age": 10,
        "manual_review_status": status,
        "reviewed_cliff_lap": reviewed_cliff_lap,
        "weather_context": {"track_temp": 35.0},
        "training_overlap": ["2023:Test Grand Prix:R"],
        "predicted_curve": [
            {
                "tire_age": age,
                "predicted_lap_time_seconds": baseline,
            }
            for age in range(first_age, 11)
        ],
    }


class EvaluateTireCliffsTest(unittest.TestCase):
    def test_scores_classification_and_matched_cliff_timing(self):
        captures = [
            _capture("positive-hit", 90, "confirmed_cliff", 6),
            _capture("positive-miss", 91, "confirmed_cliff", 8),
            _capture("negative-false", 92, "confirmed_no_cliff"),
            _capture("negative-true", 93, "confirmed_no_cliff"),
        ]

        def detector(curve):
            outcomes = {
                90: (5, "high"),
                91: (None, None),
                92: (4, "medium"),
                93: (None, None),
            }
            cliff_lap, confidence = outcomes[round(float(curve[0]))]
            return {
                "performance_cliff_lap": cliff_lap,
                "cliff_confidence": confidence,
                "cliff_reason": "fixture",
            }

        report = evaluate_calibration_predictions(
            {
                "artifact_type": "tire_cliff_calibration_predictions",
                "split": "calibration",
                "captures": captures,
            },
            detectors=(("fixture", detector),),
        )

        metrics = report["detectors"]["fixture"]["overall"]
        self.assertEqual(metrics["true_positive_count"], 1)
        self.assertEqual(metrics["false_negative_count"], 1)
        self.assertEqual(metrics["false_positive_count"], 1)
        self.assertEqual(metrics["true_negative_count"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["false_cliff_rate"], 0.5)
        self.assertEqual(metrics["missed_cliff_rate"], 0.5)
        self.assertEqual(metrics["mean_absolute_cliff_lap_error"], 1.0)
        self.assertEqual(metrics["mean_cliff_lap_error_bias"], -1.0)
        self.assertEqual(metrics["early_detection_count"], 1)

    def test_applies_tire_age_offset_for_interval_only_capture(self):
        capture = _capture(
            "offset",
            94,
            "confirmed_cliff",
            4,
            first_age=3,
        )

        def detector(_curve):
            return {
                "performance_cliff_lap": 2,
                "cliff_confidence": "high",
                "cliff_reason": "fixture",
            }

        report = evaluate_calibration_predictions(
            {
                "artifact_type": "tire_cliff_calibration_predictions",
                "split": "calibration",
                "captures": [capture],
            },
            detectors=(("fixture", detector),),
        )

        row = report["detectors"]["fixture"]["stints"][0]
        self.assertEqual(row["detected_cliff_lap"], 4)
        self.assertEqual(row["classification"], "true_positive")
        self.assertEqual(row["cliff_lap_error"], 0)

    def test_rejects_non_calibration_capture(self):
        capture = _capture("holdout", 90, "confirmed_no_cliff")
        capture["split"] = "holdout"

        with self.assertRaisesRegex(ValueError, "non-calibration"):
            evaluate_calibration_predictions(
                {
                    "artifact_type": (
                        "tire_cliff_calibration_predictions"
                    ),
                    "split": "calibration",
                    "captures": [capture],
                }
            )

    def test_rejects_non_continuous_curve(self):
        capture = _capture("gap", 90, "confirmed_no_cliff")
        capture["predicted_curve"].pop(3)

        with self.assertRaisesRegex(ValueError, "continuous"):
            _captured_curve(capture)

    def test_observed_curve_interpolates_removed_tire_ages(self):
        curve, offset = _observed_curve(
            {
                "reference_id": "observed-gap",
                "observed_curve": [
                    {
                        "tire_age": 2,
                        "fuel_corrected_lap_time_seconds": 90.0,
                    },
                    {
                        "tire_age": 4,
                        "fuel_corrected_lap_time_seconds": 92.0,
                    },
                ],
            }
        )

        self.assertEqual(offset, 1)
        self.assertEqual(curve.tolist(), [90.0, 91.0, 92.0])

    def test_raw_curve_loader_preserves_pre_monotonic_signal(self):
        capture = _capture(
            "raw",
            90,
            "confirmed_no_cliff",
        )
        for index, point in enumerate(capture["predicted_curve"]):
            point["raw_predicted_drop_off_seconds"] = [
                0.0,
                -0.2,
                0.1,
                0.05,
                0.2,
                0.15,
                0.4,
                0.35,
                0.5,
                0.45,
            ][index]

        curve, offset = _raw_captured_curve(capture)

        self.assertEqual(offset, 0)
        self.assertEqual(curve[:4].tolist(), [0.0, -0.2, 0.1, 0.05])


if __name__ == "__main__":
    unittest.main()
