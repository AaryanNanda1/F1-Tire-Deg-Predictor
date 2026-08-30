import unittest

from scripts.diagnose_cliff_curves import diagnose


class DiagnoseCliffCurvesTest(unittest.TestCase):
    def test_reports_post_cliff_slope_attenuation(self):
        capture = {
            "reference_id": "calibration-test",
            "split": "calibration",
            "compound": "MEDIUM",
            "manual_review_status": "confirmed_cliff",
            "reviewed_cliff_lap": 4,
            "observed_curve": [
                {
                    "tire_age": age,
                    "fuel_corrected_lap_time_seconds": value,
                }
                for age, value in enumerate(
                    [90.0, 90.1, 90.2, 90.5, 91.0, 91.6],
                    start=1,
                )
            ],
            "predicted_curve": [
                {
                    "tire_age": age,
                    "predicted_lap_time_seconds": value,
                }
                for age, value in enumerate(
                    [90.0, 90.05, 90.1, 90.15, 90.2, 90.25],
                    start=1,
                )
            ],
        }

        report = diagnose(
            {
                "artifact_type": (
                    "tire_cliff_calibration_predictions"
                ),
                "split": "calibration",
                "captures": [capture],
            }
        )

        summary = report["aggregate"]["review_status"][
            "confirmed_cliff"
        ]
        self.assertLess(
            summary["post_cliff_slope_retention_ratio"],
            0.5,
        )
        self.assertEqual(
            report["findings"][0]["code"],
            "predicted_post_cliff_slope_attenuation",
        )

    def test_rejects_holdout_capture(self):
        with self.assertRaisesRegex(ValueError, "holdout"):
            diagnose(
                {
                    "artifact_type": (
                        "tire_cliff_calibration_predictions"
                    ),
                    "split": "calibration",
                    "captures": [{"split": "holdout"}],
                }
            )


if __name__ == "__main__":
    unittest.main()
