import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.capture_cliff_predictions import (
    ERA_NAME,
    _capture_label,
    _model_provenance,
    _training_overlap,
    _weather,
)


class _CaptureSimulator:
    def _simulate_compound(self, *args, **kwargs):
        if not kwargs.get("return_raw"):
            raise AssertionError("capture must request the raw curve")
        return [0.0, 0.1], [90.0, 90.1], [0.0, -0.1]

    def _analyze_curve(self, *args, **kwargs):
        return {
            "performance_cliff_lap": None,
            "cliff_confidence": None,
            "cliff_method": "sustained",
            "cliff_reason": "fixture",
            "strategy_useful_life_lap": 1,
        }


class CaptureCliffPredictionsTest(unittest.TestCase):
    def test_weather_uses_stint_mean_and_records_wind_fallback(self):
        label = {
            "mean_air_temp_c": 24.5,
            "mean_track_temp_c": 36.0,
        }
        clean_laps = [
            {"humidity_percent": 40.0},
            {"humidity_percent": 50.0},
            {"humidity_percent": None},
        ]

        weather = _weather(label, clean_laps)

        self.assertEqual(weather["air_temp"], 24.5)
        self.assertEqual(weather["track_temp"], 36.0)
        self.assertEqual(weather["humidity"], 45.0)
        self.assertEqual(weather["fallback_fields"], ["wind_speed"])
        self.assertFalse(weather["rainfall"])

    def test_training_overlap_matches_only_same_event(self):
        label = {
            "season": 2023,
            "event_name": "British Grand Prix",
        }
        loaded_events = [
            "2023:British Grand Prix:R",
            "2023:British Grand Prix:FP2",
            "2023:Hungarian Grand Prix:R",
        ]

        self.assertEqual(
            _training_overlap(label, loaded_events),
            [
                "2023:British Grand Prix:R",
                "2023:British Grand Prix:FP2",
            ],
        )

    def test_capture_preserves_raw_and_monotonic_dropoff(self):
        label = {
            "reference_id": "calibration-2023-bahrain:DRV:stint-1:MEDIUM",
            "race_id": "calibration-2023-bahrain",
            "season": 2023,
            "event_name": "Bahrain Grand Prix",
            "driver": "DRV",
            "team": "Test Team",
            "stint": 1,
            "compound": "MEDIUM",
            "starting_tyre_age": 1,
            "ending_tyre_age": 2,
            "manual_review_status": "confirmed_no_cliff",
            "reviewed_cliff_lap": None,
            "mean_air_temp_c": 25.0,
            "mean_track_temp_c": 35.0,
        }
        clean_laps = [
            {
                "tire_age": age,
                "fuel_corrected_lap_time_seconds": 90.0 + age / 10,
                "humidity_percent": 50.0,
            }
            for age in (1, 2)
        ]

        capture = _capture_label(
            label,
            clean_laps,
            _CaptureSimulator(),
            [],
        )

        second_point = capture["full_predicted_curve"][1]
        self.assertEqual(second_point["predicted_drop_off_seconds"], 0.1)
        self.assertEqual(
            second_point["raw_predicted_drop_off_seconds"],
            -0.1,
        )

    def test_model_provenance_hashes_isolated_artifacts(self):
        with TemporaryDirectory() as directory:
            models_dir = Path(directory)
            (models_dir / f"{ERA_NAME}_model.joblib").write_bytes(b"model")
            (models_dir / f"{ERA_NAME}_features.joblib").write_bytes(
                b"features"
            )
            (
                models_dir / f"{ERA_NAME}_degradation_model.joblib"
            ).write_bytes(b"degradation-model")
            (
                models_dir / f"{ERA_NAME}_degradation_features.joblib"
            ).write_bytes(b"degradation-features")
            metadata = {
                ERA_NAME: {
                    "as_of": "2025-12-31",
                    "trained_at": "2026-07-30T00:00:00+00:00",
                    "loaded_events": ["2023:British Grand Prix:R"],
                    "degradation_target": "within-stint change",
                }
            }
            (models_dir / "era_training_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            provenance = _model_provenance(models_dir)

        self.assertEqual(provenance["era"], ERA_NAME)
        self.assertEqual(provenance["as_of"], "2025-12-31")
        self.assertEqual(
            provenance["loaded_events"],
            ["2023:British Grand Prix:R"],
        )
        self.assertEqual(len(provenance["model_sha256"]), 64)
        self.assertEqual(len(provenance["features_sha256"]), 64)
        self.assertEqual(
            provenance["degradation_target"],
            "within-stint change",
        )
        self.assertEqual(
            len(provenance["degradation_model_sha256"]),
            64,
        )


if __name__ == "__main__":
    unittest.main()
