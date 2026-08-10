import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from train_era_models import (
    build_active_aero_training_data,
    load_persistent_active_aero_data,
    load_persistent_ground_effect_data,
    perform_walk_forward_validation,
    train_and_save,
)
from training_data_store import (
    ACTIVE_AERO_ROLE,
    GROUND_EFFECT_ROLE,
    PHYSICS_PRIOR_ROLE,
    ProcessedSessionStore,
    SessionSpec,
    TrainingDataStoreError,
)


class PersistentSingleModelTrainingTest(unittest.TestCase):
    def _spec(self, year, event, role, session_code="R"):
        return SessionSpec(
            year=year,
            round_number=1,
            event_name=event,
            event_date=f"{year}-03-01",
            session_code=session_code,
            role=role,
        )

    def _frame(self, year, event, rows, *, soft=False):
        return pd.DataFrame(
            {
                "TyreLife": list(range(1, rows + 1)),
                "LapTimeDelta": [value / 100 for value in range(rows)],
                "EventDate": [f"{year}-03-01"] * rows,
                f"EventName_{event}": [1] * rows,
                "Compound_SOFT": [int(soft)] * rows,
                "IsWet": [0] * rows,
            }
        )

    def test_loads_active_and_prior_roles_from_one_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            store.save_session(
                self._spec(2026, "Australian Grand Prix", ACTIVE_AERO_ROLE),
                self._frame(2026, "Australian Grand Prix", 4),
            )
            store.save_session(
                self._spec(2025, "British Grand Prix", PHYSICS_PRIOR_ROLE),
                self._frame(2025, "British Grand Prix", 6),
            )

            active, prior, details = load_persistent_active_aero_data(store.root)

            self.assertEqual(len(active), 4)
            self.assertEqual(len(prior), 6)
            self.assertEqual(details["active_session_count"], 1)
            self.assertEqual(details["prior_session_count"], 1)
            self.assertEqual(
                details["training_data_source"],
                "persistent_processed_sessions",
            )

    def test_preserves_original_row_prior_sample_and_soft_weighting(self):
        active = self._frame(2026, "Australian Grand Prix", 4, soft=True)
        active["SampleWeight"] = 1.0
        prior = self._frame(2025, "British Grand Prix", 10)
        prior["SampleWeight"] = 0.5

        combined, sampled_prior_rows = build_active_aero_training_data(active, prior)

        self.assertEqual(sampled_prior_rows, 5)
        self.assertEqual(len(combined), 9)
        self.assertEqual(combined.iloc[0]["SampleWeight"], 1.1)
        self.assertEqual(combined.iloc[3]["SampleWeight"], 1.4)
        self.assertEqual(
            set(combined.iloc[4:]["SampleWeight"].unique().tolist()),
            {1.0},
        )

    def test_ground_store_must_declare_complete_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            store.save_session(
                self._spec(2022, "Bahrain Grand Prix", GROUND_EFFECT_ROLE),
                self._frame(2022, "Bahrain Grand Prix", 5, soft=True),
            )
            store.set_metadata(
                "coverage",
                {
                    "complete": True,
                    "expected_race_count": 1,
                    "missing_race_count": 0,
                    "missing_session_count": 0,
                },
            )

            data, details = load_persistent_ground_effect_data(store.root)

            self.assertEqual(len(data), 5)
            self.assertEqual(details["race_count"], 1)
            self.assertEqual(data.iloc[0]["SampleWeight"], 1.1)

            store.set_metadata(
                "coverage",
                {
                    "complete": False,
                    "expected_race_count": 1,
                    "missing_race_count": 1,
                    "missing_session_count": 2,
                },
            )
            with self.assertRaises(TrainingDataStoreError):
                load_persistent_ground_effect_data(store.root)

    def test_provenance_columns_are_not_model_features(self):
        frame = self._frame(2026, "Australian Grand Prix", 40)
        frame["SampleWeight"] = 1.0
        frame["EventName"] = "Australian Grand Prix"
        frame["SessionKey"] = "2026:Australian Grand Prix:R"
        frame["SessionCode"] = "R"
        frame["TrainingRole"] = ACTIVE_AERO_ROLE
        frame["Season"] = 2026

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            train_and_save(
                frame,
                output / "model.joblib",
                output / "features.joblib",
            )
            features = joblib.load(output / "features.joblib")

        for column in (
            "EventDate",
            "EventName",
            "SessionKey",
            "SessionCode",
            "TrainingRole",
            "Season",
        ):
            self.assertNotIn(column, features)

    @patch("train_era_models.HistGradientBoostingRegressor")
    def test_active_validation_scores_only_current_era_events(self, regressor):
        model = regressor.return_value
        model.predict.side_effect = lambda frame: [0.0] * len(frame)
        frame = pd.DataFrame(
            {
                "LapTimeDelta": [0.0, 0.0, 0.0, 0.0],
                "SampleWeight": [1.0, 1.0, 1.0, 1.0],
                "EventDate": [
                    "2024-03-01",
                    "2025-03-01",
                    "2026-03-01",
                    "2026-04-01",
                ],
                "EventName": [
                    "Bahrain Grand Prix",
                    "Bahrain Grand Prix",
                    "Bahrain Grand Prix",
                    "Japanese Grand Prix",
                ],
                "TyreLife": [1, 1, 1, 1],
            }
        )

        mae = perform_walk_forward_validation(frame, test_start_year=2026)

        self.assertEqual(mae, 0.0)
        self.assertEqual(model.fit.call_count, 2)


if __name__ == "__main__":
    unittest.main()
