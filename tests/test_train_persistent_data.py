import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from train_era_models import (
    _eligible_walk_forward_test_events,
    _validation_event_keys,
    build_active_aero_training_data,
    load_persistent_active_aero_data,
    load_persistent_ground_effect_data,
    train_model_pair,
)
from training_data_store import (
    ACTIVE_AERO_ROLE,
    GROUND_EFFECT_ROLE,
    PHYSICS_PRIOR_ROLE,
    ProcessedSessionStore,
    SessionSpec,
    TrainingDataStoreError,
)


class PersistentTrainingDataTest(unittest.TestCase):
    def _spec(self, year, event, role):
        return SessionSpec(
            year=year,
            round_number=1,
            event_name=event,
            event_date=f"{year}-03-01",
            session_code="R",
            role=role,
        )

    def _frame(self, year, event, rows):
        return pd.DataFrame(
            {
                "TyreLife": list(range(1, rows + 1)),
                "LapTimeDelta": [0.1] * rows,
                "EventDate": [f"{year}-03-01"] * rows,
                f"EventName_{event}": [1] * rows,
                "IsWet": [0] * rows,
            }
        )

    def test_loads_active_and_prior_sessions_with_provenance(self):
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

            active, prior, details = load_persistent_active_aero_data(
                store.root
            )

            self.assertEqual(len(active), 4)
            self.assertEqual(len(prior), 6)
            self.assertEqual(details["active_session_count"], 1)
            self.assertEqual(details["prior_session_count"], 1)
            self.assertEqual(
                details["training_data_source"],
                "persistent_processed_sessions",
            )

    def test_blends_half_of_prior_deterministically(self):
        active = self._frame(2026, "Australian Grand Prix", 4)
        active["SampleWeight"] = 1.0
        prior = self._frame(2025, "British Grand Prix", 10)
        prior["SampleWeight"] = 0.5

        combined, sampled_prior_rows = build_active_aero_training_data(
            active,
            prior,
        )

        self.assertEqual(sampled_prior_rows, 5)
        self.assertEqual(len(combined), 9)
        self.assertEqual(
            set(combined.iloc[4:]["SampleWeight"].unique().tolist()),
            {0.2},
        )

    def test_blends_complete_prior_sessions_when_provenance_exists(self):
        active = self._frame(2026, "Australian Grand Prix", 4)
        active["SampleWeight"] = 1.0
        active["SessionKey"] = "2026:Australian Grand Prix:R"
        prior_frames = []
        for index in range(4):
            session = self._frame(
                2025,
                f"Prior Grand Prix {index}",
                5,
            )
            session["SampleWeight"] = 0.5
            session["SessionKey"] = f"2025:Prior Grand Prix {index}:R"
            prior_frames.append(session)
        prior = pd.concat(prior_frames, ignore_index=True)

        combined, sampled_prior_rows = build_active_aero_training_data(
            active,
            prior,
        )

        selected_prior = combined[
            combined["SessionKey"].astype(str).str.startswith("2025:")
        ]
        self.assertEqual(sampled_prior_rows, 10)
        self.assertEqual(selected_prior["SessionKey"].nunique(), 2)
        self.assertEqual(
            set(selected_prior.groupby("SessionKey").size().tolist()),
            {5},
        )

    def test_does_not_train_from_prior_without_active_data(self):
        prior = self._frame(2025, "British Grand Prix", 10)
        combined, sampled_prior_rows = build_active_aero_training_data(
            pd.DataFrame(),
            prior,
        )

        self.assertTrue(combined.empty)
        self.assertEqual(sampled_prior_rows, 0)

    def test_loads_complete_local_ground_effect_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            store.save_session(
                self._spec(2022, "Bahrain Grand Prix", GROUND_EFFECT_ROLE),
                self._frame(2022, "Bahrain Grand Prix", 5),
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
            self.assertEqual(details["training_data_source"], "local_processed_sessions")
            self.assertEqual(details["coverage_start_year"], 2022)
            self.assertEqual(details["coverage_end_year"], 2025)
            self.assertEqual(details["race_count_by_year"]["2022"], 1)
            self.assertEqual(
                data["EventName"].unique().tolist(),
                ["Bahrain Grand Prix"],
            )

    def test_rejects_incomplete_local_ground_effect_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            store.save_session(
                self._spec(2022, "Bahrain Grand Prix", GROUND_EFFECT_ROLE),
                self._frame(2022, "Bahrain Grand Prix", 5),
            )
            store.set_metadata(
                "coverage",
                {
                    "complete": False,
                    "missing_race_count": 1,
                    "missing_session_count": 2,
                },
            )

            with self.assertRaisesRegex(
                TrainingDataStoreError,
                "1 races and 2 sessions remain",
            ):
                load_persistent_ground_effect_data(store.root)

    def test_validation_event_keys_distinguish_repeated_annual_events(self):
        rows = pd.DataFrame(
            {
                "EventDate": ["2022-03-20", "2023-03-05"],
                "EventName": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            }
        )

        keys = _validation_event_keys(rows)

        self.assertEqual(keys.nunique(), 2)
        self.assertEqual(
            keys.tolist(),
            [
                "2022-03-20::Bahrain Grand Prix",
                "2023-03-05::Bahrain Grand Prix",
            ],
        )

    def test_active_aero_validation_scores_only_current_era_tests(self):
        events = pd.DataFrame(
            {
                "EventDate": [
                    "2024-03-02",
                    "2025-03-16",
                    "2026-03-08",
                    "2026-03-22",
                ],
                "EventName": [
                    "Bahrain Grand Prix",
                    "Australian Grand Prix",
                    "Bahrain Grand Prix",
                    "Chinese Grand Prix",
                ],
            }
        )

        eligible = _eligible_walk_forward_test_events(
            events,
            test_start_year=2026,
        )

        self.assertEqual(
            eligible["EventDate"].tolist(),
            ["2026-03-08", "2026-03-22"],
        )

    @patch("train_era_models.train_degradation_model")
    @patch("train_era_models.train_and_save")
    def test_model_pair_reuses_one_processed_frame(
        self,
        train_pace,
        train_degradation,
    ):
        frame = pd.DataFrame({"LapTimeDelta": [0.1]})
        train_pace.return_value = {
            "mae": 0.8,
            "mae_validation_scope": (
                "walk_forward_2026_plus_test_events"
            ),
        }
        train_degradation.return_value = {
            "degradation_mae": 0.4,
            "degradation_mae_validation_scope": (
                "walk_forward_2026_plus_test_events"
            ),
        }

        result = train_model_pair(
            frame,
            "active_aero_2026_2030",
            Path("models"),
            validation_test_start_year=2026,
        )

        self.assertIs(train_pace.call_args.args[0], frame)
        self.assertIs(train_degradation.call_args.args[0], frame)
        self.assertEqual(
            result["training_architecture"],
            "additive_pace_plus_degradation",
        )
        self.assertEqual(
            result["degradation_model_path"],
            "models/active_aero_2026_2030_degradation_model.joblib",
        )
        self.assertTrue(result["training_run_id"])


if __name__ == "__main__":
    unittest.main()
