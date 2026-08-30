import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from training_data_store import (
    ACTIVE_AERO_ROLE,
    PHYSICS_PRIOR_ROLE,
    ProcessedSessionStore,
    SessionSpec,
    refresh_sessions,
    session_specs_from_schedule,
)
from preprocessing import PREPROCESSING_VERSION


class TrainingDataStoreTest(unittest.TestCase):
    def _spec(
        self,
        *,
        year=2026,
        round_number=1,
        event_name="Australian Grand Prix",
        session_code="R",
        role=ACTIVE_AERO_ROLE,
    ):
        return SessionSpec(
            year=year,
            round_number=round_number,
            event_name=event_name,
            event_date=f"{year}-03-08",
            session_code=session_code,
            role=role,
        )

    def _frame(self, **extra_columns):
        payload = {
            "TyreLife": [1.0, 2.0],
            "LapTimeDelta": [0.1, 0.2],
            "EventDate": ["2026-03-08", "2026-03-08"],
            "EventName": ["Australian Grand Prix", "Australian Grand Prix"],
            **extra_columns,
        }
        return pd.DataFrame(payload)

    def test_saves_manifest_and_loads_union_of_session_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            first = self._spec()
            second = self._spec(
                round_number=2,
                event_name="Chinese Grand Prix",
                session_code="S",
            )

            self.assertTrue(store.save_session(first, self._frame(Driver_VER=[1, 1])))
            self.assertTrue(store.save_session(second, self._frame(Driver_NOR=[1, 1])))
            self.assertFalse(store.save_session(first, self._frame()))

            manifest = json.loads(
                (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["preprocessing_version"], PREPROCESSING_VERSION)
            self.assertIn("preprocessing_git_commit", manifest)
            self.assertEqual(len(manifest["sessions"]), 2)

            combined = store.load_role(ACTIVE_AERO_ROLE)
            self.assertEqual(len(combined), 4)
            self.assertIn("Driver_VER", combined.columns)
            self.assertIn("Driver_NOR", combined.columns)
            self.assertEqual(int(combined["Driver_VER"].isna().sum()), 2)
            self.assertEqual(int(combined["Driver_NOR"].isna().sum()), 2)
            self.assertEqual(
                sorted(combined["SampleWeight"].unique().tolist()),
                [0.75, 1.0],
            )
            self.assertEqual(
                set(combined["SessionCode"].tolist()),
                {"R", "S"},
            )
            self.assertEqual(
                set(combined["TrainingRole"].tolist()),
                {ACTIVE_AERO_ROLE},
            )
            self.assertEqual(set(combined["Season"].tolist()), {2026})
            store.validate(verify_hashes=True)

    def test_builds_only_completed_relevant_sessions_from_schedule(self):
        schedule = pd.DataFrame(
            [
                {
                    "RoundNumber": 1,
                    "EventName": "Australian Grand Prix",
                    "EventDate": pd.Timestamp("2026-03-08"),
                    "Session1": "Practice 1",
                    "Session2": "Practice 2",
                    "Session5": "Race",
                },
                {
                    "RoundNumber": 2,
                    "EventName": "Chinese Grand Prix",
                    "EventDate": pd.Timestamp("2026-03-15"),
                    "Session3": "Sprint",
                    "Session5": "Race",
                },
                {
                    "RoundNumber": 3,
                    "EventName": "Future Grand Prix",
                    "EventDate": pd.Timestamp("2026-08-01"),
                    "Session5": "Race",
                },
            ]
        )

        specs = session_specs_from_schedule(
            schedule,
            year=2026,
            as_of=date(2026, 7, 29),
            role=ACTIVE_AERO_ROLE,
        )

        self.assertEqual(
            [spec.key for spec in specs],
            [
                "2026:Australian Grand Prix:R",
                "2026:Australian Grand Prix:FP2",
                "2026:Chinese Grand Prix:R",
                "2026:Chinese Grand Prix:S",
            ],
        )

    def test_refresh_skips_stored_sessions_and_reports_mandatory_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            existing = self._spec()
            missing_race = self._spec(
                round_number=2,
                event_name="Chinese Grand Prix",
            )
            store.save_session(existing, self._frame())

            loader = Mock(side_effect=RuntimeError("upstream unavailable"))
            preprocessor = Mock()
            result = refresh_sessions(
                store,
                [existing, missing_race],
                loader=loader,
                preprocessor=preprocessor,
            )

            self.assertEqual(result["skipped"], [existing.key])
            self.assertEqual(
                [failure["session_key"] for failure in result["mandatory_failures"]],
                [missing_race.key],
            )
            loader.assert_called_once_with(
                missing_race.year,
                missing_race.event_name,
                missing_race.session_code,
            )
            preprocessor.assert_not_called()

    def test_roles_remain_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            active = self._spec()
            prior = self._spec(
                year=2025,
                event_name="British Grand Prix",
                role=PHYSICS_PRIOR_ROLE,
            )
            store.save_session(active, self._frame())
            store.save_session(prior, self._frame())

            self.assertEqual(store.session_keys(ACTIVE_AERO_ROLE), [active.key])
            self.assertEqual(store.session_keys(PHYSICS_PRIOR_ROLE), [prior.key])

    def test_reprocess_existing_replaces_immutable_session_row_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            spec = self._spec()
            store.save_session(spec, self._frame(OldFeature=[1, 1]))

            loader = Mock(return_value=object())
            result = refresh_sessions(
                store,
                [spec],
                loader=loader,
                preprocessor=Mock(return_value=self._frame(NewFeature=[2, 2])),
                reprocess_existing=True,
            )

            self.assertEqual(result["reprocessed"], [spec.key])
            self.assertEqual(result["added"], [])
            loaded = store.load_role(ACTIVE_AERO_ROLE)
            self.assertIn("NewFeature", loaded.columns)
            self.assertNotIn("OldFeature", loaded.columns)

    def test_persists_preprocessing_filter_audit_in_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            frame = self._frame()
            frame.attrs["filter_audit"] = {
                "pit_in_out_removed": 2,
                "robust_outlier_removed": 3,
                "final_rows": 2,
            }
            store.save_session(self._spec(), frame)
            manifest = json.loads(
                (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["sessions"][self._spec().key]["filter_audit"][
                    "robust_outlier_removed"
                ],
                3,
            )

    def test_legacy_manifest_is_explicitly_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            store.save_session(self._spec(), self._frame())
            manifest_path = Path(temporary) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 1
            manifest.pop("preprocessing_version", None)
            manifest.pop("preprocessing_git_commit", None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            legacy = ProcessedSessionStore(temporary)
            self.assertTrue(legacy.requires_rebuild)
            with self.assertRaisesRegex(Exception, "run scripts/rebuild_processed_training_data.py"):
                legacy.load_role(ACTIVE_AERO_ROLE)


if __name__ == "__main__":
    unittest.main()
