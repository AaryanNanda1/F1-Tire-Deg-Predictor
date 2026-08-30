import tempfile
import unittest

import pandas as pd

from scripts.update_ground_effect_training_data import (
    _coverage,
    _import_prior_sessions,
    _select_missing_specs,
)
from training_data_store import (
    GROUND_EFFECT_ROLE,
    PHYSICS_PRIOR_ROLE,
    ProcessedSessionStore,
    SessionSpec,
)


class GroundEffectTrainingDataUpdateTest(unittest.TestCase):
    def _spec(
        self,
        year,
        round_number,
        event_name,
        session_code="R",
        role=GROUND_EFFECT_ROLE,
    ):
        return SessionSpec(
            year=year,
            round_number=round_number,
            event_name=event_name,
            event_date=f"{year}-03-01",
            session_code=session_code,
            role=role,
        )

    def _frame(self, year):
        return pd.DataFrame(
            {
                "TyreLife": [1.0, 2.0],
                "LapTimeDelta": [0.1, 0.2],
                "EventDate": [f"{year}-03-01"] * 2,
            }
        )

    def test_selects_whole_events_and_honors_coverage_only_mode(self):
        expected = [
            self._spec(2022, 1, "Bahrain Grand Prix"),
            self._spec(2022, 1, "Bahrain Grand Prix", "FP2"),
            self._spec(2022, 2, "Saudi Arabian Grand Prix"),
            self._spec(2023, 1, "Bahrain Grand Prix"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)

            selected = _select_missing_specs(
                expected,
                store,
                fetch_years=[2022, 2023],
                max_new_events=1,
            )
            status_only = _select_missing_specs(
                expected,
                store,
                fetch_years=[2022, 2023],
                max_new_events=0,
            )

        self.assertEqual(
            [spec.key for spec in selected],
            [
                "2022:Bahrain Grand Prix:R",
                "2022:Bahrain Grand Prix:FP2",
            ],
        )
        self.assertEqual(status_only, [])

    def test_imports_prior_sessions_as_ground_effect_rows_idempotently(self):
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as target_temp:
            source = ProcessedSessionStore(source_temp)
            source.save_session(
                self._spec(
                    2024,
                    1,
                    "Bahrain Grand Prix",
                    role=PHYSICS_PRIOR_ROLE,
                ),
                self._frame(2024),
            )
            target = ProcessedSessionStore(target_temp)

            first = _import_prior_sessions(target, source_temp)
            second = _import_prior_sessions(target, source_temp)

            self.assertEqual(first, ["2024:Bahrain Grand Prix:R"])
            self.assertEqual(second, [])
            self.assertEqual(
                target.session_keys(GROUND_EFFECT_ROLE),
                ["2024:Bahrain Grand Prix:R"],
            )

    def test_coverage_requires_every_expected_session(self):
        expected = [
            self._spec(2022, 1, "Bahrain Grand Prix"),
            self._spec(2022, 1, "Bahrain Grand Prix", "FP2"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = ProcessedSessionStore(temporary)
            store.save_session(expected[0], self._frame(2022))

            coverage = _coverage(expected, store)

        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["expected_race_count"], 1)
        self.assertEqual(coverage["stored_race_count"], 1)
        self.assertEqual(coverage["missing_session_count"], 1)
        self.assertEqual(
            coverage["missing_session_keys"],
            ["2022:Bahrain Grand Prix:FP2"],
        )


if __name__ == "__main__":
    unittest.main()
