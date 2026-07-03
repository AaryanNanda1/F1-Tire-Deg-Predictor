import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from scripts import check_active_aero_retrain_needed as preflight


class ActiveAeroPreflightTest(unittest.TestCase):
    def _schedule(self):
        return pd.DataFrame(
            [
                {
                    "RoundNumber": 1,
                    "EventName": "Australian Grand Prix",
                    "EventDate": pd.Timestamp("2026-03-08"),
                },
                {
                    "RoundNumber": 2,
                    "EventName": "Chinese Grand Prix",
                    "EventDate": pd.Timestamp("2026-03-15"),
                },
                {
                    "RoundNumber": 3,
                    "EventName": "Future Grand Prix",
                    "EventDate": pd.Timestamp("2026-08-01"),
                },
            ]
        )

    def test_lists_completed_active_aero_races(self):
        with patch.object(preflight.fastf1, "get_event_schedule", return_value=self._schedule()):
            races = preflight.list_completed_races(date(2026, 7, 3))

        self.assertEqual(
            [race.race_key for race in races],
            [
                "2026:Australian Grand Prix:R",
                "2026:Chinese Grand Prix:R",
            ],
        )

    def test_empty_active_aero_schedule_is_a_preflight_error(self):
        with patch.object(
            preflight.fastf1,
            "get_event_schedule",
            return_value=pd.DataFrame(),
        ):
            with self.assertRaises(preflight.PreflightError):
                preflight.list_completed_races(date(2026, 7, 3))

    def test_retrains_when_active_aero_metadata_is_missing(self):
        completed = [
            preflight.CompletedRace(
                year=2026,
                round_number=1,
                event_name="Australian Grand Prix",
                event_date="2026-03-08",
                race_key="2026:Australian Grand Prix:R",
            )
        ]

        decision = preflight.decide_retrain_needed({}, completed, date(2026, 7, 3))

        self.assertTrue(decision.should_retrain)
        self.assertEqual(decision.reason, "active_aero_metadata_missing")

    def test_skips_when_metadata_has_all_completed_races(self):
        metadata = {
            preflight.ACTIVE_AERO_KEY: {
                "status": "trained_hybrid",
                "loaded_events": [
                    "2026:Australian Grand Prix:R",
                    "2026:Australian Grand Prix:FP2",
                    "2026:Chinese Grand Prix:R",
                ],
            }
        }
        completed = [
            preflight.CompletedRace(
                year=2026,
                round_number=1,
                event_name="Australian Grand Prix",
                event_date="2026-03-08",
                race_key="2026:Australian Grand Prix:R",
            ),
            preflight.CompletedRace(
                year=2026,
                round_number=2,
                event_name="Chinese Grand Prix",
                event_date="2026-03-15",
                race_key="2026:Chinese Grand Prix:R",
            ),
        ]

        decision = preflight.decide_retrain_needed(metadata, completed, date(2026, 7, 3))

        self.assertFalse(decision.should_retrain)
        self.assertEqual(decision.reason, "no_new_completed_active_aero_races")

    def test_retrains_when_completed_race_is_missing_from_metadata(self):
        metadata = {
            preflight.ACTIVE_AERO_KEY: {
                "status": "trained_hybrid",
                "loaded_events": ["2026:Australian Grand Prix:R"],
            }
        }
        completed = [
            preflight.CompletedRace(
                year=2026,
                round_number=1,
                event_name="Australian Grand Prix",
                event_date="2026-03-08",
                race_key="2026:Australian Grand Prix:R",
            ),
            preflight.CompletedRace(
                year=2026,
                round_number=2,
                event_name="Chinese Grand Prix",
                event_date="2026-03-15",
                race_key="2026:Chinese Grand Prix:R",
            ),
        ]

        decision = preflight.decide_retrain_needed(metadata, completed, date(2026, 7, 3))

        self.assertTrue(decision.should_retrain)
        self.assertEqual(
            decision.reason,
            "completed_race_data_missing_from_model_metadata",
        )
        self.assertEqual(decision.missing_race_keys, ["2026:Chinese Grand Prix:R"])


if __name__ == "__main__":
    unittest.main()
