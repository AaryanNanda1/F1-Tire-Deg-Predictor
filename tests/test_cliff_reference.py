import copy
import unittest

import numpy as np
import pandas as pd

from cliff_reference import (
    CliffReferenceValidationError,
    construct_observed_cliff_reference,
    extract_session_cliff_references,
    prepare_fastf1_laps,
    validate_cliff_manifest,
)


def _race():
    return {
        "id": "calibration-test-race",
        "split": "calibration",
        "season": 2024,
        "event_name": "Test Grand Prix",
        "session": "R",
        "tags": ["synthetic"],
    }


def _manifest():
    return {
        "schema_version": 1,
        "races": [_race()],
    }


def _synthetic_session_laps(count=20, with_cliff=True):
    ages = np.arange(1, count + 1, dtype=float)
    if with_cliff:
        degradation = np.where(
            ages <= 10,
            0.02 * (ages - 1),
            0.18 + 0.20 * (ages - 10),
        )
    else:
        degradation = 0.03 * (ages - 1)
    raw_lap_seconds = 90.0 + degradation - 0.07 * (ages - 1)
    lap_times = pd.to_timedelta(raw_lap_seconds, unit="s")
    times = pd.to_timedelta(np.arange(1, count + 1) * 95, unit="s")

    return pd.DataFrame(
        {
            "Driver": ["TST"] * count,
            "Team": ["Test Team"] * count,
            "Stint": [1] * count,
            "Compound": ["MEDIUM"] * count,
            "TyreLife": ages,
            "LapNumber": ages,
            "LapTime": lap_times,
            "Time": times,
            "TrackStatus": ["1"] * count,
            "PitInTime": pd.Series(
                [pd.NaT] * count,
                dtype="timedelta64[ns]",
            ),
            "PitOutTime": pd.Series(
                [pd.NaT] * count,
                dtype="timedelta64[ns]",
            ),
            "IsAccurate": [True] * count,
            "Deleted": [False] * count,
        }
    )


def _weather(count=20):
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(
                np.arange(0, count + 2) * 95,
                unit="s",
            ),
            "Rainfall": [False] * (count + 2),
            "AirTemp": [25.0] * (count + 2),
            "TrackTemp": [35.0] * (count + 2),
            "Humidity": [50.0] * (count + 2),
        }
    )


class CliffReferenceTest(unittest.TestCase):
    def test_prepares_laps_with_fuel_correction_and_rejection_reasons(self):
        laps = _synthetic_session_laps()
        laps.loc[2, "TrackStatus"] = "4"
        laps.loc[19, "PitInTime"] = pd.Timedelta(seconds=1)

        prepared = prepare_fastf1_laps(laps, _weather())

        self.assertIn(
            "non_green_track_status",
            prepared.loc[prepared["SourceRow"] == 2, "RejectionReasons"].iloc[0],
        )
        self.assertIn(
            "pit_in_lap",
            prepared.loc[prepared["SourceRow"] == 19, "RejectionReasons"].iloc[0],
        )
        first = prepared[prepared["TyreLifeNumeric"] == 1].iloc[0]
        tenth = prepared[prepared["TyreLifeNumeric"] == 10].iloc[0]
        self.assertAlmostEqual(
            tenth["FuelCorrectedLapTimeSeconds"]
            - first["FuelCorrectedLapTimeSeconds"],
            0.18,
            places=3,
        )

    def test_constructs_review_candidate_near_known_breakpoint(self):
        prepared = prepare_fastf1_laps(
            _synthetic_session_laps(),
            _weather(),
        )
        clean = prepared[prepared["AcceptedLap"]]

        reference = construct_observed_cliff_reference(clean)

        self.assertEqual(
            reference["reference_status"],
            "candidate_for_manual_review",
        )
        self.assertIn(reference["observed_cliff_lap"], range(10, 13))
        self.assertGreater(
            reference["post_cliff_slope_sec_per_lap"],
            reference["pre_cliff_slope_sec_per_lap"],
        )

    def test_linear_stint_remains_no_cliff_candidate(self):
        prepared = prepare_fastf1_laps(
            _synthetic_session_laps(with_cliff=False),
            _weather(),
        )

        reference = construct_observed_cliff_reference(
            prepared[prepared["AcceptedLap"]]
        )

        self.assertEqual(reference["reference_status"], "no_cliff_candidate")
        self.assertIsNone(reference["observed_cliff_lap"])

    def test_rejects_short_stint(self):
        result = extract_session_cliff_references(
            _synthetic_session_laps(count=6),
            _weather(count=6),
            _race(),
        )

        self.assertEqual(result["accepted_stints"], [])
        self.assertEqual(
            result["rejected_stints"][0]["stint_rejection_reason"],
            "insufficient_clean_laps",
        )

    def test_rejects_duplicate_manifest_race(self):
        manifest = _manifest()
        manifest["races"].append(copy.deepcopy(manifest["races"][0]))

        with self.assertRaisesRegex(
            CliffReferenceValidationError,
            "duplicate race id",
        ):
            validate_cliff_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
