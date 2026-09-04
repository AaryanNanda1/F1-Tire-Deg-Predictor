import unittest

import numpy as np
import pandas as pd

from analysis.feature_groups import TRACK_FEATURES, TRACK_AGE_INTERACTIONS, LEAKAGE_FEATURES
from analysis.run_final_model_selection import (
    ABLATIONS, REFERENCE, VARIANTS, apply_ablation, stage_features,
)


def frame():
    data = {"EventName": ["A", "A", "B", "B"], "EventDate": pd.date_range("2022-01-01", periods=4), "TyreLife": [1, 2, 1, 2], "LapTimeDelta": np.zeros(4)}
    for col in TRACK_FEATURES: data[col] = [0.1, 0.2, 0.3, 0.4]
    for col in TRACK_AGE_INTERACTIONS: data[col] = [0.1, 0.2, 0.3, 0.4]
    for col in ["Driver", "Team", "Compound", "TrackType", "SessionCode"]: data[col] = "x"
    for col in ["TyreLifeKM", "TyreLifeSquared", "TireAgeRatio", "LapNumber", "NormalizedLap", "LapsRemaining", "FuelLoad", "FuelLoadMissing", "Stint", "AirTemp", "TrackTemp"]: data[col] = 1.0
    return pd.DataFrame(data)


class FinalSelectionFeatureTests(unittest.TestCase):
    def test_stage_membership(self):
        data = frame()
        self.assertFalse(set(TRACK_FEATURES + TRACK_AGE_INTERACTIONS) & set(stage_features(data, "no_circuit")))
        self.assertTrue(set(TRACK_FEATURES).issubset(stage_features(data, "raw_circuit_7")))
        self.assertTrue(set(TRACK_AGE_INTERACTIONS).issubset(stage_features(data, "raw_circuit_age_interactions")))
        pca_features = stage_features(data, REFERENCE) + [f"PC{i}" for i in range(1, 5)]
        self.assertFalse(set(TRACK_FEATURES + TRACK_AGE_INTERACTIONS) & set(pca_features))

    def test_ablation_removes_source_and_derived_columns(self):
        features = stage_features(frame(), REFERENCE)
        self.assertNotIn("Driver", apply_ablation(features, "no_driver"))
        self.assertNotIn("Team", apply_ablation(features, "no_constructor"))
        for name in ("no_tire_age", "no_compound", "no_fuel_race_progress"):
            self.assertFalse(set(ABLATIONS[name]) & set(apply_ablation(features, name)))

    def test_pca_variant_has_no_prohibited_columns(self):
        features = stage_features(frame(), REFERENCE) + [f"PC{i}" for i in range(1, 5)]
        self.assertFalse(set(features) & (LEAKAGE_FEATURES - {"EventName", "EventDate"}))


if __name__ == "__main__":
    unittest.main()
