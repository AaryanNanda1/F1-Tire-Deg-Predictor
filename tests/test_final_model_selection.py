import unittest

import numpy as np
import pandas as pd

from analysis.feature_groups import TRACK_FEATURES, TRACK_AGE_INTERACTIONS, LEAKAGE_FEATURES
from analysis.run_final_model_selection import (
    ABLATIONS, REFERENCE, VARIANTS, apply_ablation, prepare, stage_features,
)
from train_era_models import select_active_aero_prior_weight, validate_active_aero_prior_weight_metadata


def frame():
    rows = 16
    data = {"EventName": [f"C{i // 2}" for i in range(rows)], "EventDate": pd.date_range("2022-01-01", periods=rows), "TyreLife": ([1, 2] * 8), "LapTimeDelta": np.zeros(rows)}
    for col in TRACK_FEATURES: data[col] = np.linspace(0.1, 0.9, rows)
    for col in TRACK_AGE_INTERACTIONS: data[col] = np.linspace(0.1, 0.9, rows)
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

    def test_every_pca4_ablation_contains_four_components(self):
        training = frame(); held_out = frame()
        for name in ABLATIONS:
            train_x, test_x, _ = prepare(training, held_out, REFERENCE, removed=name)
            self.assertEqual([f"PC{i}" for i in range(1, 5)], [c for c in train_x if c.startswith("PC")])
            self.assertFalse(set(TRACK_FEATURES) & set(train_x.columns))

    def test_pca_variant_has_no_prohibited_columns(self):
        features = stage_features(frame(), REFERENCE) + [f"PC{i}" for i in range(1, 5)]
        self.assertFalse(set(features) & (LEAKAGE_FEATURES - {"EventName", "EventDate"}))

    def test_active_aero_selected_and_fitted_prior_weight_must_match(self):
        candidates = {"0.20": {"status": "evaluated", "mae": 2.0}, "0.30": {"status": "evaluated", "mae": 1.9}}
        self.assertEqual(select_active_aero_prior_weight(candidates), 0.30)
        validate_active_aero_prior_weight_metadata({"active_aero_prior_weight_selected": 0.30, "active_aero_prior_weight_fitted": 0.30})
        with self.assertRaises(Exception):
            validate_active_aero_prior_weight_metadata({"active_aero_prior_weight_selected": 0.30, "active_aero_prior_weight_fitted": 0.20})


if __name__ == "__main__":
    unittest.main()
