import unittest

import numpy as np
import pandas as pd

from analysis.feature_experiments import TrackPCATransformer, make_feature_set, prepare_variant
from analysis.feature_groups import LEAKAGE_FEATURES, TRACK_FEATURES


def fixture(rows=20):
    frame = pd.DataFrame({"EventName": [f"Circuit {i // 4}" for i in range(rows)], "EventDate": pd.date_range("2022-01-01", periods=rows), "LapTimeDelta": np.arange(rows, dtype=float)})
    for i, feature in enumerate(TRACK_FEATURES): frame[feature] = np.linspace(0.1 + i * .01, .9 - i * .005, rows)
    frame["Driver"] = "DRV"; frame["Team"] = "TEAM"; frame["Compound"] = "MEDIUM"; frame["TrackType"] = "Fast"; frame["SessionCode"] = "R"; frame["TyreLife"] = np.arange(1, rows + 1); frame["TyreLifeKM"] = frame.TyreLife * 5; frame["TyreLifeSquared"] = frame.TyreLife ** 2; frame["TireAgeRatio"] = frame.TyreLife / 50
    return frame


class FeatureAnalysisTests(unittest.TestCase):
    def test_leakage_features_are_not_model_features(self):
        features = make_feature_set(fixture(), "hgb_raw")
        self.assertFalse(set(features) & LEAKAGE_FEATURES)

    def test_track_pca_uses_unique_circuit_profiles(self):
        transformer = TrackPCATransformer(2).fit(fixture())
        self.assertEqual(len(transformer.loadings_), 7)
        self.assertEqual(len(transformer.explained_variance_ratio_), 2)

    def test_pca_removes_raw_track_features(self):
        transformed = TrackPCATransformer(2).fit(fixture()).transform(fixture())
        self.assertFalse(set(TRACK_FEATURES) & set(transformed.columns))

    def test_pca_is_fitted_from_training_frame_only(self):
        training = fixture()
        held_out = fixture()
        held_out["traction"] = 100.0
        train_x, test_x, transformer = prepare_variant(training, held_out, "hgb_track_pca4")
        self.assertEqual(train_x.shape[1], test_x.shape[1])
        self.assertEqual(transformer.pca.n_components_, 4)
        self.assertLess(abs(float(train_x["PC1"].mean())), 10.0)
