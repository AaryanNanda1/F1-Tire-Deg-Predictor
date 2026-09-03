import unittest

import numpy as np
import pandas as pd

from analysis.feature_experiments import PCA_AGE_INTERACTIONS, TrackPCATransformer, make_feature_set, prepare_variant
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

    def test_pca_age_interactions_are_exactly_the_requested_columns(self):
        train_x, test_x, _ = prepare_variant(fixture(), fixture(), "hgb_track_pca4_age_interactions")
        self.assertEqual([c for c in train_x if c.startswith("tire_age_x_PC")], PCA_AGE_INTERACTIONS)
        self.assertTrue(set(PCA_AGE_INTERACTIONS).issubset(test_x.columns))
        self.assertFalse(set(TRACK_FEATURES) & set(train_x.columns))
        plain_train_x, _, _ = prepare_variant(fixture(), fixture(), "hgb_track_pca4")
        self.assertFalse(set(PCA_AGE_INTERACTIONS) & set(plain_train_x.columns))

    def test_pca_and_interactions_do_not_use_prohibited_columns(self):
        frame = fixture()
        for column in LEAKAGE_FEATURES - {"EventName", "EventDate"}:
            frame[column] = 1.0
        for variant in ("hgb_raw", "hgb_track_pca4", "hgb_track_pca4_age_interactions"):
            train_x, test_x, _ = prepare_variant(frame, frame, variant)
            self.assertFalse(set(train_x.columns) & LEAKAGE_FEATURES)
            self.assertFalse(set(test_x.columns) & LEAKAGE_FEATURES)

    def test_pca_scaler_is_not_refit_on_held_out_values(self):
        training = fixture()
        held_out = fixture()
        held_out[TRACK_FEATURES[0]] = 10_000.0
        transformer = TrackPCATransformer(4).fit(training)
        expected_mean = training.groupby("EventName")[TRACK_FEATURES].median()[TRACK_FEATURES[0]].mean()
        self.assertAlmostEqual(float(transformer.scaler.mean_[0]), float(expected_mean))
