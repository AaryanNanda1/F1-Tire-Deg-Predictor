"""Leakage-safe fold transforms and model experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_groups import (
    AGE_REDUNDANCY_REMOVALS, ALL_RAW_FEATURES, CATEGORICAL_FEATURES,
    LEAKAGE_FEATURES, TRACK_AGE_INTERACTIONS, TRACK_FEATURES, assert_safe_features,
)


PCA_AGE_INTERACTION_VARIANT = "hgb_track_pca4_age_interactions"
PCA_AGE_INTERACTIONS = [f"tire_age_x_PC{i}" for i in range(1, 5)]


def raw_model_features(frame: pd.DataFrame) -> list[str]:
    present = [c for c in ALL_RAW_FEATURES if c in frame.columns]
    assert_safe_features(present)
    return present


def make_feature_set(frame: pd.DataFrame, variant: str) -> list[str]:
    features = raw_model_features(frame)
    if variant == "hgb_raw" or variant == "ridge_raw":
        return features
    if variant == "hgb_pruned":
        return [c for c in features if c not in AGE_REDUNDANCY_REMOVALS]
    if variant.startswith("hgb_track_pca") or variant.startswith("ridge_pca"):
        return [c for c in features if c not in TRACK_FEATURES + TRACK_AGE_INTERACTIONS]
    raise ValueError(f"Unknown feature variant: {variant}")


def _encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_model(X: pd.DataFrame, y: pd.Series, variant: str, weights: pd.Series | None = None):
    categorical = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler() if variant.startswith("ridge") else "passthrough"),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _encoder()),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    if variant.startswith("ridge"):
        estimator = Ridge(alpha=10.0)
    else:
        estimator = HistGradientBoostingRegressor(
            # Analysis-only screening configuration. Production training uses
            # the separate trainer and is never changed by this module.
            loss="absolute_error", max_iter=12, max_leaf_nodes=15, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10, random_state=42,
        )
    model = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
    fit_kwargs = {"regressor__sample_weight": weights} if weights is not None else {}
    model.fit(X, y, **fit_kwargs)
    return model


@dataclass
class TrackPCATransformer:
    """Fits track PCA on unique training-fold circuit profiles only."""

    n_components: int
    event_column: str = "EventName"

    def fit(self, training: pd.DataFrame) -> "TrackPCATransformer":
        if self.event_column not in training.columns:
            raise ValueError("EventName is required to fit circuit profiles")
        profiles = training.groupby(self.event_column, as_index=False)[TRACK_FEATURES].median()
        self.scaler = StandardScaler().fit(profiles[TRACK_FEATURES])
        self.pca = PCA(n_components=self.n_components, random_state=42).fit(
            self.scaler.transform(profiles[TRACK_FEATURES])
        )
        self.loadings_ = pd.DataFrame(
            self.pca.components_.T,
            index=TRACK_FEATURES,
            columns=[f"PC{i}" for i in range(1, self.n_components + 1)],
        )
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = self.pca.transform(self.scaler.transform(frame[TRACK_FEATURES]))
        result = frame.drop(columns=TRACK_FEATURES + TRACK_AGE_INTERACTIONS, errors="ignore").copy()
        for index in range(self.n_components):
            result[f"PC{index + 1}"] = values[:, index]
        return result


def _pca_component_count(variant: str) -> int | None:
    match = re.search(r"(?:hgb_track_pca|ridge_pca)(\d+)", variant)
    return int(match.group(1)) if match else None


def _add_pca_age_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for index in range(1, 5):
        result[f"tire_age_x_PC{index}"] = result["TyreLife"] * result[f"PC{index}"]
    return result


def prepare_variant(train: pd.DataFrame, test: pd.DataFrame, variant: str):
    transformer = None
    if variant.startswith("hgb_track_pca") or variant.startswith("ridge_pca"):
        components = _pca_component_count(variant)
        if components is None:
            raise ValueError(f"Could not determine PCA component count for {variant}")
        transformer = TrackPCATransformer(components).fit(train)
        train = transformer.transform(train)
        test = transformer.transform(test)
        if variant == PCA_AGE_INTERACTION_VARIANT:
            if transformer.n_components != 4:
                raise ValueError("PCA-4 age interactions require exactly four components")
            train = _add_pca_age_interactions(train)
            test = _add_pca_age_interactions(test)
    features = make_feature_set(train, variant)
    if transformer is not None:
        features += [f"PC{i}" for i in range(1, transformer.n_components + 1)]
    if variant == PCA_AGE_INTERACTION_VARIANT:
        features += PCA_AGE_INTERACTIONS
    assert_safe_features(features)
    return train.reindex(columns=features), test.reindex(columns=features), transformer


def score_predictions(y_true, predicted) -> dict:
    mse = float(mean_squared_error(y_true, predicted))
    return {
        "mae": float(mean_absolute_error(y_true, predicted)),
        "mse": mse,
        "rmse": mse ** 0.5,
    }


def event_keys(frame: pd.DataFrame) -> pd.DataFrame:
    events = frame[["EventDate", "EventName"]].drop_duplicates().copy()
    events["EventDate"] = pd.to_datetime(events["EventDate"], errors="coerce")
    return events.dropna().sort_values(["EventDate", "EventName"]).reset_index(drop=True)
