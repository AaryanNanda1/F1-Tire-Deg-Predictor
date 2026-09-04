#!/usr/bin/env python3
"""Ground Effect model-selection experiment and submission-bundle builder.

This module is deliberately analysis-only.  It evaluates the requested feature
sets with expanding event folds, fits PCA from training-fold circuit profiles,
and writes a self-contained report bundle.  It does not overwrite models/ or
change dashboard inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.feature_experiments import TrackPCATransformer
from analysis.feature_groups import (
    ALL_RAW_FEATURES, CATEGORICAL_FEATURES, LEAKAGE_FEATURES, TARGET,
    TRACK_AGE_INTERACTIONS, TRACK_FEATURES, assert_safe_features,
)
from train_era_models import make_production_hgb, PRODUCTION_HGB_PARAMS

VARIANTS = ["no_circuit", "raw_circuit_7", "raw_circuit_age_interactions", "pca4_no_age_interactions"]
REFERENCE = "pca4_no_age_interactions"
ABLATIONS = {
    "no_driver": ["Driver"],
    "no_constructor": ["Team"],
    "no_air_temperature": ["AirTemp"],
    "no_track_temperature": ["TrackTemp"],
    "no_fuel_race_progress": ["LapNumber", "NormalizedLap", "LapsRemaining", "FuelLoad", "FuelLoadMissing", "Stint"],
    "no_tire_age": ["TyreLife", "TyreLifeKM", "TyreLifeSquared", "TireAgeRatio"] + TRACK_AGE_INTERACTIONS + ["soft_age_interaction", "medium_age_interaction", "hard_age_interaction", "soft_abrasiveness_interaction", "soft_traction_interaction"],
    "no_compound": ["Compound", "soft_age_interaction", "medium_age_interaction", "hard_age_interaction", "soft_abrasiveness_interaction", "soft_traction_interaction"],
}


def load_data() -> pd.DataFrame:
    files = sorted((ROOT / "training_data/ground_effect").rglob("*.csv.gz"))
    if not files:
        raise FileNotFoundError("No Ground Effect processed sessions found")
    data = pd.concat((pd.read_csv(path, compression="gzip") for path in files), ignore_index=True)
    dates = pd.to_datetime(data["EventDate"], errors="coerce")
    data = data[dates.dt.year.between(2022, 2025)].copy()
    prohibited = sorted((set(data.columns) & LEAKAGE_FEATURES) - {TARGET, "EventName", "EventDate"})
    if prohibited:
        raise ValueError(f"Unexpected leakage columns in source data: {prohibited}")
    return data


def raw_features(frame: pd.DataFrame) -> list[str]:
    return [col for col in ALL_RAW_FEATURES if col in frame.columns]


def stage_features(frame: pd.DataFrame, variant: str) -> list[str]:
    features = raw_features(frame)
    if variant == "no_circuit":
        removed = TRACK_FEATURES + TRACK_AGE_INTERACTIONS
    elif variant == "raw_circuit_7":
        removed = TRACK_AGE_INTERACTIONS
    elif variant in {"raw_circuit_age_interactions", "full_pca4"}:
        removed = []
    elif variant in {"pca4_no_age_interactions", *ABLATIONS.keys()}:
        removed = TRACK_FEATURES + TRACK_AGE_INTERACTIONS
    else:
        raise ValueError(f"Unknown variant {variant}")
    return [col for col in features if col not in removed]


def apply_ablation(features: list[str], name: str | list[str] | None) -> list[str]:
    if not name:
        return features
    if isinstance(name, list):
        removed = name
    else:
        if name not in ABLATIONS:
            raise ValueError(f"Unknown ablation {name}")
        removed = ABLATIONS[name]
    return [col for col in features if col not in removed]


def encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_hgb(train_x: pd.DataFrame, train_y: pd.Series, weights: pd.Series | None = None) -> Pipeline:
    categorical = [col for col in CATEGORICAL_FEATURES if col in train_x.columns]
    numeric = [col for col in train_x.columns if col not in categorical]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder())]), categorical))
    model = Pipeline([
        ("preprocessor", ColumnTransformer(transforms, remainder="drop")),
        ("regressor", make_production_hgb()),
    ])
    kwargs = {"regressor__sample_weight": weights} if weights is not None else {}
    model.fit(train_x, train_y, **kwargs)
    return model


def prepare(train: pd.DataFrame, test: pd.DataFrame, variant: str, removed: str | list[str] | None = None):
    pca = None
    if variant == "pca4_no_age_interactions":
        pca = TrackPCATransformer(4).fit(train)
        train = pca.transform(train)
        test = pca.transform(test)
    features = stage_features(train, variant)
    if pca is not None:
        features += [f"PC{i}" for i in range(1, 5)]
    features = apply_ablation(features, removed)
    assert_safe_features(features)
    return train.reindex(columns=features), test.reindex(columns=features), pca


def events_for(data: pd.DataFrame, *, development: bool = True) -> pd.DataFrame:
    dates = pd.to_datetime(data["EventDate"], errors="coerce")
    source = data[dates.dt.year <= 2024] if development else data[dates.dt.year == 2025]
    events = source[["EventDate", "EventName"]].drop_duplicates().copy()
    events["EventDate"] = pd.to_datetime(events["EventDate"], errors="coerce")
    return events.dropna().sort_values(["EventDate", "EventName"]).reset_index(drop=True)


def metrics(y_true, pred, frame: pd.DataFrame) -> dict:
    mse = float(mean_squared_error(y_true, pred))
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "mse": mse, "rmse": float(np.sqrt(mse)),
        "degradation_slope_error": degradation_slope_error(frame, np.asarray(pred)),
    }


def bootstrap_ci(values: np.ndarray, seed: int = 42, draws: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    """Write a small Markdown table without requiring optional tabulate."""
    view = frame.copy()
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def degradation_slope_error(frame: pd.DataFrame, predictions: np.ndarray) -> float:
    """Mean absolute within-compound tire-age slope error."""
    scored = frame[["Compound", "TyreLife", TARGET]].copy()
    scored["prediction"] = predictions
    errors = []
    for _, group in scored.groupby("Compound"):
        if group["TyreLife"].nunique() < 3:
            continue
        actual = np.polyfit(group["TyreLife"], group[TARGET], 1)[0]
        predicted = np.polyfit(group["TyreLife"], group["prediction"], 1)[0]
        errors.append(abs(float(actual - predicted)))
    return float(np.mean(errors)) if errors else float("nan")


def evaluate(data: pd.DataFrame, output: Path, seed: int = 42, max_events: int | None = None, rows_per_event: int | None = None, variants: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dev = data[pd.to_datetime(data.EventDate).dt.year <= 2024].copy()
    frozen = data[pd.to_datetime(data.EventDate).dt.year == 2025].copy()
    events = events_for(data)
    eligible = []
    variants = variants or VARIANTS
    for _, event in events.iterrows():
        train = dev[pd.to_datetime(dev.EventDate) < event.EventDate]
        if train["EventName"].nunique() >= 7:
            eligible.append(event)
    events = pd.DataFrame(eligible, columns=events.columns)
    if max_events:
        events = events.iloc[-max_events:]
    rows, event_rows = [], []
    for _, event in events.iterrows():
        train = dev[pd.to_datetime(dev.EventDate) < event.EventDate]
        test = dev[(pd.to_datetime(dev.EventDate) == event.EventDate) & (dev.EventName == event.EventName)]
        if rows_per_event:
            test = test.head(rows_per_event)
        if train.empty or test.empty:
            continue
        fold = f"{event.EventDate.date()}::{event.EventName}"
        for variant in variants:
            started = time.perf_counter()
            train_x, test_x, _ = prepare(train, test, variant)
            model = fit_hgb(train_x, train[TARGET], train.get("SampleWeight"))
            pred = model.predict(test_x)
            summary = metrics(test[TARGET], pred, test)
            rows.append({"variant": variant, "split": "development", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "n_features": len(train_x.columns), "n_rows": len(test), "n_train_rows": len(train), "n_train_events": int(train[["EventDate", "EventName"]].drop_duplicates().shape[0]), "n_events": 1, "runtime_seconds": time.perf_counter() - started, **summary})
            for index, (_, observation) in enumerate(test.iterrows()):
                event_rows.append({"variant": variant, "split": "development", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "row_index": int(index), "y_true": float(observation[TARGET]), "prediction": float(pred[index]), "absolute_error": abs(float(observation[TARGET]) - float(pred[index]))})
            del model, train_x, test_x, pred
            gc.collect()
    # Evaluate each candidate on frozen 2025 after development selection, but
    # retain one metric row per actual event so event dispersion is genuine.
    for variant in variants:
        first_event = events_for(data, development=False).iloc[0]
        first_test = frozen[(pd.to_datetime(frozen.EventDate) == first_event.EventDate) & (frozen.EventName == first_event.EventName)]
        train_x, _, _ = prepare(dev, first_test, variant)
        model = fit_hgb(train_x, dev[TARGET], dev.get("SampleWeight"))
        for _, event in events_for(data, development=False).iterrows():
            test = frozen[(pd.to_datetime(frozen.EventDate) == event.EventDate) & (frozen.EventName == event.EventName)]
            if test.empty:
                continue
            _, test_x, _ = prepare(dev, test, variant)
            pred = model.predict(test_x); summary = metrics(test[TARGET], pred, test)
            fold = f"{event.EventDate.date()}::{event.EventName}"
            rows.append({"variant": variant, "split": "frozen_2025", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "n_features": len(train_x.columns), "n_rows": len(test), "n_train_rows": len(dev), "n_train_events": int(dev[["EventDate", "EventName"]].drop_duplicates().shape[0]), "n_events": 1, "runtime_seconds": 0.0, **summary})
            for index, (_, observation) in enumerate(test.iterrows()):
                event_rows.append({"variant": variant, "split": "frozen_2025", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "row_index": int(index), "y_true": float(observation[TARGET]), "prediction": float(pred[index]), "absolute_error": abs(float(observation[TARGET]) - float(pred[index]))})
        del model, train_x
        gc.collect()
    result = pd.DataFrame(rows)
    event_result = pd.DataFrame(event_rows)
    return result, event_result, {"development_folds": int(result[result.split == "development"].fold.nunique()), "frozen_events": int(frozen[["EventDate", "EventName"]].drop_duplicates().shape[0])}


def add_ablation_results(data: pd.DataFrame, result: pd.DataFrame, event_result: pd.DataFrame, seed: int = 42):
    dev = data[pd.to_datetime(data.EventDate).dt.year <= 2024].copy()
    events = events_for(data)
    eligible = []
    for _, event in events.iterrows():
        train = dev[pd.to_datetime(dev.EventDate) < event.EventDate]
        if train["EventName"].nunique() >= 7: eligible.append(event)
    added, event_added = [], []
    for _, event in pd.DataFrame(eligible, columns=events.columns).iterrows():
        train = dev[pd.to_datetime(dev.EventDate) < event.EventDate]
        test = dev[(pd.to_datetime(dev.EventDate) == event.EventDate) & (dev.EventName == event.EventName)]
        fold = f"{event.EventDate.date()}::{event.EventName}"
        for name in ABLATIONS:
            train_x, test_x, _ = prepare(train, test, REFERENCE, removed=name)
            model = fit_hgb(train_x, train[TARGET], train.get("SampleWeight"))
            pred = model.predict(test_x); summary = metrics(test[TARGET], pred, test)
            added.append({"variant": name, "split": "ablation_development", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "n_features": len(train_x.columns), "n_rows": len(test), "n_train_rows": len(train), "n_train_events": int(train[["EventDate", "EventName"]].drop_duplicates().shape[0]), "n_events": 1, "runtime_seconds": 0.0, **summary})
            for index, (_, observation) in enumerate(test.iterrows()):
                event_added.append({"variant": name, "split": "ablation_development", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "row_index": int(index), "y_true": float(observation[TARGET]), "prediction": float(pred[index]), "absolute_error": abs(float(observation[TARGET]) - float(pred[index]))})
            del model, train_x, test_x, pred; gc.collect()
    return pd.concat([result, pd.DataFrame(added)], ignore_index=True), pd.concat([event_result, pd.DataFrame(event_added)], ignore_index=True)


def evaluate_pca4_custom(data: pd.DataFrame, specs: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate combined/conditional PCA-4 removals with the same folds."""
    dev = data[pd.to_datetime(data.EventDate).dt.year <= 2024].copy()
    frozen = data[pd.to_datetime(data.EventDate).dt.year == 2025].copy()
    events = events_for(data)
    events = pd.DataFrame([event for _, event in events.iterrows() if dev[pd.to_datetime(dev.EventDate) < event.EventDate]["EventName"].nunique() >= 7], columns=events.columns)
    rows, event_rows = [], []
    for name, removed in specs.items():
        for _, event in events.iterrows():
            train = dev[pd.to_datetime(dev.EventDate) < event.EventDate]
            test = dev[(pd.to_datetime(dev.EventDate) == event.EventDate) & (dev.EventName == event.EventName)]
            if train.empty or test.empty: continue
            tx, vx, _ = prepare(train, test, REFERENCE, removed=removed); model = fit_hgb(tx, train[TARGET], train.get("SampleWeight")); pred = model.predict(vx)
            fold = f"{event.EventDate.date()}::{event.EventName}"; rows.append({"variant": name, "split": "conditional_development", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "n_features": len(tx.columns), "n_rows": len(test), "n_train_rows": len(train), "n_train_events": int(train[["EventDate", "EventName"]].drop_duplicates().shape[0]), "n_events": 1, "runtime_seconds": 0.0, **metrics(test[TARGET], pred, test)})
            for index, (_, observation) in enumerate(test.iterrows()): event_rows.append({"variant": name, "split": "conditional_development", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "row_index": index, "y_true": float(observation[TARGET]), "prediction": float(pred[index]), "absolute_error": abs(float(observation[TARGET]) - float(pred[index]))})
            del model, tx, vx, pred; gc.collect()
        # Frozen evaluation is intentionally performed only after the custom
        # candidate has been defined from development evidence.
        first = events_for(data, development=False).iloc[0]
        first_test = frozen[(pd.to_datetime(frozen.EventDate) == first.EventDate) & (frozen.EventName == first.EventName)]
        tx, _, _ = prepare(dev, first_test, REFERENCE, removed=removed); model = fit_hgb(tx, dev[TARGET], dev.get("SampleWeight"))
        for _, event in events_for(data, development=False).iterrows():
            test = frozen[(pd.to_datetime(frozen.EventDate) == event.EventDate) & (frozen.EventName == event.EventName)]
            if test.empty: continue
            _, vx, _ = prepare(dev, test, REFERENCE, removed=removed); pred = model.predict(vx); fold = f"{event.EventDate.date()}::{event.EventName}"
            rows.append({"variant": name, "split": "conditional_frozen_2025", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "n_features": len(tx.columns), "n_rows": len(test), "n_train_rows": len(dev), "n_train_events": int(dev[["EventDate", "EventName"]].drop_duplicates().shape[0]), "n_events": 1, "runtime_seconds": 0.0, **metrics(test[TARGET], pred, test)})
            for index, (_, observation) in enumerate(test.iterrows()): event_rows.append({"variant": name, "split": "conditional_frozen_2025", "fold": fold, "event_date": str(event.EventDate.date()), "event_name": event.EventName, "row_index": index, "y_true": float(observation[TARGET]), "prediction": float(pred[index]), "absolute_error": abs(float(observation[TARGET]) - float(pred[index]))})
        del model, tx; gc.collect()
    return pd.concat([pd.DataFrame(rows)], ignore_index=True), pd.concat([pd.DataFrame(event_rows)], ignore_index=True)


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    return result.groupby(["variant", "split"], as_index=False).agg(
        mae=("mae", "mean"), mse=("mse", "mean"), rmse=("rmse", "mean"),
        event_mae_median=("mae", "median"), event_mae_iqr=("mae", lambda x: float(x.quantile(.75) - x.quantile(.25))),
        degradation_slope_error=("degradation_slope_error", "mean"), event_count=("fold", "nunique"),
        feature_count=("n_features", "first"), row_count=("n_rows", "sum"),
    )


def figures(summary: pd.DataFrame, event_result: pd.DataFrame, output: Path):
    dev = summary[summary.split == "development"].copy()
    order = VARIANTS + list(ABLATIONS)
    dev["order"] = dev.variant.map({name: i for i, name in enumerate(order)})
    dev = dev.sort_values("order")
    for metric, filename, title in [("mae", "four_stage_performance.png", "Ground Effect development MAE"), ("rmse", "four_stage_rmse.png", "Ground Effect development RMSE")]:
        fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(dev.variant, dev[metric], color=["#9b2c2c" if v == REFERENCE else "#2f6f9f" for v in dev.variant]); ax.set_ylabel("Seconds per lap"); ax.set_title(title); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(output / "figures" / filename, dpi=300); plt.close(fig)
    # Paired event bootstrap ablations relative to PCA-4.
    piv = event_result[event_result.split.isin(["development", "ablation_development", "conditional_development"])].pivot_table(index="fold", columns="variant", values="absolute_error", aggfunc="mean")
    paired = []
    comparison_names = list(VARIANTS[:3]) + list(ABLATIONS) + ["pca4_pruned", "pca4_joint_no_temperatures", "pca4_joint_no_driver_constructor"]
    for name in comparison_names:
        if name not in piv or REFERENCE not in piv: continue
        diff = (piv[name] - piv[REFERENCE]).dropna()
        lo, hi = bootstrap_ci(diff.to_numpy())
        paired.append({"comparison": f"{name}_minus_{REFERENCE}", "baseline": REFERENCE, "candidate": name, "event_count": len(diff), "mean_difference": float(diff.mean()), "median_difference": float(diff.median()), "proportion_events_improved": float((diff < 0).mean()), "bootstrap_95ci_lower": lo, "bootstrap_95ci_upper": hi})
    paired_df = pd.DataFrame(paired); paired_df.to_csv(output / "tables" / "paired_event_mae_differences.csv", index=False); write_markdown(paired_df, output / "tables" / "paired_event_mae_differences.md")
    expected = set(ABLATIONS)
    if not expected.issubset(set(paired_df.candidate)):
        raise RuntimeError(f"Missing paired event ablation results: {sorted(expected - set(paired_df.candidate))}")
    ab = paired_df[paired_df.candidate.isin(ABLATIONS)]
    fig, ax = plt.subplots(figsize=(9, 5));
    if not ab.empty:
        y = np.arange(len(ab)); ax.errorbar(ab.mean_difference, y, xerr=[ab.mean_difference-ab.bootstrap_95ci_lower, ab.bootstrap_95ci_upper-ab.mean_difference], fmt="o", color="#c75c2c"); ax.set_yticks(y, ab.candidate)
    ax.axvline(0, color="black", linewidth=.8); ax.set_xlabel("ΔMAE: ablated − full PCA-4 (seconds/lap)"); ax.set_title("Feature ablation paired event MAE (95% bootstrap CI)"); ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(output / "figures" / "individual_ablation_delta_mae.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5));
    ax.scatter(dev.feature_count, dev.mae, color="#2f6f9f")
    for _, row in dev.iterrows(): ax.annotate(row.variant, (row.feature_count, row.mae), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Engineered feature count", ylabel="Development MAE (seconds/lap)", title="Performance versus feature complexity"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "figures" / "performance_vs_complexity.png", dpi=300); plt.close(fig)
    return paired_df


def pca_visuals(data: pd.DataFrame, output: Path):
    profiles = data.groupby("EventName", as_index=False)[TRACK_FEATURES].median()
    scaled = StandardScaler().fit_transform(profiles[TRACK_FEATURES]); pca = PCA().fit(scaled)
    variance = pd.DataFrame({"component": range(1, 8), "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(output / "tables" / "pca_explained_variance.csv", index=False); write_markdown(variance, output / "tables" / "pca_explained_variance.md")
    loadings = pd.DataFrame(pca.components_.T, index=TRACK_FEATURES, columns=[f"PC{i}" for i in range(1, 8)]); loadings.to_csv(output / "tables" / "pca_loadings.csv"); write_markdown(loadings.reset_index(names="feature"), output / "tables" / "pca_loadings.md")
    corr = profiles[TRACK_FEATURES].corr(method="spearman"); corr.to_csv(output / "tables" / "track_spearman.csv")
    for matrix, path, title in [(corr, "circuit_correlation_heatmap.png", "Circuit-characteristic Spearman correlation"), (loadings.iloc[:, :4], "pca4_loadings_heatmap.png", "PCA-4 circuit loadings")]:
        fig, ax = plt.subplots(figsize=(9, 6)); im=ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1); ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right"); ax.set_yticks(range(matrix.shape[0]), matrix.index); ax.set_title(title); fig.colorbar(im, ax=ax)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(output / "figures" / path, dpi=300); plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(8, 5)); x=variance.component; ax1.bar(x, variance.explained_variance_ratio, color="#2f6f9f", label="Individual"); ax1.set_xlabel("Principal component"); ax1.set_ylabel("Individual explained variance"); ax2=ax1.twinx(); ax2.plot(x, variance.cumulative_variance, marker="o", color="#c75c2c", label="Cumulative"); ax2.axvline(4, linestyle="--", color="black"); ax2.set_ylabel("Cumulative explained variance"); ax1.set_title("Circuit PCA variance (four components marked)"); fig.tight_layout(); fig.savefig(output / "figures" / "pca_scree_cumulative.png", dpi=300); plt.close(fig)
    return {"circuits": int(len(profiles)), "pc4_cumulative": float(variance.loc[variance.component == 4, "cumulative_variance"].iloc[0])}


class SubmissionCircuitPCA:
    """Persistable full-data PCA transformer for an analysis submission bundle."""
    def __init__(self, n_components=4): self.n_components = n_components
    def fit(self, X, y=None):
        profiles = X.groupby("EventName", as_index=False)[TRACK_FEATURES].median()
        self.scaler_ = StandardScaler().fit(profiles[TRACK_FEATURES])
        self.pca_ = PCA(n_components=self.n_components, random_state=42).fit(self.scaler_.transform(profiles[TRACK_FEATURES]))
        self.loadings_ = self.pca_.components_.T; self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        return self
    def transform(self, X):
        result = X.drop(columns=TRACK_FEATURES + TRACK_AGE_INTERACTIONS, errors="ignore").copy()
        pcs = self.pca_.transform(self.scaler_.transform(X[TRACK_FEATURES]))
        for i in range(self.n_components): result[f"PC{i+1}"] = pcs[:, i]
        return result
    def fit_transform(self, X, y=None): return self.fit(X, y).transform(X)


def train_bundle(data: pd.DataFrame, output: Path, selected: str, removed: list[str] | None = None):
    # Refit a production-shaped pipeline with the persisted full-data circuit transform.
    is_pca = selected in {REFERENCE, "pca4_pruned"}
    if is_pca:
        feature_names = stage_features(data, REFERENCE) + [f"PC{i}" for i in range(1, 5)]
        feature_names = apply_ablation(feature_names, removed)
        circuit = SubmissionCircuitPCA(4)
        transformed = circuit.fit_transform(data).reindex(columns=feature_names)
    else:
        feature_names = stage_features(data, selected)
        transformed = data.reindex(columns=feature_names)
        circuit = None
    model = fit_hgb(transformed, data[TARGET], data.get("SampleWeight"))
    # Compose the PCA step with the fitted preprocessing/model pipeline so the
    # saved bundle accepts canonical raw rows and never refits at inference.
    inference_model = Pipeline([("circuit_pca", circuit), *model.steps]) if circuit is not None else model
    bundle = {"model": inference_model, "circuit_pca": circuit, "input_features": (["EventName"] + raw_features(data)) if circuit is not None else feature_names, "transformed_features": feature_names, "selected_variant": selected, "target": TARGET, "pca_policy": "full-data submission fit; circuit median profiles, scaler and PCA persisted; no inference refit"}
    artifact = output / "artifacts" / "ground_effect_final_submission_bundle.joblib"; joblib.dump(bundle, artifact)
    manifest = {"selected_variant": selected, "feature_count": len(feature_names), "input_features": (["EventName"] + raw_features(data)) if circuit is not None else feature_names, "transformed_features": feature_names, "circuit_features": TRACK_FEATURES if circuit is not None else [], "pca_components": 4 if circuit is not None else 0, "pca_loadings": circuit.loadings_.tolist() if circuit is not None else [], "pca_explained_variance_ratio": circuit.explained_variance_ratio_.tolist() if circuit is not None else [], "rows": len(data), "events": int(data[["EventDate", "EventName"]].drop_duplicates().shape[0]), "training_cutoff": "2025-12-31", "random_seed": 42, "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "production_estimator_params": PRODUCTION_HGB_PARAMS, "artifact_sha256": None, "production_artifact_changed": False}
    manifest["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest(); (output / "artifacts" / "submission_manifest.json").write_text(json.dumps(manifest, indent=2)); return artifact, manifest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="reports/final_model_selection"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--skip-ablation", action="store_true"); parser.add_argument("--skip-artifact", action="store_true"); parser.add_argument("--smoke", action="store_true", help="Run two small chronological folds for infrastructure testing"); parser.add_argument("--max-events", type=int, default=None, help="Limit chronological development folds (for resumable pilots)"); parser.add_argument("--rows-per-event", type=int, default=None, help="Limit held-out rows per event (for smoke/pilot runs)"); parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None, help="Run only selected stage variants")
    args = parser.parse_args(); np.random.seed(args.seed)
    output = Path(args.output_dir); [ (output / name).mkdir(parents=True, exist_ok=True) for name in ("metrics", "tables", "figures", "artifacts") ]
    variants = args.variants or VARIANTS
    data = load_data(); result, event_result, fold_info = evaluate(data, output, args.seed, max_events=2 if args.smoke else args.max_events, rows_per_event=100 if args.smoke else args.rows_per_event, variants=variants)
    if not args.skip_ablation and not args.smoke:
        result, event_result = add_ablation_results(data, result, event_result, args.seed)
        base_summary = summarize(result)
        reference_mae = float(base_summary[(base_summary.variant == REFERENCE) & (base_summary.split == "development")].mae.iloc[0])
        supported = [name for name in ("no_driver", "no_constructor", "no_air_temperature", "no_track_temperature", "no_fuel_race_progress") if float(base_summary[(base_summary.variant == name) & (base_summary.split == "ablation_development")].mae.iloc[0]) <= reference_mae * 1.01]
        removed = sorted({column for name in supported for column in ABLATIONS[name]})
        custom = {"pca4_pruned": removed}
        if "no_air_temperature" in supported and "no_track_temperature" in supported:
            custom["pca4_joint_no_temperatures"] = sorted(set(ABLATIONS["no_air_temperature"] + ABLATIONS["no_track_temperature"]))
        if "no_driver" in supported and "no_constructor" in supported:
            custom["pca4_joint_no_driver_constructor"] = sorted(set(ABLATIONS["no_driver"] + ABLATIONS["no_constructor"]))
        custom_result, custom_events = evaluate_pca4_custom(data, custom)
        result = pd.concat([result, custom_result], ignore_index=True); event_result = pd.concat([event_result, custom_events], ignore_index=True)
    summary = summarize(result); result.to_csv(output / "metrics/model_comparison.csv", index=False); write_markdown(result, output / "metrics/model_comparison.md"); event_result.to_csv(output / "metrics/per_event_predictions.csv", index=False); summary.to_csv(output / "metrics/model_comparison_summary.csv", index=False); write_markdown(summary, output / "metrics/model_comparison_summary.md")
    paired = figures(summary, event_result, output); pca_info = pca_visuals(data, output)
    dev = summary[summary.split.isin(["development", "conditional_development"])].copy(); eligible_names = {"no_circuit", "raw_circuit_7", "raw_circuit_age_interactions", REFERENCE, "pca4_pruned"}; dev = dev[dev.variant.isin(eligible_names)].sort_values("mae"); selected = str(dev.iloc[0].variant) if not dev.empty else REFERENCE
    selected_removed = sorted({column for name in (supported if not args.skip_ablation and not args.smoke else []) for column in ABLATIONS[name]}) if selected == "pca4_pruned" else None
    if args.skip_artifact:
        artifact, artifact_manifest = None, {"status": "not_run", "reason": "--skip-artifact"}
    else:
        artifact, artifact_manifest = train_bundle(data, output, selected, selected_removed)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(); manifest_path = ROOT / "training_data/ground_effect/manifest.json"
    provenance = {"source_commit": source, "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "row_count": len(data), "folds": fold_info, "variants": VARIANTS + list(ABLATIONS) + ["pca4_pruned", "pca4_joint_no_temperatures", "pca4_joint_no_driver_constructor"], "feature_definitions": {name: stage_features(data, name) for name in VARIANTS}, "ablations": ABLATIONS, "pca_fitting_policy": "training-fold-only for validation; one median circuit profile per training circuit; submission bundle refit on all eligible 2022-2025 data", "seed": args.seed, "pca": pca_info, "selected_submission_variant": selected, "selection_note": "Selection uses development results only; frozen 2025 is reported after locking. Tire age and compound are diagnostic controls and remain eligible features.", "production_estimator_params": PRODUCTION_HGB_PARAMS, "production_artifact_changed": False, "active_aero_changed": False, "artifact": str(artifact), "artifact_manifest": artifact_manifest, "no_cliff_or_useful_life_retuning": True}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    figure_manifest = [
        {"filename": "figures/circuit_correlation_heatmap.png", "purpose": "Equal-weight circuit-characteristic correlation", "data_source": "one median profile per circuit", "recommended_caption": "Spearman correlation of the seven source-backed circuit characteristics.", "report_location": "supplementary"},
        {"filename": "figures/pca_scree_cumulative.png", "purpose": "PCA variance selection", "data_source": "one median profile per circuit", "recommended_caption": "Individual and cumulative explained variance; four components are marked.", "report_location": "main report"},
        {"filename": "figures/pca4_loadings_heatmap.png", "purpose": "PCA-4 interpretation", "data_source": "standardized circuit profiles", "recommended_caption": "PCA-4 loadings for the seven circuit characteristics.", "report_location": "main report"},
        {"filename": "figures/four_stage_performance.png", "purpose": "Circuit ablation comparison", "data_source": "chronological development folds", "recommended_caption": "Development MAE across the four circuit representations.", "report_location": "main report"},
        {"filename": "figures/individual_ablation_delta_mae.png", "purpose": "Input ablation comparison", "data_source": "paired chronological event errors", "recommended_caption": "Paired event-level ΔMAE; positive values indicate the removed feature block helped.", "report_location": "main report"},
        {"filename": "figures/performance_vs_complexity.png", "purpose": "Accuracy-complexity tradeoff", "data_source": "development summary", "recommended_caption": "Development MAE versus engineered feature count.", "report_location": "supplementary"},
    ]
    (output / "tables" / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2))
    guide = output / "REPORT_GUIDE.md"; guide.write_text(f"# Final Ground Effect model selection\n\nSelected submission variant: **{selected}**. The package compares four circuit representations, corrected PCA-4 feature ablations, conditional joint ablations, and a combined pruned candidate using chronological development folds through 2024 and a genuine per-event frozen 2025 evaluation.\n\nPCA describes variance in circuit descriptors; predictive value is established only through chronological model comparison. Positive ablation ΔMAE means removal worsened MAE. Tire-age and compound are diagnostic positive controls and remain required for compound-specific degradation predictions. Cliff thresholds were not retuned and useful-life ranges are operational ranges, not calibrated confidence intervals.\n\nSee `provenance.json`, `selection_decision.json`, `metrics/model_comparison_summary.csv`, and `artifacts/submission_manifest.json` for authoritative generated values.\n")
    decision = {"eligible_candidates": sorted(eligible_names), "selected_variant": selected, "development_selection_only": True, "selection_thresholds": {"primary": "lowest development MAE", "smaller_model_tolerance": 0.01, "bootstrap_draws": 2000}, "supported_removals": supported if not args.skip_ablation and not args.smoke else [], "rejected_alternatives": [name for name in eligible_names if name != selected], "rationale": "Selected by development MAE; secondary metrics and paired event results are reported separately. Frozen 2025 was not used for selection.", "seed": args.seed}
    (output / "selection_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps({"rows": len(data), "summary": str(output / 'metrics/model_comparison_summary.csv'), "artifact": str(artifact), "selected": selected}, indent=2))


if __name__ == "__main__": main()
