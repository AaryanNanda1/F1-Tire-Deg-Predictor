#!/usr/bin/env python3
"""Reproduce leakage-safe feature analysis and model comparisons.

The runner writes only under the requested report directory. Production model
artifacts and processed stores are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from analysis.feature_experiments import (
    PCA_AGE_INTERACTION_VARIANT, event_keys, fit_model, make_feature_set, prepare_variant,
    raw_model_features, score_predictions,
)
from analysis.feature_groups import (
    ALL_RAW_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS, LEAKAGE_FEATURES, TARGET, TRACK_FEATURES,
)
from analysis.plotting import bar_plot, heatmap, line_plot, metric_dashboard


VARIANTS = ["hgb_raw", "hgb_pruned", "hgb_track_pca4", "hgb_track_pca5", "ridge_raw", "ridge_pca"]
PCA_SWEEP_VARIANTS = ["hgb_raw"] + [f"hgb_track_pca{i}" for i in range(1, 8)]
FOCUSED_PCA_VARIANTS = ["hgb_raw", "hgb_track_pca1", "hgb_track_pca2", "hgb_track_pca4"]
RAW_VS_PCA4_VARIANTS = ["hgb_raw", "hgb_track_pca4"]
PCA4_INTERACTION_VARIANTS = ["hgb_raw", "hgb_track_pca4", PCA_AGE_INTERACTION_VARIANT]


def load_store(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv.gz"))
    if not files: raise FileNotFoundError(f"No processed files found in {path}")
    return pd.concat([pd.read_csv(f, compression="gzip") for f in files], ignore_index=True, sort=False)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def correlations(data: pd.DataFrame, output: Path) -> None:
    dev = data[pd.to_datetime(data.EventDate).dt.year <= 2024].copy()
    numeric = [c for c in raw_model_features(dev) if c not in CATEGORICAL_FEATURES]
    for method in ("pearson", "spearman"):
        matrix = dev[numeric].corr(method=method)
        matrix.to_csv(output / "tables" / f"lap_numeric_{method}.csv")
        heatmap(matrix, f"Development lap-level {method.title()} correlations (n={len(dev):,})", output / "figures" / f"03_numeric_feature_redundancy_{method}.png")
        pairs = []
        for i, left in enumerate(numeric):
            for right in numeric[i + 1:]:
                pairs.append({"feature_a": left, "feature_b": right, "correlation": matrix.loc[left, right], "absolute_correlation": abs(matrix.loc[left, right])})
        pd.DataFrame(pairs).sort_values("absolute_correlation", ascending=False).to_csv(output / "tables" / f"lap_{method}_ranked_pairs.csv", index=False)
    track = dev.groupby("EventName", as_index=False)[TRACK_FEATURES].median()
    for method in ("pearson", "spearman"):
        matrix = track[TRACK_FEATURES].corr(method=method)
        matrix.to_csv(output / "tables" / f"track_{method}.csv")
        heatmap(matrix, f"Equal-weight circuit-profile {method.title()} correlations (n={len(track)})", output / "figures" / f"01_track_{method}.png", annotate=True)
    target_rows = []
    for col in numeric:
        target_rows.append({"feature": col, "pearson": dev[[col, TARGET]].corr(method="pearson").iloc[0, 1], "spearman": dev[[col, TARGET]].corr(method="spearman").iloc[0, 1]})
    pd.DataFrame(target_rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False).to_csv(output / "tables" / "target_associations.csv", index=False)


def degradation_slope_error(frame: pd.DataFrame, predictions: np.ndarray) -> float:
    """Compare within-stint tire-age slopes without using future rows as inputs."""
    scored = frame[["Compound", "TyreLife", TARGET]].copy()
    scored["prediction"] = predictions
    errors = []
    for _, group in scored.groupby("Compound"):
        if group["TyreLife"].nunique() < 3: continue
        actual = np.polyfit(group["TyreLife"], group[TARGET], 1)[0]
        predicted = np.polyfit(group["TyreLife"], group["prediction"], 1)[0]
        errors.append(abs(float(actual - predicted)))
    return float(np.mean(errors)) if errors else float("nan")


def pca_report(data: pd.DataFrame, output: Path) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    profiles = data.groupby("EventName", as_index=False)[TRACK_FEATURES].median()
    scaled = StandardScaler().fit_transform(profiles[TRACK_FEATURES])
    pca = PCA().fit(scaled)
    variance = pd.DataFrame({"component": np.arange(1, len(TRACK_FEATURES)+1), "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(output / "tables" / "pca_explained_variance.csv", index=False)
    loadings = pd.DataFrame(pca.components_.T, index=TRACK_FEATURES, columns=[f"PC{i}" for i in range(1, len(TRACK_FEATURES)+1)])
    loadings.to_csv(output / "tables" / "pca_loadings.csv")
    scores = pd.DataFrame(pca.transform(scaled)[:, :2], columns=["PC1", "PC2"]); scores["EventName"] = profiles.EventName
    scores.to_csv(output / "tables" / "pca_circuit_scores.csv", index=False)
    line_plot(variance.component, {"Explained variance": variance.explained_variance_ratio}, "Circuit PCA scree plot", "Principal component", "Explained variance ratio", output / "figures" / "04_pca_scree.png")
    line_plot(variance.component, {"Cumulative variance": variance.cumulative_variance}, "Circuit PCA cumulative variance", "Principal component", "Cumulative explained variance", output / "figures" / "05_pca_cumulative_variance.png")
    heatmap(loadings, "Standardized circuit PCA loadings", output / "figures" / "06_pca_loadings.png", annotate=True)
    fig_path = output / "figures" / "07_pca_circuit_scores.png"; import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7)); ax.scatter(scores.PC1, scores.PC2, color="#c75c2c")
    for _, row in scores.iterrows(): ax.annotate(str(row.EventName), (row.PC1, row.PC2), fontsize=7, xytext=(3,3), textcoords="offset points")
    ax.set(title=f"Circuit profiles in PCA space (n={len(scores)})", xlabel="PC1", ylabel="PC2"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(fig_path, dpi=300, bbox_inches="tight"); fig.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight'); plt.close(fig)
    return {"components": len(TRACK_FEATURES), "pc4_cumulative": float(variance.loc[variance.component == 4, "cumulative_variance"].iloc[0]), "pc5_cumulative": float(variance.loc[variance.component == 5, "cumulative_variance"].iloc[0]), "circuits": len(profiles)}


def _paired_event_differences(result: pd.DataFrame, output: Path) -> None:
    development = result[result["split"] == "development"]
    pivot = development.pivot_table(index="fold", columns="variant", values="mae", aggfunc="first")
    rows = []
    for baseline in ("hgb_raw", "hgb_track_pca4"):
        if PCA_AGE_INTERACTION_VARIANT not in pivot or baseline not in pivot:
            continue
        difference = (pivot[PCA_AGE_INTERACTION_VARIANT] - pivot[baseline]).dropna()
        rows.append({
            "candidate": PCA_AGE_INTERACTION_VARIANT,
            "baseline": baseline,
            "event_count": int(len(difference)),
            "mean_mae_difference_candidate_minus_baseline": float(difference.mean()),
            "median_mae_difference_candidate_minus_baseline": float(difference.median()),
            "proportion_events_improved": float((difference < 0).mean()) if len(difference) else None,
        })
    pd.DataFrame(rows).to_csv(output / "metrics" / "paired_event_mae_differences.csv", index=False)
    if PCA_AGE_INTERACTION_VARIANT in pivot and "hgb_track_pca4" in pivot:
        difference = (pivot[PCA_AGE_INTERACTION_VARIANT] - pivot["hgb_track_pca4"]).dropna().sort_index()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4.5))
        colors = ["#2f8f5b" if value < 0 else "#c75c2c" for value in difference]
        ax.bar(np.arange(len(difference)), difference.values, color=colors)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title("Development event MAE difference: PCA-4 + age interactions − PCA-4")
        ax.set_ylabel("MAE difference (seconds; negative favors interactions)")
        ax.set_xlabel("Chronological development event fold")
        ax.set_xticks(np.arange(len(difference)), [str(index).split("::", 1)[-1] for index in difference.index], rotation=75, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=.25); fig.tight_layout()
        path = output / "figures" / "pca4_age_interaction_event_mae_difference.png"
        fig.savefig(path, dpi=300, bbox_inches="tight"); fig.savefig(path.with_suffix(".svg"), bbox_inches="tight"); plt.close(fig)


def evaluate(data: pd.DataFrame, output: Path, smoke: bool, random_state: int, *, pca_sweep: bool = False, focused_pca: bool = False, raw_vs_pca4: bool = False, pca4_interactions: bool = False) -> dict:
    dates = pd.to_datetime(data.EventDate)
    dev = data[dates.dt.year <= 2024].copy(); final = data[dates.dt.year == 2025].copy()
    if smoke:
        # Keep smoke tests fast while retaining chronological event structure.
        dev = dev.groupby(["EventDate", "EventName"], group_keys=False).head(250).copy()
        final = final.groupby(["EventDate", "EventName"], group_keys=False).head(250).copy()
    events = event_keys(dev)
    # A common fold must support the largest candidate (PCA-7). Early-season
    # folds with fewer than seven distinct training circuits cannot fit that
    # candidate without silently changing the representation.
    eligible = []
    for _, candidate in events.iterrows():
        prior = dev[pd.to_datetime(dev.EventDate) < candidate.EventDate]
        if prior["EventName"].nunique() >= 7:
            eligible.append(candidate)
    events = pd.DataFrame(eligible, columns=events.columns)
    if smoke: events = events.iloc[-min(4, len(events)):]
    rows = []; importance_rows = []; fold_ids = []
    final_predictions = None
    ablation_specs = {
        "remove_tyre_age": ["TyreLife", "TyreLifeKM", "TyreLifeSquared", "TireAgeRatio"],
        "remove_race_progress": ["LapNumber", "NormalizedLap", "LapsRemaining", "FuelLoad", "Stint"],
        "remove_weather": ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed", "IsWet", "FuelLoadMissing"],
        "remove_circuit": TRACK_FEATURES,
        "remove_track_age_interactions": ["tire_age_x_abrasiveness", "track_temp_x_tyre_stress", "tire_age_x_traction", "tire_age_x_lateral_load", "soft_abrasiveness_interaction", "soft_traction_interaction"],
        "remove_compound_age_interactions": ["soft_age_interaction", "medium_age_interaction", "hard_age_interaction"],
        "remove_driver_team": ["Driver", "Team"],
        "remove_session_context": ["SessionCode", "TrackType"],
    }
    variants = PCA4_INTERACTION_VARIANTS if pca4_interactions else (RAW_VS_PCA4_VARIANTS if raw_vs_pca4 else (FOCUSED_PCA_VARIANTS if focused_pca else (PCA_SWEEP_VARIANTS if pca_sweep else (["hgb_raw", "hgb_track_pca4"] if smoke else VARIANTS))))
    if smoke:
        ablation_specs = dict(list(ablation_specs.items())[:2])
    if pca_sweep or focused_pca or raw_vs_pca4 or pca4_interactions:
        ablation_specs = {}
    for _, test_event in events.iterrows():
        train = dev[pd.to_datetime(dev.EventDate) < test_event.EventDate]
        test = dev[(pd.to_datetime(dev.EventDate) == test_event.EventDate) & (dev.EventName == test_event.EventName)]
        if train.empty or test.empty: continue
        fold_id = f"{test_event.EventDate.date()}::{test_event.EventName}"; fold_ids.append(fold_id)
        for variant in variants:
            started = time.perf_counter(); train_x, test_x, transformer = prepare_variant(train, test, variant)
            model = fit_model(train_x, train[TARGET], variant, train.get("SampleWeight"))
            predictions = model.predict(test_x); metrics = score_predictions(test[TARGET], predictions)
            rows.append({"variant": variant, "fold": fold_id, "event_date": str(test_event.EventDate.date()), "event_name": test_event.EventName, "split": "development", "n_features": train_x.shape[1], "degradation_slope_error": degradation_slope_error(test, predictions), **metrics, "runtime_seconds": time.perf_counter() - started})
            if variant == "hgb_raw" and smoke:
                perm = permutation_importance(model, test_x, test[TARGET], scoring="neg_mean_absolute_error", n_repeats=3, random_state=random_state)
                for name, value, std in zip(train_x.columns, perm.importances_mean, perm.importances_std): importance_rows.append({"fold": fold_id, "feature": name, "importance_in_mae": -float(value), "std": float(std)})
        baseline_features = make_feature_set(train, "hgb_raw")
        for experiment, removed in ablation_specs.items():
            features = [column for column in baseline_features if column not in removed]
            model = fit_model(train.reindex(columns=features), train[TARGET], "hgb_raw", train.get("SampleWeight"))
            predictions = model.predict(test.reindex(columns=features))
            rows.append({"variant": experiment, "fold": fold_id, "split": "ablation_development", "n_features": len(features), "degradation_slope_error": degradation_slope_error(test, predictions), **score_predictions(test[TARGET], predictions), "runtime_seconds": 0.0})
    # One final chronological 2025 evaluation after development is frozen.
    if not final.empty and not dev.empty:
        for variant in variants:
            train_x, test_x, _ = prepare_variant(dev, final, variant); model = fit_model(train_x, dev[TARGET], variant, dev.get("SampleWeight")); predictions = model.predict(test_x); rows.append({"variant": variant, "fold": "2025-final", "split": "final_2025", "n_features": train_x.shape[1], "degradation_slope_error": degradation_slope_error(final, predictions), **score_predictions(final[TARGET], predictions), "runtime_seconds": 0.0})
            if variant == "hgb_raw":
                final_predictions = final[["Compound", "TyreLife", TARGET]].copy()
                final_predictions["prediction"] = predictions
    result = pd.DataFrame(rows); result.to_csv(output / "metrics" / "model_comparison.csv", index=False)
    if not result.empty:
        summary = result.groupby(["variant", "split"], as_index=False).agg(mae=("mae", "mean"), mse=("mse", "mean"), rmse=("rmse", "mean"), event_mae_median=("mae", "median"), event_mae_iqr=("mae", lambda values: float(values.quantile(.75) - values.quantile(.25))), degradation_slope_error=("degradation_slope_error", "mean"), events=("fold", "nunique"), feature_count=("n_features", "first"))
        summary.to_csv(output / "metrics" / "model_comparison_summary.csv", index=False)
        dev_summary = summary[summary.split == "development"]; bar_plot(dev_summary, "variant", "mae", "Development event-fold MAE", "Mean absolute error (seconds)", output / "figures" / "08_model_mae_comparison.png"); bar_plot(dev_summary, "variant", "rmse", "Development event-fold RMSE", "RMSE (seconds)", output / "figures" / "09_model_rmse_comparison.png")
        metric_dashboard(dev_summary, "Development model metric comparison", output / "figures" / "15_all_metrics_comparison.png")
        ablation_summary = summary[summary.split == "ablation_development"]
        if not ablation_summary.empty:
            ablation_summary.to_csv(output / "metrics" / "ablation_summary.csv", index=False)
            bar_plot(ablation_summary, "variant", "mae", "Grouped feature ablation MAE", "Mean absolute error (seconds)", output / "figures" / "10_ablation_results.png")
    if pca4_interactions:
        _paired_event_differences(result, output)
    pd.DataFrame(importance_rows).to_csv(output / "metrics" / "permutation_importance_folds.csv", index=False)
    if importance_rows:
        imp = pd.DataFrame(importance_rows).groupby("feature", as_index=False).agg(mean_increase_mae=("importance_in_mae", "mean"), std_across_folds=("importance_in_mae", "std"), events=("fold", "nunique")).sort_values("mean_increase_mae", ascending=False)
        imp.to_csv(output / "metrics" / "permutation_importance_aggregated.csv", index=False); bar_plot(imp.head(15), "feature", "mean_increase_mae", "Held-out permutation importance (HGB raw)", "Increase in MAE (seconds)", output / "figures" / "11_permutation_importance.png")
    if final_predictions is not None:
        compound_rows = []
        for name, group in final_predictions.groupby("Compound"):
            compound_rows.append({"Compound": name, **score_predictions(group[TARGET], group.prediction)})
        compound = pd.DataFrame(compound_rows)
        compound.to_csv(output / "tables" / "mae_by_compound.csv", index=False)
        bar_plot(compound, "Compound", "mae", "Frozen 2025 MAE by compound", "Mean absolute error (seconds)", output / "figures" / "12_mae_by_compound.png")
        final_predictions["tyre_age_band"] = pd.cut(final_predictions["TyreLife"], [0, 5, 10, 15, 20, np.inf], labels=["1-5", "6-10", "11-15", "16-20", "21+"])
        age_rows = []
        for name, group in final_predictions.groupby("tyre_age_band", observed=False):
            if len(group): age_rows.append({"tyre_age_band": str(name), **score_predictions(group[TARGET], group.prediction)})
        age = pd.DataFrame(age_rows)
        age.to_csv(output / "tables" / "mae_by_tyre_age.csv", index=False)
        bar_plot(age, "tyre_age_band", "mae", "Frozen 2025 MAE by tire-age band", "Mean absolute error (seconds)", output / "figures" / "13_mae_by_tyre_age.png")
        curve = final_predictions.groupby(["Compound", "TyreLife"], as_index=False)[[TARGET, "prediction"]].mean()
        curve.to_csv(output / "tables" / "predicted_vs_observed_degradation.csv", index=False)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for compound_name, group in curve.groupby("Compound"):
            axes[0].plot(group.TyreLife, group[TARGET], marker="o", label=compound_name)
            axes[1].plot(group.TyreLife, group.prediction, marker="o", label=compound_name)
        axes[0].set_title("Observed mean degradation"); axes[1].set_title("Predicted mean degradation")
        for axis in axes: axis.set_xlabel("Tire age (laps)"); axis.set_ylabel("Lap-time delta (seconds)"); axis.grid(alpha=.25); axis.legend(fontsize=8)
        fig.suptitle("Frozen 2025 degradation curves"); fig.tight_layout(); path = output / "figures" / "14_predicted_vs_observed_degradation.png"; fig.savefig(path, dpi=300, bbox_inches="tight"); fig.savefig(path.with_suffix(".svg"), bbox_inches="tight"); plt.close(fig)
    (output / "metrics" / "downstream_benchmarks.json").write_text(json.dumps({"status": "not_run", "reason": "Existing cliff and strategy benchmarks require simulation-specific curve and race-state inputs; no production behavior was changed by this analysis runner."}, indent=2))
    return {"development_folds": len(fold_ids), "rows": len(result), "selected_variant": "hgb_raw"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--era", default="ground_effect", choices=["ground_effect"]); parser.add_argument("--output-dir", default="reports/feature_engineering"); parser.add_argument("--random-state", type=int, default=42); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--pca-sweep", action="store_true", help="Compare PCA-1 through PCA-7 against hgb_raw"); parser.add_argument("--focused-pca", action="store_true", help="Full-data second stage: raw, PCA-1, PCA-2, and PCA-4"); parser.add_argument("--raw-vs-pca4", action="store_true", help="Full-data comparison of only hgb_raw and PCA-4"); parser.add_argument("--pca4-interactions", action="store_true", help="Compare raw, PCA-4, and PCA-4 with tire-age interactions"); args = parser.parse_args()
    np.random.seed(args.random_state); output = Path(args.output_dir); (output / "figures").mkdir(parents=True, exist_ok=True); (output / "tables").mkdir(exist_ok=True); (output / "metrics").mkdir(exist_ok=True)
    data = load_store(PROJECT_ROOT / "training_data/ground_effect"); years = pd.to_datetime(data.EventDate).dt.year; data = data[years.between(2022, 2025)].copy()
    prohibited = sorted(set(data.columns) & LEAKAGE_FEATURES - {TARGET, "EventName", "EventDate"});
    if prohibited: raise ValueError(f"Unexpected prohibited columns present: {prohibited}")
    correlations(data, output); pca = pca_report(data, output); evaluation = evaluate(data, output, args.smoke, args.random_state, pca_sweep=args.pca_sweep, focused_pca=args.focused_pca, raw_vs_pca4=args.raw_vs_pca4, pca4_interactions=args.pca4_interactions)
    source_commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    provenance = {"source_commit": source_commit, "baseline_source_commit": "4b2c3ea0e92486ba93d7d0c2754b1e90093232da", "generation_timestamp_utc": pd.Timestamp.utcnow().isoformat(), "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "random_state": args.random_state, "input_manifest": "training_data/ground_effect/manifest.json", "input_manifest_sha256": hash_file(PROJECT_ROOT / "training_data/ground_effect/manifest.json"), "dataset_years": [2022, 2023, 2024, 2025], "row_count": len(data), "session_count": int(data[["EventDate", "EventName"]].drop_duplicates().shape[0]), "feature_groups": FEATURE_GROUPS, "variants": {"hgb_raw": make_feature_set(data, "hgb_raw"), "hgb_track_pca4": make_feature_set(data, "hgb_track_pca4") + [f"PC{i}" for i in range(1, 5)], PCA_AGE_INTERACTION_VARIANT: make_feature_set(data, "hgb_track_pca4") + [f"PC{i}" for i in range(1, 5)] + [f"tire_age_x_PC{i}" for i in range(1, 5)]}, "pca_fitting_policy": "Each fold fits StandardScaler and PCA on one median profile per circuit using training-fold rows only; held-out rows are transformed with those fitted objects.", "folds": evaluation, "pca": pca, "production_artifact_changed": False, "production_training_or_inference_changed": False}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str)); print(json.dumps({"rows": len(data), "pca": pca, "evaluation": evaluation}, indent=2))


if __name__ == "__main__": main()
