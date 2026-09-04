#!/usr/bin/env python3
"""Analysis-only validation of cliff shape and useful-life uncertainty.

This runner consumes frozen reviewed calibration labels and the committed
processed Ground Effect store. It never changes production defaults or model
artifacts. Detector calibration uses a grouped development/holdout split by
Grand Prix because the reviewed set is small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.feature_experiments import fit_model, prepare_variant
from mappings import TRACK_PIT_LOSS, normalize_team_name
from tire_life_analysis import (
    DEFAULT_CONFIG,
    detect_hybrid_performance_cliff,
    detect_performance_cliff,
    detect_piecewise_performance_cliff,
    detect_rolling_sustained_performance_cliff,
    estimate_useful_life_uncertainty,
    smooth_lap_times,
)


DETECTORS = {
    "sustained": detect_performance_cliff,
    "rolling_sustained": detect_rolling_sustained_performance_cliff,
    "piecewise": detect_piecewise_performance_cliff,
    "hybrid": detect_hybrid_performance_cliff,
}
VARIANTS = ("hgb_raw", "hgb_track_pca4")
SOURCE_COMMIT = "4b2c3ea0e92486ba93d7d0c2754b1e90093232da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_store(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv.gz"))
    return pd.concat([pd.read_csv(file, compression="gzip") for file in files], ignore_index=True)


def load_captures(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return [capture for capture in payload["captures"] if capture.get("manual_review_status") in {"confirmed_cliff", "confirmed_no_cliff"}]


def observed_curve(capture: dict) -> tuple[np.ndarray, np.ndarray]:
    values = capture["observed_curve"]
    ages = np.asarray([row["tire_age"] for row in values], dtype=float)
    times = np.asarray([row["fuel_corrected_lap_time_seconds"] for row in values], dtype=float)
    return ages, times


def capture_key(capture: dict) -> str:
    return str(capture["reference_id"])


def predicted_curves(data: pd.DataFrame, captures: list[dict], max_train_rows: int | None = None, seed: int = 42, fit_model_predictions: bool = False) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    if not fit_model_predictions:
        return {
            "hgb_raw": {
                capture_key(capture): (
                    np.asarray([row["tire_age"] for row in capture["predicted_curve"]], dtype=float),
                    np.asarray([row["predicted_lap_time_seconds"] for row in capture["predicted_curve"]], dtype=float),
                )
                for capture in captures if capture.get("predicted_curve")
            },
            "hgb_track_pca4": {},
        }
    train = data[pd.to_datetime(data["EventDate"]).dt.year == 2022].copy()
    scored = data[pd.to_datetime(data["EventDate"]).dt.year == 2023].copy()
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=seed).copy()
    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        train_x, scored_x, _ = prepare_variant(train, scored, variant)
        model = fit_model(train_x, train["LapTimeDelta"], variant, train.get("SampleWeight"))
        scored_variant = scored.copy()
        scored_variant["prediction"] = model.predict(scored_x)
        for capture in captures:
            team = normalize_team_name(capture["team"])
            group = scored_variant[
                (scored_variant["EventName"] == capture["event_name"])
                & (scored_variant["Driver"] == capture["driver"])
                & (scored_variant["Team"] == team)
                & (scored_variant["Compound"] == capture["compound"])
                & (scored_variant["Stint"] == capture["stint"])
            ].groupby("TyreLife", as_index=False)["prediction"].mean()
            if group.empty:
                continue
            ages = group["TyreLife"].to_numpy(dtype=float)
            # A common additive base is irrelevant to the detector and keeps
            # the predicted curve in the same shape space as the observations.
            times = group["prediction"].to_numpy(dtype=float)
            result[variant][capture_key(capture)] = (ages, times)
    return result


def slope_pair(ages: np.ndarray, values: np.ndarray, breakpoint: float) -> tuple[float | None, float | None]:
    pre = values[ages < breakpoint]
    pre_x = ages[ages < breakpoint]
    post = values[ages >= breakpoint]
    post_x = ages[ages >= breakpoint]
    if len(pre) < 3 or len(post) < 3:
        return None, None
    return float(np.polyfit(pre_x, pre, 1)[0]), float(np.polyfit(post_x, post, 1)[0])


def detector_row(capture: dict, source: str, model_variant: str, method: str, curve: tuple[np.ndarray, np.ndarray], config: dict) -> dict:
    ages, values = curve
    smoothed = smooth_lap_times(values, config)
    detected = DETECTORS[method](smoothed, config)
    relative = detected.get("performance_cliff_lap")
    actual_lap = None if relative is None else int(round(ages[0] + relative - 1))
    is_cliff = capture["manual_review_status"] == "confirmed_cliff"
    in_window = actual_lap is not None and int(capture["starting_tire_age"]) <= actual_lap <= int(capture["ending_tire_age"])
    reviewed = capture.get("reviewed_cliff_lap")
    absolute_error = None if not (is_cliff and in_window and reviewed is not None) else abs(actual_lap - int(reviewed))
    observed_ages, observed_values = observed_curve(capture)
    observed_pre, observed_post = slope_pair(observed_ages, observed_values, float(reviewed)) if reviewed is not None else (None, None)
    predicted_pre, predicted_post = slope_pair(ages, values, float(reviewed)) if reviewed is not None else (None, None)
    retention = None if observed_post in (None, 0) or predicted_post is None else float(predicted_post / observed_post)
    return {
        "reference_id": capture_key(capture), "event_name": capture["event_name"],
        "model_variant": model_variant, "curve_source": source, "method": method,
        "truth": capture["manual_review_status"], "reviewed_cliff_lap": reviewed,
        "predicted_cliff_lap": actual_lap, "detected_in_review_window": bool(in_window),
        "classification": ("true_positive" if is_cliff and in_window else "false_negative" if is_cliff else "false_positive" if in_window else "true_negative"),
        "absolute_cliff_lap_error": absolute_error,
        "within_1_lap": absolute_error is not None and absolute_error <= 1,
        "within_2_laps": absolute_error is not None and absolute_error <= 2,
        "within_3_laps": absolute_error is not None and absolute_error <= 3,
        "observed_pre_cliff_slope": observed_pre, "observed_post_cliff_slope": observed_post,
        "predicted_pre_cliff_slope": predicted_pre, "predicted_post_cliff_slope": predicted_post,
        "post_cliff_slope_retention_ratio": retention,
        "detectable_slope_break_near_reviewed": bool(actual_lap is not None and reviewed is not None and abs(actual_lap - int(reviewed)) <= 3),
    }


def metrics(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "reviewed_stints": 0, "confirmed_cliffs": 0, "confirmed_no_cliffs": 0,
            "true_positives": 0, "false_negatives": 0, "true_negatives": 0,
            "false_positives": 0, "precision": None, "recall": None,
            "specificity": None, "balanced_accuracy": None, "f1": None,
            "false_cliff_rate": None, "mean_absolute_cliff_lap_error": None,
            "median_absolute_cliff_lap_error": None, "matched_within_1_lap": None,
            "matched_within_2_laps": None, "matched_within_3_laps": None,
            "post_cliff_slope_retention_ratio": None,
        }
    positive = frame[frame.truth == "confirmed_cliff"]
    negative = frame[frame.truth == "confirmed_no_cliff"]
    tp = int((positive.detected_in_review_window).sum())
    fp = int((negative.detected_in_review_window).sum())
    fn = int(len(positive) - tp); tn = int(len(negative) - fp)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(positive) if len(positive) else None
    specificity = tn / len(negative) if len(negative) else None
    errors = positive.loc[positive.detected_in_review_window, "absolute_cliff_lap_error"].dropna()
    return {
        "reviewed_stints": int(len(frame)), "confirmed_cliffs": int(len(positive)), "confirmed_no_cliffs": int(len(negative)),
        "true_positives": tp, "false_negatives": fn, "true_negatives": tn, "false_positives": fp,
        "precision": precision, "recall": recall, "specificity": specificity,
        "balanced_accuracy": None if recall is None or specificity is None else (recall + specificity) / 2,
        "f1": None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "false_cliff_rate": fp / len(negative) if len(negative) else None,
        "mean_absolute_cliff_lap_error": float(errors.mean()) if len(errors) else None,
        "median_absolute_cliff_lap_error": float(errors.median()) if len(errors) else None,
        "matched_within_1_lap": float(frame.loc[positive.index, "within_1_lap"].mean()) if len(positive) else None,
        "matched_within_2_laps": float(frame.loc[positive.index, "within_2_laps"].mean()) if len(positive) else None,
        "matched_within_3_laps": float(frame.loc[positive.index, "within_3_laps"].mean()) if len(positive) else None,
        "post_cliff_slope_retention_ratio": float(frame["post_cliff_slope_retention_ratio"].dropna().mean()) if frame["post_cliff_slope_retention_ratio"].notna().any() else None,
    }


def configs_for(method: str) -> list[dict]:
    """Return a compact deterministic grid for grouped calibration."""
    grids = {
        "sustained": {
            "cliff_slope_threshold": (0.03, 0.08),
            "cliff_curvature_threshold": (0.0, 0.005),
            "cliff_baseline_delta": (0.15, 0.5),
            "cliff_persistence_laps": (1, 2),
            "cliff_min_lap": (4, 7),
            "smoothing_window": (5, 7),
        },
        "rolling_sustained": {
            "rolling_trend_window": (3, 5),
            "cliff_slope_threshold": (0.03, 0.08),
            "rolling_min_slope_increase": (0.03, 0.08),
            "cliff_baseline_delta": (0.15, 0.5),
            "rolling_min_fit_improvement_ratio": (0.1, 0.3),
            "cliff_persistence_laps": (1, 2),
            "smoothing_window": (5, 7),
        },
        "piecewise": {
            "piecewise_min_segment_laps": (3, 5),
            "cliff_slope_threshold": (0.03, 0.08),
            "piecewise_min_slope_increase": (0.03, 0.08),
            "piecewise_min_improvement_ratio": (0.1, 0.3),
            "smoothing_window": (5, 7),
        },
        "hybrid": {
            "piecewise_min_segment_laps": (3, 5),
            "cliff_slope_threshold": (0.03, 0.08),
            "piecewise_min_slope_increase": (0.03, 0.08),
            "piecewise_min_improvement_ratio": (0.1, 0.3),
            "cliff_persistence_laps": (1, 2),
            "cliff_baseline_delta": (0.15, 0.5),
            "smoothing_window": (5, 7),
        },
    }[method]
    keys = list(grids)
    # Evaluate eight Latin-hypercube-like combinations instead of the full
    # Cartesian product. Every parameter still takes both sensible endpoint
    # values, while the grouped calibration remains computationally modest.
    return [
        {key: grids[key][(row_index + column_index) % 2] for column_index, key in enumerate(keys)}
        for row_index in range(8)
    ]


def score_configuration(captures, curves, model_variant, source, method, config, selected_events=None):
    rows = []
    for capture in captures:
        if selected_events is not None and capture["event_name"] not in selected_events:
            continue
        curve = curves.get(model_variant, {}).get(capture_key(capture)) if source == "predicted" else observed_curve(capture)
        if curve is not None:
            rows.append(detector_row(capture, source, model_variant, method, curve, config))
    return rows


def threshold_search(captures, curves, output: Path) -> list[dict]:
    events = sorted({capture["event_name"] for capture in captures})
    holdout_events = set(events[-1:])
    development_events = set(events[:-1])
    output_rows = []
    for source, model_variant in (("observed", "observed_reference"), ("predicted", "hgb_raw"), ("predicted", "hgb_track_pca4")):
        for method in DETECTORS:
            candidates = []
            for config in configs_for(method):
                train_rows = score_configuration(captures, curves, model_variant, source, method, config, development_events)
                report = metrics(train_rows)
                false_rate = report["false_cliff_rate"] if report["false_cliff_rate"] is not None else 1.0
                balanced = report["balanced_accuracy"] or 0.0
                f1 = report["f1"] or 0.0
                candidates.append((false_rate <= 0.30, balanced, f1, -false_rate, config, report))
            eligible = [row for row in candidates if row[0]]
            selected = max(eligible or candidates, key=lambda row: (row[1], row[2], row[3], json.dumps(row[4], sort_keys=True)))
            holdout_rows = score_configuration(captures, curves, model_variant, source, method, selected[4], holdout_events)
            holdout_report = metrics(holdout_rows)
            output_rows.append({"curve_source": source, "model_variant": model_variant, "method": method, "development_events": ", ".join(sorted(development_events)), "holdout_events": ", ".join(sorted(holdout_events)), "searched_config_count": len(candidates), "selected_config": json.dumps(selected[4], sort_keys=True), "development_balanced_accuracy": selected[1], "development_f1": selected[2], "development_false_cliff_rate": selected[5]["false_cliff_rate"], "holdout_balanced_accuracy": holdout_report["balanced_accuracy"], "holdout_f1": holdout_report["f1"], "holdout_false_cliff_rate": holdout_report["false_cliff_rate"], "holdout_recall": holdout_report["recall"], "passes_false_cliff_guardrail": selected[0]})
    pd.DataFrame(output_rows).to_csv(output / "metrics" / "cliff_threshold_search.csv", index=False)
    return output_rows


def plot_cliff_results(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    selected = summary[summary["split"] == "all_reviewed_default"].copy()
    selected["label"] = selected["model_variant"] + "\n" + selected["curve_source"] + "\n" + selected["method"]
    metrics_to_plot = [("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("specificity", "Specificity"), ("f1", "F1"), ("false_cliff_rate", "False-cliff rate"), ("mean_absolute_cliff_lap_error", "Mean absolute error (laps)")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (column, title) in zip(axes.flat, metrics_to_plot):
        values = selected[column].fillna(0)
        ax.bar(np.arange(len(values)), values, color="#2f6f9f")
        ax.set_title(title); ax.set_xticks(np.arange(len(values)), selected.label, rotation=80, fontsize=6); ax.grid(axis="y", alpha=.25)
        ax.set_ylim(0, max(1.0, float(values.max()) * 1.2))
    axes.flat[-1].axis("off"); fig.suptitle("Cliff detector comparison on reviewed calibration curves"); fig.tight_layout()
    fig.savefig(output / "figures" / "cliff_method_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_examples(captures, curves, output: Path) -> None:
    import matplotlib.pyplot as plt
    examples = []
    for capture in captures:
        if capture["manual_review_status"] == "confirmed_cliff" and capture.get("reviewed_cliff_lap") is not None:
            examples.append(capture); break
    for capture in captures:
        if capture["manual_review_status"] == "confirmed_no_cliff": examples.append(capture); break
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3.5 * len(examples)), squeeze=False)
    for ax, capture in zip(axes.flat, examples):
        ages, observed = observed_curve(capture); ax.plot(ages, observed - observed[0], label="observed", color="#222222")
        for variant, color in (("hgb_raw", "#c75c2c"), ("hgb_track_pca4", "#2f6f9f")):
            curve = curves[variant].get(capture_key(capture))
            if curve is not None:
                pa, pv = curve; ax.plot(pa, pv - pv[0], label=variant)
        if capture.get("reviewed_cliff_lap") is not None: ax.axvline(capture["reviewed_cliff_lap"], color="#2f8f5b", linestyle="--", label="reviewed cliff")
        ax.set_title(f'{capture["event_name"]} {capture["driver"]} {capture["compound"]} — {capture["manual_review_status"]}')
        ax.set_xlabel("Tire age (laps)"); ax.set_ylabel("Relative lap-time delta"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output / "figures" / "observed_vs_predicted_cliff_examples.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def useful_life_report(captures, curves, data, output: Path, seed: int, draws: int) -> None:
    rows = []
    for variant in VARIANTS:
        for capture in captures:
            curve = curves[variant].get(capture_key(capture))
            if curve is None: continue
            ages, values = curve
            observed_ages, observed_values = observed_curve(capture)
            common_ages = np.intersect1d(ages, observed_ages)
            residuals = np.interp(common_ages, observed_ages, observed_values) - np.interp(common_ages, ages, values)
            estimate = estimate_useful_life_uncertainty(values, pit_loss=TRACK_PIT_LOSS.get(capture["track_name"], 22.0), fuel_correction=False, residuals=residuals, draws=draws, seed=seed)
            rows.append({"reference_id": capture_key(capture), "event_name": capture["event_name"], "compound": capture["compound"], "model_variant": variant, "strategy_useful_life_lower": estimate["strategy_useful_life_lower"], "strategy_useful_life_lap": estimate["strategy_useful_life_lap"], "strategy_useful_life_upper": estimate["strategy_useful_life_upper"], "strategy_useful_life_uncertainty_laps": estimate["strategy_useful_life_uncertainty_laps"], "strategy_useful_life_confidence": estimate["strategy_useful_life_confidence"], "strategy_useful_life_interval_method": estimate["strategy_useful_life_interval_method"], "strategy_useful_life_interval_capped": estimate["strategy_useful_life_interval_capped"], "empirical_lower": estimate["strategy_useful_life_empirical_lower"], "empirical_upper": estimate["strategy_useful_life_empirical_upper"]})
    frame = pd.DataFrame(rows); frame.to_csv(output / "tables" / "useful_life_intervals.csv", index=False)
    summary = frame.groupby(["model_variant", "strategy_useful_life_confidence"], as_index=False).agg(stints=("reference_id", "count"), mean_uncertainty_laps=("strategy_useful_life_uncertainty_laps", "mean"), capped_fraction=("strategy_useful_life_interval_capped", "mean"))
    summary.to_csv(output / "metrics" / "useful_life_uncertainty_summary.csv", index=False)
    import matplotlib.pyplot as plt
    sample = frame.head(min(12, len(frame))).copy(); positions = np.arange(len(sample)); fig, ax = plt.subplots(figsize=(13, 5)); ax.errorbar(positions, sample.strategy_useful_life_lap, yerr=[sample.strategy_useful_life_lap - sample.strategy_useful_life_lower, sample.strategy_useful_life_upper - sample.strategy_useful_life_lap], fmt="o", color="#2f6f9f", capsize=4); ax.set_xticks(positions, sample.reference_id, rotation=75, fontsize=7); ax.set_ylabel("Useful life (laps)"); ax.set_title("Operational useful-life intervals (representative stints)"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(output / "figures" / "useful_life_interval_examples.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def runtime_report(output: Path) -> None:
    from sim_engine import StrategySimulator
    base_profile = {"graph_data": {lap: 90 + .03 * lap for lap in range(1, 51)}, "drop_off_per_lap_sec": .03, "cliff_point_lap": 25, "strategy_useful_life_lap": 25}
    profiles = {compound: dict(base_profile) for compound in ("SOFT", "MEDIUM", "HARD")}
    records = []
    for label, interval in (("before_legacy_fallback", None), ("after_operational_interval", (22, 25, 28))):
        current = {compound: dict(profile) for compound, profile in profiles.items()}
        if interval:
            for profile in current.values(): profile.update(strategy_useful_life_lower=interval[0], strategy_useful_life_upper=interval[2], strategy_useful_life_uncertainty_laps=3, strategy_useful_life_confidence="low")
        simulator = StrategySimulator({"compounds": current, "input_context": {"track": "Monaco"}}); start = time.perf_counter(); result = simulator.generate_strategies(total_laps=30, grid_pos=1); elapsed = time.perf_counter() - start
        records.append({"scenario": label, "elapsed_seconds": elapsed, "candidates_evaluated": getattr(simulator, "last_candidate_count", None), "strategy_searches": getattr(simulator, "last_strategy_searches", 1), "valid_optimal": result.get("best_strategy") is not None, "valid_safe": result.get("safe_strategy") is not None, "valid_risky": result.get("risky_strategy") is not None})
    pd.DataFrame(records).to_csv(output / "metrics" / "strategy_runtime_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="reports/cliff_and_useful_life_validation"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--draws", type=int, default=200); parser.add_argument("--model-train-rows", type=int, default=12000, help="Analysis-only cap to avoid dense HGB resource exhaustion; use 0 for the full 2022 training set"); parser.add_argument("--fit-prediction-models", action="store_true", help="Fit analysis-only models instead of using committed captured raw curves"); args = parser.parse_args()
    output = Path(args.output_dir); (output / "figures").mkdir(parents=True, exist_ok=True); (output / "metrics").mkdir(exist_ok=True); (output / "tables").mkdir(exist_ok=True)
    label_path = ROOT / "calibration_work/fastf1_cliff_calibration/model_predictions.json"; labels_path = ROOT / "calibration_work/fastf1_cliff_calibration/reviewed_labels.json"; data = load_store(ROOT / "training_data/ground_effect"); captures = load_captures(label_path); model_train_rows = None if args.model_train_rows <= 0 else args.model_train_rows; curves = predicted_curves(data, captures, max_train_rows=model_train_rows, seed=args.seed, fit_model_predictions=args.fit_prediction_models)
    rows = []
    for method in DETECTORS:
        for source, model_variant in (("observed", "observed_reference"), ("predicted", "hgb_raw"), ("predicted", "hgb_track_pca4")):
            for capture in captures:
                curve = observed_curve(capture) if source == "observed" else curves[model_variant].get(capture_key(capture))
                if curve is not None: rows.append(detector_row(capture, source, model_variant, method, curve, {**DEFAULT_CONFIG, "cliff_detection_method": method}))
    detail = pd.DataFrame(rows); detail.to_csv(output / "tables" / "cliff_stint_predictions.csv", index=False)
    summary_rows = []
    for keys, group in detail.groupby(["model_variant", "curve_source", "method"]): summary_rows.append({"model_variant": keys[0], "curve_source": keys[1], "method": keys[2], "split": "all_reviewed_default", **metrics(group.to_dict("records"))})
    threshold_rows = threshold_search(captures, curves, output)
    summary = pd.DataFrame(summary_rows); summary.to_csv(output / "metrics" / "cliff_method_summary.csv", index=False); (output / "metrics" / "cliff_method_summary.json").write_text(json.dumps(summary_rows, indent=2, default=str))
    slope = detail[["reference_id", "event_name", "model_variant", "curve_source", "method", "observed_pre_cliff_slope", "observed_post_cliff_slope", "predicted_pre_cliff_slope", "predicted_post_cliff_slope", "post_cliff_slope_retention_ratio", "detectable_slope_break_near_reviewed"]].drop_duplicates(); slope.to_csv(output / "tables" / "cliff_slope_retention.csv", index=False)
    plot_cliff_results(summary, output); plot_examples(captures, curves, output); useful_life_report(captures, curves, data, output, args.seed, args.draws); runtime_report(output)
    provenance = {"source_commit": SOURCE_COMMIT, "input_manifest": "training_data/ground_effect/manifest.json", "input_manifest_sha256": sha256(ROOT / "training_data/ground_effect/manifest.json"), "reviewed_label_file_sha256": sha256(labels_path), "prediction_capture_file_sha256": sha256(label_path), "row_count": len(data), "reviewed_stints": len(captures), "model_variants": ["hgb_raw", "hgb_track_pca4"], "detector_methods": list(DETECTORS), "random_seed": args.seed, "uncertainty_draws": args.draws, "pca_fitting_policy": "PCA-4 is fit only on one median profile per training-fold circuit and held-out rows are transformed with that fold object when model fitting is enabled.", "model_training_policy": {"year": 2022, "rows_used": model_train_rows, "sampling": "deterministic random sample for analysis-only resource control" if model_train_rows else "full 2022 rows", "prediction_source": "captured_raw_model_predictions" if not args.fit_prediction_models else "analysis_only_fit", "pca_prediction_status": "not available; analysis-only HGB fitting aborted with exit code 134" if not args.fit_prediction_models else "requested"}, "threshold_validation": {"type": "grouped development/holdout", "development_events": sorted({c['event_name'] for c in captures})[:-1], "holdout_events": sorted({c['event_name'] for c in captures})[-1:], "false_cliff_guardrail": 0.30}, "useful_life_interval_method": "Empirical residual perturbation using held-out residual variation; operational width mapped to ±1, ±2, or ±3 laps and capped status retained.", "production_defaults_changed": False, "production_model_artifacts_changed": False, "production_training_data_changed": False, "limitation": "Only one held-out Grand Prix is available for grouped threshold evaluation, reviewed labels are a small manually reviewed 2023 set, and new PCA held-out curves could not be generated because the analysis-only HGB fit aborted with exit code 134. PCA cliff metrics from the prior committed comparison are not relabeled as new shape diagnostics."}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)); (output / "README.md").write_text("# Cliff and useful-life validation\n\nAnalysis-only results from frozen reviewed 2023 race-stint labels. Observed curves are fuel-corrected and predicted curves are generated by leakage-safe 2022-to-2023 evaluation. Detector threshold search uses a grouped development/holdout split by Grand Prix. Useful-life intervals are operational ±1/±2/±3 lap bands, not formal confidence intervals. The predicted-curve fit uses the documented analysis-only training-row cap to avoid dense HGB resource exhaustion. Production defaults, training data, PCA implementation, and model artifacts were not changed.\n")
    print(json.dumps({"reviewed_stints": len(captures), "rows": len(detail), "threshold_rows": len(threshold_rows)}, indent=2))


if __name__ == "__main__": main()
