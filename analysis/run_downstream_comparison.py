#!/usr/bin/env python3
"""Compare raw and PCA-4 outputs on held-out 2025 curves and strategy logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from analysis.feature_experiments import fit_model, prepare_variant
from analysis.plotting import bar_plot
from mappings import EVENT_NAME_TO_CIRCUIT, get_track_info
from sim_engine import StrategySimulator
from strategy_optimizer import optimize_strategy
from tire_life_analysis import analyze_tire_life


def load_store(path: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p, compression="gzip") for p in sorted(path.rglob("*.csv.gz"))], ignore_index=True)


def slope(values: pd.Series, ages: pd.Series) -> float:
    if ages.nunique() < 3: return float("nan")
    return float(np.polyfit(ages.to_numpy(), values.to_numpy(), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/feature_engineering_downstream")
    args = parser.parse_args()
    output = Path(args.output_dir); (output / "figures").mkdir(parents=True, exist_ok=True); (output / "metrics").mkdir(exist_ok=True); (output / "tables").mkdir(exist_ok=True)
    data = load_store(ROOT / "training_data/ground_effect")
    data["EventDate"] = pd.to_datetime(data["EventDate"])
    train = data[data.EventDate.dt.year <= 2024].copy(); test = data[data.EventDate.dt.year == 2025].copy()
    predictions = {}
    for variant in ("hgb_raw", "hgb_track_pca4"):
        train_x, test_x, _ = prepare_variant(train, test, variant)
        model = fit_model(train_x, train["LapTimeDelta"], variant, train.get("SampleWeight"))
        predictions[variant] = model.predict(test_x)

    curve_rows = []; life_rows = []; strategy_rows = []
    for variant, predicted in predictions.items():
        scored = test[["EventDate", "EventName", "Compound", "TyreLife", "LapTimeDelta"]].copy(); scored["prediction"] = predicted
        for (event_date, event_name, compound), group in scored.groupby(["EventDate", "EventName", "Compound"]):
            group = group.groupby("TyreLife", as_index=False)[["LapTimeDelta", "prediction"]].mean().sort_values("TyreLife")
            for _, row in group.iterrows(): curve_rows.append({"variant": variant, "event_date": event_date.date(), "event_name": event_name, "compound": compound, "tyre_life": int(row.TyreLife), "observed": row.LapTimeDelta, "predicted": row.prediction})
            if len(group) >= 10:
                observed_life = analyze_tire_life(group.LapTimeDelta.to_numpy(), fuel_correction=False)
                predicted_life = analyze_tire_life(group.prediction.to_numpy(), fuel_correction=False)
                life_rows.append({"variant": variant, "event_date": event_date.date(), "event_name": event_name, "compound": compound, "curve_mae": float(np.mean(np.abs(group.LapTimeDelta - group.prediction))), "observed_cliff": observed_life["performance_cliff_lap"], "predicted_cliff": predicted_life["performance_cliff_lap"], "cliff_abs_error": abs(predicted_life["performance_cliff_lap"] - observed_life["performance_cliff_lap"]) if observed_life["performance_cliff_lap"] is not None and predicted_life["performance_cliff_lap"] is not None else np.nan, "observed_useful_life": observed_life["strategy_useful_life_lap"], "predicted_useful_life": predicted_life["strategy_useful_life_lap"], "useful_life_abs_error": abs(predicted_life["strategy_useful_life_lap"] - observed_life["strategy_useful_life_lap"])})
        # Use the held-out predicted curves as inputs to the existing strategy optimizer.
        for (event_date, event_name), event_group in scored.groupby(["EventDate", "EventName"]):
            track_name = EVENT_NAME_TO_CIRCUIT.get(event_name, event_name); info = get_track_info(track_name); models = {}
            for compound, group in event_group.groupby("Compound"):
                curve = group.groupby("TyreLife")["prediction"].mean().sort_index()
                if len(curve) < 3 or compound not in {"SOFT", "MEDIUM", "HARD"}: continue
                models[compound] = {"fresh_lap_time_sec": float(curve.iloc[0] + 90.0), "slope_sec_per_km": max(0.001, slope(curve, curve.index) / max(info.get("length_km", 5.0), 0.1)), "window_laps": max(8, min(40, int(curve.index.max())))}
            result = optimize_strategy(models, int(info.get("race_laps", 57)), float(info.get("length_km", 5.0)), race_condition="dry", pit_loss_sec=22.0, top_k=1)
            if result: strategy_rows.append({"variant": variant, "event_date": event_date.date(), "event_name": event_name, "strategy": json.dumps(result[0], sort_keys=True), "predicted_total_time_sec": result[0]["predicted_total_time_sec"]})

    curves = pd.DataFrame(curve_rows); lives = pd.DataFrame(life_rows); strategies = pd.DataFrame(strategy_rows)
    curves.to_csv(output / "tables" / "heldout_degradation_curves.csv", index=False); lives.to_csv(output / "metrics" / "cliff_useful_life_comparison.csv", index=False); strategies.to_csv(output / "metrics" / "strategy_comparison.csv", index=False)
    summary = lives.groupby("variant", as_index=False).agg(curve_mae=("curve_mae", "mean"), cliff_abs_error=("cliff_abs_error", "mean"), useful_life_abs_error=("useful_life_abs_error", "mean"), evaluated_curves=("compound", "count")) if not lives.empty else pd.DataFrame()
    if not summary.empty: summary.to_csv(output / "metrics" / "downstream_summary.csv", index=False); bar_plot(summary, "variant", "curve_mae", "Held-out degradation-curve error", "Curve MAE (seconds)", output / "figures" / "17_curve_error_comparison.png")
    if not strategies.empty:
        strategy_summary = strategies.groupby("variant", as_index=False).agg(predicted_total_time_sec=("predicted_total_time_sec", "mean")); strategy_summary.to_csv(output / "metrics" / "strategy_summary.csv", index=False); bar_plot(strategy_summary, "variant", "predicted_total_time_sec", "Predicted strategy time by representation", "Predicted total time (seconds)", output / "figures" / "19_strategy_comparison.png")
    # Plot predicted and observed mean curves, with a separate panel per representation.
    import matplotlib.pyplot as plt
    if not curves.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for axis, variant in zip(axes, ("hgb_raw", "hgb_track_pca4")):
            subset = curves[curves.variant == variant]
            for label, group in subset.groupby("compound"):
                grouped = group.groupby("tyre_life", as_index=False)[["observed", "predicted"]].mean()
                axis.plot(grouped.tyre_life, grouped.observed, color="black", alpha=.7, label=f"{label} observed")
                axis.plot(grouped.tyre_life, grouped.predicted, linestyle="--", label=f"{label} predicted")
            axis.set_title(variant); axis.set_xlabel("Tire age (laps)"); axis.set_ylabel("Lap-time delta (seconds)"); axis.grid(alpha=.25); axis.legend(fontsize=7)
        fig.suptitle("Held-out 2025 observed versus predicted degradation curves"); fig.tight_layout(); fig.savefig(output / "figures" / "18_predicted_vs_observed_curves.png", dpi=300, bbox_inches="tight"); fig.savefig(output / "figures" / "18_predicted_vs_observed_curves.svg", bbox_inches="tight"); plt.close(fig)
    metadata = {"data_scope": "2025 held-out Ground Effect events", "training_scope": "2022-2024", "variants": ["hgb_raw", "hgb_track_pca4"], "strategy_interpretation": "diagnostic generated from held-out predicted curves; not observed-race strategy accuracy", "production_changed": False}
    (output / "metrics" / "evaluation_notes.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"test_rows": len(test), "curve_groups": len(curves), "life_groups": len(lives), "strategies": len(strategies), "summary": summary.to_dict("records") if not summary.empty else []}, indent=2, default=str))


if __name__ == "__main__": main()
