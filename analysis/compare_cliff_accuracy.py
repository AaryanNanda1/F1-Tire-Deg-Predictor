#!/usr/bin/env python3
"""Evaluate raw versus PCA-4 cliff detection on frozen reviewed labels."""

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
from mappings import normalize_team_name
from tire_life_analysis import analyze_tire_life


def load_store(path):
    return pd.concat([pd.read_csv(p, compression="gzip") for p in sorted(path.rglob("*.csv.gz"))], ignore_index=True)


def score(rows):
    positive = [r for r in rows if r["truth"] == "confirmed_cliff"]
    negative = [r for r in rows if r["truth"] == "confirmed_no_cliff"]
    tp = sum(r["detected_in_window"] for r in positive); fn = len(positive) - tp
    fp = sum(r["detected_in_window"] for r in negative); tn = len(negative) - fp
    precision = tp / (tp + fp) if tp + fp else None; recall = tp / len(positive) if positive else None; specificity = tn / len(negative) if negative else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    errors = [abs(r["predicted_cliff"] - r["reviewed_cliff"]) for r in positive if r["detected_in_window"] and r["reviewed_cliff"] is not None and r["predicted_cliff"] is not None]
    return {"stints": len(rows), "confirmed_cliffs": len(positive), "confirmed_no_cliffs": len(negative), "true_positives": tp, "false_negatives": fn, "true_negatives": tn, "false_positives": fp, "precision": precision, "recall": recall, "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None, "f1": f1, "false_cliff_rate": fp / len(negative) if negative else None, "mean_absolute_cliff_lap_error": float(np.mean(errors)) if errors else None, "within_1_lap_rate": sum(e <= 1 for e in errors) / len(errors) if errors else None, "matched_cliffs": len(errors)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="reports/feature_engineering_downstream/cliff_accuracy"); args = parser.parse_args()
    out = Path(args.output_dir); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "metrics").mkdir(exist_ok=True); (out / "tables").mkdir(exist_ok=True)
    data = load_store(ROOT / "training_data/ground_effect"); data["EventDate"] = pd.to_datetime(data["EventDate"])
    train = data[data.EventDate.dt.year == 2022].copy(); scored = data[data.EventDate.dt.year == 2023].copy()
    labels = json.loads((ROOT / "calibration_work/fastf1_cliff_calibration/reviewed_labels.json").read_text())["labels"]
    labels = [label for label in labels if label["season"] == 2023 and label["manual_review_status"] in {"confirmed_cliff", "confirmed_no_cliff"}]
    output_rows = []
    for variant in ("hgb_raw", "hgb_track_pca4"):
        train_x, scored_x, _ = prepare_variant(train, scored, variant); model = fit_model(train_x, train["LapTimeDelta"], variant, train.get("SampleWeight")); scored = scored.copy(); scored["prediction"] = model.predict(scored_x)
        for label in labels:
            team = normalize_team_name(label["team"]); group = scored[(scored.EventName == label["event_name"]) & (scored.Driver == label["driver"]) & (scored.Team == team) & (scored.Compound == label["compound"]) & (scored.Stint == label["stint"])]
            curve = group.groupby("TyreLife")["prediction"].mean().sort_index()
            if len(curve) < 10: continue
            result = analyze_tire_life(curve.to_numpy(), fuel_correction=False); relative = result["performance_cliff_lap"]; predicted = int(relative + curve.index.min() - 1) if relative is not None else None
            detected = predicted is not None and int(label["starting_tyre_age"]) <= predicted <= int(label["ending_tyre_age"])
            output_rows.append({"variant": variant, "reference_id": label["reference_id"], "event_name": label["event_name"], "driver": label["driver"], "compound": label["compound"], "truth": label["manual_review_status"], "reviewed_cliff": label["reviewed_cliff_lap"], "predicted_cliff": predicted, "detected_in_window": detected, "classification": "true_positive" if label["manual_review_status"] == "confirmed_cliff" and detected else "false_negative" if label["manual_review_status"] == "confirmed_cliff" else "false_positive" if detected else "true_negative"})
    rows = pd.DataFrame(output_rows); rows.to_csv(out / "tables" / "stint_predictions.csv", index=False)
    reports = []
    for variant, group in rows.groupby("variant"):
        report = score(group.to_dict("records")); report["variant"] = variant; reports.append(report)
    summary = pd.DataFrame(reports); summary.to_csv(out / "metrics" / "cliff_accuracy_summary.csv", index=False); (out / "metrics" / "cliff_accuracy_summary.json").write_text(json.dumps(reports, indent=2, default=str))
    import matplotlib.pyplot as plt
    metrics = [("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("specificity", "Specificity"), ("f1", "F1 score"), ("false_cliff_rate", "False-cliff rate")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); positions = np.arange(len(summary)); labels_x = summary.variant.tolist()
    for axis, (column, title) in zip(axes.flat, metrics):
        axis.bar(positions, summary[column].fillna(0), color="#2f6f9f"); axis.set_title(title); axis.set_ylim(0, 1); axis.set_xticks(positions, labels_x, rotation=25); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Reviewed 2023 cliff-detection comparison"); fig.tight_layout(); fig.savefig(out / "figures" / "20_cliff_accuracy_comparison.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "figures" / "20_cliff_accuracy_comparison.svg", bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"evaluated_rows": len(rows), "reports": reports}, indent=2, default=str))


if __name__ == "__main__": main()
