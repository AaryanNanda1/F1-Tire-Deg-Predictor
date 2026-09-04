#!/usr/bin/env python3
"""Generate publication visuals from committed full-run outputs only.

This script never fits a model and never reads values from an image.  Missing
downstream evidence is represented in the README/provenance instead of being
replaced with pilot results.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/paper_visuals"
FINAL = ROOT / "reports/final_model_selection"
DOWNSTREAM = ROOT / "reports/feature_engineering_downstream"
TRACK = ["traction", "tyre_stress", "asphalt_grip", "corner_speed_energy", "abrasiveness", "braking_severity", "lateral_load"]
SHORT = {"traction": "Traction", "tyre_stress": "Tyre stress", "asphalt_grip": "Asphalt grip", "corner_speed_energy": "High-speed corner index", "abrasiveness": "Asphalt abrasion", "braking_severity": "Braking", "lateral_load": "Lateral load"}
ORDER = ["no_circuit", "pca4_no_age_interactions", "raw_circuit_7", "raw_circuit_age_interactions", "pca4_pruned"]
COLORS = {"no_circuit": "#0072B2", "pca4_no_age_interactions": "#E69F00", "raw_circuit_7": "#009E73", "raw_circuit_age_interactions": "#CC79A7", "pca4_pruned": "#D55E00"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def md_table(df: pd.DataFrame, path: Path) -> None:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    path.write_text("\n".join(lines) + "\n")


def save(fig, name: str):
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "figures" / f"{name}.svg", bbox_inches="tight"); plt.close(fig)


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 5)); ax.axis("off")
    blocks = [("Training inputs", "FastF1\nRace / Sprint / FP2"), ("Simulation inputs", "Open-Meteo\nweather + race state"), ("Context", "Pirelli + Mercedes\ncircuit descriptors"), ("Preprocessing", "filtering + leakage\ncontrols"), ("Learned model", "era-specific HGB\nregression"), ("Post-processing", "degradation curves\ncliff + useful life"), ("Strategy", "constrained search\npit recommendations")]
    xs = np.linspace(.07, .93, len(blocks))
    for i, ((title, text), x) in enumerate(zip(blocks, xs)):
        ax.text(x, .56, title, ha="center", va="center", fontsize=10, weight="bold", bbox={"boxstyle": "round,pad=.55", "facecolor": "#EAF2F8" if i < 4 else "#FCEFE3", "edgecolor": "#4C566A"})
        ax.text(x, .37, text, ha="center", va="center", fontsize=8)
        if i < len(blocks)-1: ax.annotate("", xy=(xs[i+1]-.065, .56), xytext=(x+.065, .56), arrowprops={"arrowstyle": "->", "color": "#4C566A"})
    ax.set_title("Reproducible Formula 1 tire-degradation and strategy pipeline", pad=15); save(fig, "figure_01_research_pipeline")


def coverage_figure():
    manifest = json.loads((ROOT / "training_data/ground_effect/manifest.json").read_text())
    rows = []
    for item in manifest["sessions"].values(): rows.append({"season": item["year"], "session": item["session_code"], "rows": item["rows"]})
    df = pd.DataFrame(rows); counts = df.groupby(["season", "session"], as_index=False).rows.sum(); pivot = counts.pivot(index="season", columns="session", values="rows").fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4)); pivot.plot.bar(ax=ax, color={"R": "#0072B2", "S": "#009E73", "FP2": "#E69F00"}); ax.set(xlabel="Season", ylabel="Retained processed laps", title="Ground Effect data coverage by season and session"); ax.legend(title="Session", labels=["Race", "Sprint", "FP2"]); ax.grid(axis="y", alpha=.25); save(fig, "figure_02_dataset_coverage")
    audit = pd.DataFrame([{"season": item["year"], "session": item["session_code"], **{k: item.get("filter_audit", {}).get(k, 0) for k in ["input_rows", "pit_in_out_removed", "short_stint_removed", "robust_outlier_removed", "final_rows"]}} for item in manifest["sessions"].values()]); audit.to_csv(OUT / "source_data/filter_audit.csv", index=False)


def pca_figures():
    corr = pd.read_csv(FINAL / "tables/track_spearman.csv", index_col=0); variance = pd.read_csv(FINAL / "tables/pca_explained_variance.csv"); load = pd.read_csv(FINAL / "tables/pca_loadings.csv", index_col=0); corr.index = [SHORT.get(x, x) for x in corr.index]; corr.columns = [SHORT.get(x, x) for x in corr.columns]; load.index = [SHORT.get(x, x) for x in load.index]
    fig, ax = plt.subplots(figsize=(7, 6)); im=ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); ax.set_xticks(range(7), corr.columns, rotation=45, ha="right"); ax.set_yticks(range(7), corr.index); ax.set_title("Spearman correlation of circuit descriptors"); fig.colorbar(im, ax=ax, label="Correlation");
    for i in range(7):
        for j in range(7): ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    save(fig, "figure_03_circuit_correlation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); x=variance.component; axes[0].bar(x, variance.explained_variance_ratio, color="#0072B2"); axes[0].plot(x, variance.cumulative_variance, color="#D55E00", marker="o"); axes[0].axvline(4, ls="--", color="black"); axes[0].set(xlabel="Component", ylabel="Explained variance", title="PCA variance; PC1–PC4 marked"); axes[0].legend(["Cumulative", "Individual"], fontsize=8); im=axes[1].imshow(load.iloc[:, :4], cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"); axes[1].set_xticks(range(4), load.columns[:4]); axes[1].set_yticks(range(7), load.index); axes[1].set_title("PCA-4 loadings"); fig.colorbar(im, ax=axes[1], label="Loading");
    for i in range(7):
        for j in range(4): axes[1].text(j, i, f"{load.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    save(fig, "figure_04_pca_variance_loadings")
    corr.to_csv(OUT / "source_data/circuit_spearman.csv"); variance.to_csv(OUT / "source_data/pca_explained_variance.csv", index=False); load.to_csv(OUT / "source_data/pca_loadings.csv")


def model_figures():
    summary = pd.read_csv(FINAL / "metrics/model_comparison_summary.csv"); summary.to_csv(OUT / "source_data/model_comparison_summary.csv", index=False)
    stages = summary[(summary.variant.isin(ORDER)) & (summary.split.isin(["development", "frozen_2025", "conditional_development", "conditional_frozen_2025"]))].copy(); stages["label"] = stages.variant.map({"no_circuit": "No circuit", "pca4_no_age_interactions": "PCA-4", "raw_circuit_7": "Raw 7", "raw_circuit_age_interactions": "Raw + age", "pca4_pruned": "PCA-4 pruned"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, split, title in [(axes[0], "development", "Development"), (axes[1], "frozen_2025", "Frozen 2025")]:
        part=stages[stages.split.eq(split)]; part=part.drop_duplicates("variant"); x=np.arange(len(part)); w=.36; ax.bar(x-w/2, part.mae, w, label="MAE", color="#0072B2"); ax.bar(x+w/2, part.rmse, w, label="RMSE", color="#E69F00"); ax.set_xticks(x, [f"{v}\n({int(f)})" for v,f in zip(part.label,part.feature_count)]); ax.set_title(title); ax.set_ylabel("Seconds per lap"); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
    save(fig, "figure_05_representation_comparison")
    paired = pd.read_csv(FINAL / "tables/paired_event_mae_differences.csv"); paired.to_csv(OUT / "source_data/paired_event_mae_differences.csv", index=False); ab=paired[paired.candidate.str.startswith("no_") & ~paired.candidate.isin(["no_circuit"])].copy(); fig, ax=plt.subplots(figsize=(7,4.5)); y=np.arange(len(ab)); ax.errorbar(ab.mean_difference, y, xerr=[ab.mean_difference-ab.bootstrap_95ci_lower, ab.bootstrap_95ci_upper-ab.mean_difference], fmt="o", color="#0072B2"); ax.axvline(0,color="black",lw=.8); ax.set_yticks(y, ab.candidate); ax.set_xlabel("ΔMAE: ablated − full PCA-4 (seconds/lap)"); ax.set_title("Feature ablations; PCA-4 is the reference"); ax.grid(axis="x",alpha=.25); save(fig,"figure_06_feature_ablation")
    ev=pd.read_csv(FINAL / "metrics/per_event_predictions.csv"); ev=ev[ev.variant.isin(ORDER)]; ev["event_mae"]=ev.groupby(["variant","fold"]).absolute_error.transform("mean"); fig,ax=plt.subplots(figsize=(8,4.5)); parts=[ev[(ev.variant==v)&(ev.split=="development")].groupby("fold").event_mae.first().to_numpy() for v in ORDER if v in set(ev.variant)]; labels=[v for v in ORDER if v in set(ev.variant)]; ax.boxplot(parts, labels=labels, showfliers=False); ax.set_ylabel("Event MAE (seconds/lap)"); ax.set_title("Development event-level error distribution"); ax.grid(axis="y",alpha=.25); save(fig,"figure_07_event_error_distribution")
    stages.to_csv(OUT / "source_data/representation_comparison.csv", index=False); md_table(stages, OUT / "tables/representation_comparison.md"); paired.to_csv(OUT / "tables/feature_ablation.csv", index=False); md_table(paired, OUT / "tables/feature_ablation.md")


def copy_downstream():
    mapping={"18_predicted_vs_observed_curves":"figure_08_predicted_vs_observed_curves", "20_cliff_accuracy_comparison":"figure_09_cliff_comparison", "useful_life_interval_examples":"figure_10_useful_life_examples", "19_strategy_comparison":"figure_11_strategy_comparison"}
    for source, target in mapping.items():
        candidates=list((DOWNSTREAM / "figures").glob(f"{source}.png"))+list((DOWNSTREAM / "cliff_accuracy/figures").glob(f"{source}.png"))+list((ROOT / "reports/cliff_and_useful_life_validation/figures").glob(f"{source}.png"))
        if candidates: shutil.copy2(candidates[0], OUT / "figures" / f"{target}.png")
    for source, target in [(DOWNSTREAM / "tables/heldout_degradation_curves.csv", "heldout_degradation_curves.csv"), (DOWNSTREAM / "metrics/strategy_comparison.csv", "strategy_comparison.csv"), (DOWNSTREAM / "cliff_accuracy/tables/stint_predictions.csv", "cliff_stint_predictions.csv")]:
        if source.exists(): shutil.copy2(source, OUT / "source_data" / target)


def main():
    for name in ("figures", "tables", "captions", "source_data"): (OUT/name).mkdir(parents=True, exist_ok=True)
    pipeline_figure(); coverage_figure(); pca_figures(); model_figures(); copy_downstream()
    captions = {
        1: "End-to-end research pipeline. FastF1 provides historical laps; Open-Meteo supplies simulation-time weather context. The learned era model is separated from deterministic tire-life and strategy post-processing.",
        2: "Retained Ground Effect processed laps by season and session type, using session roles recorded in the training-data manifest.",
        3: "Spearman correlation of seven circuit descriptors using equal-weight circuit profiles. Pearson correlations are supplementary.",
        4: "Descriptive circuit PCA variance and PCA-4 loadings. Predictive PCA is fitted inside chronological training folds; component signs are arbitrary, and PCA-4 was evaluated but not selected.",
        5: "Ground Effect circuit-representation comparison. Development results determine selection; frozen 2025 is reported separately and was not used to select the candidate.",
        6: "Paired event-level feature ablations relative to full PCA-4. Positive ΔMAE means removal increased error; these are predictive ablations, not causal effects.",
        7: "Development event-level MAE distributions showing fold-to-fold variation rather than only pooled lap error.",
        8: "Held-out degradation curves from the committed downstream candidate comparison; this is not evidence that the selected no-circuit model has been downstream-validated.",
        9: "Earlier committed cliff comparison. Stints without an observable cliff are treated as no-cliff labels; thresholds were not retuned in this visual package.",
        10: "Operational useful-life interval examples. The displayed range is a sensitivity range, not a calibrated statistical confidence interval.",
        11: "Earlier candidate strategy comparison using common scenarios. It does not establish downstream performance for the selected no-circuit model unless that candidate is present in the source table.",
    }
    for number, caption in captions.items(): (OUT / "captions" / f"figure_{number:02d}.txt").write_text(caption)
    manifest={"source_commit": json.loads((FINAL/"provenance.json").read_text())["source_commit"], "authoritative_workflow_run":"33846305685", "seed":42, "figures":[], "limitations":["Figures 8–11 reuse earlier committed downstream candidate outputs where available; they do not establish performance of selected no_circuit unless explicitly included.","The selected final artifact is not described as live production."]}
    for path in sorted((OUT/"figures").glob("*.png")):
        manifest["figures"].append({"filename":path.name,"svg_filename":path.with_suffix('.svg').name if path.with_suffix('.svg').exists() else None,"sha256":sha(path),"source":"committed full-run tables or committed downstream report"})
    (OUT/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)); (OUT/"provenance.json").write_text(json.dumps({"source_commit":manifest["source_commit"],"workflow_run":33846305685,"input_files":{str(p.relative_to(ROOT)):sha(p) for p in [FINAL/"metrics/model_comparison_summary.csv",FINAL/"tables/track_spearman.csv",FINAL/"tables/pca_explained_variance.csv",ROOT/"training_data/ground_effect/manifest.json"] if p.exists()},"seed":42,"selection":"no_circuit","production_changed":False},indent=2))
    (OUT/"README.md").write_text("# Paper visual package\n\nGenerated by `analysis/generate_report_visuals.py` from the successful full Ground Effect Action artifact and committed downstream reports. No model fitting occurs. The selected Ground Effect representation is `no_circuit`; PCA-4 remains an evaluated alternative. Figures 8–11 are labelled as earlier-candidate/downstream diagnostics when the selected model was not included.\n\nReproduce:\n\n```bash\nMPLBACKEND=Agg python analysis/generate_report_visuals.py\n```\n")
    print(json.dumps({"output":str(OUT),"figures":len(manifest["figures"])},indent=2))


if __name__ == "__main__": main()
