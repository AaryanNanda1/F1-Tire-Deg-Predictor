# Final Ground Effect model selection

Selected submission variant: **no_circuit**. The package compares four circuit representations, corrected PCA-4 feature ablations, conditional joint ablations, and a combined pruned candidate using chronological development folds through 2024 and a genuine per-event frozen 2025 evaluation.

PCA describes variance in circuit descriptors; predictive value is established only through chronological model comparison. Positive ablation ΔMAE means removal worsened MAE. Tire-age and compound are diagnostic positive controls and remain required for compound-specific degradation predictions. Cliff thresholds were not retuned and useful-life ranges are operational ranges, not calibrated confidence intervals.

See `provenance.json`, `selection_decision.json`, `metrics/model_comparison_summary.csv`, and `artifacts/submission_manifest.json` for authoritative generated values.
