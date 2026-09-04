# Feature-engineering analysis

This package evaluates the current leakage-safe Ground Effect representation without changing production artifacts. It produces correlation tables, descriptive circuit PCA, chronological model comparisons, ablation-ready feature groups, and held-out permutation importance.

## Reproduce

From the repository root:

```bash
python analysis/run_feature_analysis.py --era ground_effect --output-dir reports/feature_engineering --random-state 42
```

For a fast fixture-like run:

```bash
python analysis/run_feature_analysis.py --era ground_effect --output-dir reports/feature_engineering_smoke --random-state 42 --smoke
```

After an exploratory PCA sweep, the focused full-data comparison can be run with:

```bash
python analysis/run_feature_analysis.py --era ground_effect --output-dir reports/feature_engineering_focused --random-state 42 --focused-pca
```

This second stage compares the raw baseline with PCA-1, PCA-2, and PCA-4. It generates the same correlation, PCA, model-metric, importance, compound, tire-age, and degradation-curve visuals while avoiding the cost of refitting all seven PCA candidates.

The full run uses the committed Ground Effect processed store (2022–2025). Runtime depends on hardware because every serious candidate is refit on chronological event folds. The smoke run uses the final four development events. No external API calls are made.

## Final Ground Effect selection package

The final selection entry point evaluates the four circuit stages and the
requested feature ablations. It uses expanding chronological folds through
2024, locks selection on development data, and reports 2025 separately:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.analysis-mpl-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python analysis/run_final_model_selection.py \
  --output-dir reports/final_model_selection --seed 42
```

For an infrastructure check, use `--smoke`; for a controlled pilot use
`--max-events` and `--rows-per-event`. These options must not be described as
full-data evidence. `--skip-artifact` avoids the final full-data bundle fit
when only validating the experiment runner.

The submission bundle is analysis-only and is written under the report
directory. It does not overwrite `models/`, change Active Aero, or alter
dashboard inference. The persisted PCA transformer accepts canonical raw rows
and stores the scaler, loadings, signs, and explained-variance values.

## Methodology

The development folds use expanding chronological training windows through 2022–2024. A frozen 2025 evaluation is reported separately and is not used to choose features. Every variant uses identical rows and folds. Track PCA is fit on one median profile per circuit from the training fold, then applied to lap rows. Imputation, scaling, categorical encoding, and PCA therefore never see the held-out event. `LapTimeDelta`, metadata, identifiers, and legacy leakage fields are rejected from model matrices.

The six initial variants are `hgb_raw`, `hgb_pruned`, `hgb_track_pca4`, `hgb_track_pca5`, `ridge_raw`, and `ridge_pca`. The runner preserves `hgb_raw` as the provisional selection until downstream degradation, cliff, and strategy benchmarks are available; it does not overwrite production models.

## Outputs

`reports/feature_engineering/` contains PNG/SVG figures, CSV tables, metrics, and `provenance.json`. The provenance file records the input manifest hash, seed, software version, fold scope, and provisional decision. Generated report files can be removed safely; processed stores and production models are outside the output directory.
