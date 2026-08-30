# Tire Cliff and Race Strategy Implementation Plan

## Purpose

Validate and improve the tire performance-cliff calculation and race-strategy recommendations using cleaned FastF1 stint data and official Pirelli strategy evidence.

This plan replaces the former `planning.md` and `tire_cliff_strategy_plan.md` documents. It is the canonical implementation and rollout plan. The shorter `tire_cliff_strategy_plan_summary.md` remains available as a plain-language overview.

## Scope

The work will:

- Validate predicted tire degradation curves against historical race stints.
- Calibrate the performance-cliff detector using reviewed FastF1 evidence.
- Preserve the existing strategic useful-life calculation during the first calibration cycle.
- Separate mathematical race-time optimization from recommendation risk.
- Evaluate strategy quality against Pirelli guidance and actual race evidence.
- Introduce frontend changes only after the new behavior passes holdout evaluation.

The work will not:

- Treat Pirelli recommendations as ground truth or training labels.
- Treat the longest observed stint as a tire's exact physical limit.
- Tune parameters against holdout races.
- Retrain the degradation model merely to improve two pilot benchmark results.
- Change the live strategy recommendation before validation gates pass.

## Terminology

- **Predicted degradation curve:** The machine-learning model's predicted lap-time evolution as tire age increases.
- **Performance cliff:** `performance_cliff_lap`, the point where degradation begins accelerating materially.
- **Strategic useful life:** `strategy_useful_life_lap`, the point where cumulative degradation cost reaches estimated pit-stop loss.
- **Compatibility cliff field:** `cliff_point_lap`, currently an alias for strategic useful life for existing consumers.
- **Mathematical fastest:** The lowest expected race-time strategy without a separate cliff-risk penalty.
- **Recommended:** The strategy with the best validated risk-adjusted race cost.
- **Safe:** A conservative strategy constrained by strategic useful life.
- **Risky:** A strategy that permits longer stints while explicitly reporting cliff and useful-life exposure.

## Current Status

### Completed

- [x] Added Pirelli benchmark fixtures for the 2022 Bahrain and 2025 Austrian Grands Prix.
- [x] Added stop-count, compound-sequence, pit-window, and useful-life benchmark scoring.
- [x] Added production prediction capture and JSON, CSV, and Markdown reports.
- [x] Fixed historical simulator caching so engines are keyed by technical era and season.
- [x] Added strategy cost decomposition and per-stint overshoot diagnostics.
- [x] Added optional top-five mathematical candidate diagnostics.
- [x] Implemented sustained, piecewise-linear, and hybrid cliff-detector candidates.
- [x] Added synthetic detector and strategy-accounting tests.
- [x] Added a versioned six-race calibration and six-race holdout manifest.
- [x] Added conservative FastF1 lap cleaning, rejection reasons, candidate breakpoints, and SVG review plots.
- [x] Extracted all six calibration races without inspecting the holdout split.

### Calibration Extraction Summary

- 275 accepted dry stints.
- 47 rejected short stints.
- 71 possible cliff candidates.
- 204 possible no-cliff candidates.
- 46 Soft, 107 Medium, and 122 Hard accepted stints.
- 19 high-confidence and 52 medium-confidence automatic cliff candidates.

These automatic candidates are review suggestions, not ground-truth labels. The production detector remains the sustained method, cliff-risk cost remains disabled, and live strategy ranking is unchanged.

## Implementation Phases

### Phase 0: Complete and Freeze the Ground Effect Model Baseline

Build the fixed 2022–2025 model baseline before tuning the tire-cliff detector
or judging strategy quality.

Tasks:

- [x] Add a local, incremental processed-session store with checksum validation.
- [x] Exclude the Ground Effect cache and processed store from Git.
- [x] Remove scheduled Ground Effect training and deployment automation.
- [x] Backfill every expected 2022–2025 Race, Sprint, and FP2 session in bounded local batches.
- [ ] Freeze any intended feature-engineering and model-training implementation changes.
- [x] Train one Ground Effect candidate from the complete local store.
- [ ] Commit only the resulting model, feature list, and training metadata for deployment.
- [x] Record the artifact hash, training timestamp, 2025-12-31 data cutoff, session coverage, and per-year row counts.

The Pirelli fixtures and longest-stint evidence are not ML training labels.
Manual calibration-label review may proceed while the local data backfill is
running, but Phase 2 model-prediction capture must use the newly trained,
frozen artifact. If the artifact or training implementation changes later,
all dependent prediction captures are invalidated and must be regenerated.

Exit criteria:

- Local coverage reports no missing expected sessions or races.
- The trainer refuses partial coverage and records complete 2022–2025 provenance.
- The deployed artifact is fixed before detector parameter tuning begins.
- No holdout result has influenced model selection or training decisions.

### Phase 1: Prioritize and Validate Calibration Reviews

Create a manageable, balanced manual-review set before tuning any detector.

Tasks:

- [x] Rank calibration stints by review value and data quality.
- [x] Include every high-confidence cliff candidate.
- [x] Sample medium-confidence candidates across races and compounds.
- [x] Include clean no-cliff examples across races and compounds.
- [x] Avoid allowing one circuit or compound to dominate the review set.
- [x] Add supported review labels:
  - `confirmed_cliff`
  - `confirmed_no_cliff`
  - `rejected`
  - `pending`
- [x] Require a reviewed cliff lap for `confirmed_cliff`.
- [x] Forbid a cliff lap for `confirmed_no_cliff`.
- [x] Require notes or a rejection reason for `rejected`.
- [x] Validate that reviewed cliff laps fall within the observed tire-age range.
- [x] Export a versioned reviewed-label artifact with race, driver, stint, compound, source, and review metadata.

Exit criteria:

- The review set covers all six calibration races and all three dry compounds.
- Labels pass schema and range validation.
- Ambiguous examples are explicitly rejected instead of receiving invented labels.
- The reviewed artifact and review policy are frozen before detector tuning.

Frozen calibration checkpoint:

- Artifact: `calibration_work/fastf1_cliff_calibration/reviewed_labels.json`
- Schema version: 1
- Labels: 79 total; 33 confirmed cliffs, 34 confirmed no-cliffs, 12 rejected
- SHA-256: `dd2913c69c091545a33e4dc945f0032ed300c7c0bc598581d0f1805d9f8f2c2d`
- Holdout races inspected: 0

### Phase 2: Capture Matching Model Predictions

Generate model curves that correspond to the reviewed historical evidence.

Tasks:

- [x] Map each reviewed stint to season, event, circuit, driver, team, compound, and historical weather context.
- [x] Capture the model artifact name, training date, training cutoff, technical era, and input provenance.
- [x] Generate the predicted degradation curve over the same tire-age interval as the observed stint.
- [x] Store observed and predicted values in a comparison-ready artifact.
- [x] Flag unavailable drivers, teams, mappings, or weather inputs rather than silently substituting them.
- [x] Record whether an evaluated race overlaps an artifact's training data.

Exit criteria:

- Every usable reviewed label has a traceable matching prediction.
- Missing mappings are explicit.
- Calibration reports distinguish production diagnostics from leakage-free evaluations.

### Phase 3: Calibrate Cliff Detection

Compare detector behavior using calibration data only.

Candidate methods:

1. Existing sustained slope-and-curvature detector.
2. Piecewise-linear change-point detector.
3. Hybrid piecewise candidate with persistence and baseline-delta confirmation.

Metrics:

- Median and mean absolute cliff-lap error.
- Early- and late-detection rates.
- False-cliff rate on confirmed no-cliff stints.
- Missed-cliff rate on confirmed cliff stints.
- Error by compound, race, circuit type, and temperature range.
- Confidence calibration.

Tasks:

- [x] Implement calibration-only classification, timing, bias, confidence, and segmented evaluation reports.
- [x] Define a bounded parameter search for all three methods.
- [x] Tune parameters only against frozen calibration labels.
- [ ] Prefer simpler settings when performance is materially equivalent.
- [x] Select a detector only if it improves cliff timing without creating excessive false early cliffs. No searched configuration passed the initial recall and false-cliff gates, so production remains unchanged.
- [ ] Freeze the detector method, parameters, confidence rules, and evaluation code.

Exit criteria:

- The selected configuration beats or justifiably retains the current detector.
- Cost and classification metrics are reproducible.
- No holdout race has been downloaded, plotted, or used for a parameter decision.

### Phase 4: Evaluate the Untouched Holdout Split

After calibration is frozen, extract and evaluate the six holdout races.

Tasks:

- [ ] Run the same FastF1 cleaning and review process without changing its rules.
- [ ] Review holdout stints using the frozen labeling policy.
- [ ] Generate matching model predictions with complete provenance.
- [ ] Run the frozen detector exactly once for the primary holdout result.
- [ ] Report overall and segmented metrics.
- [ ] Document anomalies such as safety cars, damage, traffic, or weather changes.

Exit criteria:

- At least five leakage-free holdout races are usable.
- At least three circuit categories are represented.
- Calibration and holdout results are reported separately.
- Any later exploratory tuning is reported as a new experiment, not as the original holdout score.

### Phase 5: Decide Whether the ML Curves Need Adjustment

Use the observed-versus-predicted comparison to locate the source of error.

Decision rules:

- If observed curves are reasonable but strategy ranking is unrealistic, modify strategy policy.
- If the detector misreads otherwise reasonable predicted curves, modify cliff detection.
- Revisit strategic useful life only after pit-loss mappings and cost accounting are validated.
- Retrain or redesign the degradation model only if holdout data shows persistent raw-curve errors, such as incorrect slope, systematically early or late acceleration, compound bias, or incorrect temperature interactions.

Deliverable:

- [x] Diagnose baseline curve shape on the frozen calibration set. The baseline
  retained approximately 9.5% of observed post-cliff slope.
- [x] Reject simple loss-function and late-lap weighting variants that did not
  materially restore cliff shape.
- [x] Screen a first two-stage within-stint degradation candidate. It improved
  slope retention to approximately 18.7%, but failed the recall and
  false-cliff gates and was not advanced to holdout evaluation.
- [x] Refine degradation-target cleaning and the additive pace/degradation
  architecture without inspecting the holdout split. The refined candidate
  improved slope retention to 32.8% and sustained recall to 69.7%, but its
  50% false-cliff rate still failed the frozen calibration gate.
- [x] Update the existing Active Aero training job to train both regressors
  from one processed-data update and validate/commit the pair atomically.
- [ ] Freeze a candidate only after it passes calibration gates and full
  chronological walk-forward validation.

### Phase 6: Add Risk-Adjusted Strategy Recommendations

Keep expected physical race cost separate from recommendation risk.

Proposed per-lap penalty after a medium- or high-confidence cliff:

```text
overshoot = max(0, effective_tire_age - performance_cliff_lap)

cliff_risk_cost =
    confidence_weight
    × (linear_weight × overshoot + quadratic_weight × overshoot²)
```

Tasks:

- [ ] Preserve mathematical fastest without a cliff-risk penalty.
- [ ] Add risk-adjusted recommended strategy behind a disabled feature flag.
- [ ] Keep cliff-risk cost separate from predicted degradation cost.
- [ ] Apply no penalty when no reliable cliff is detected.
- [ ] Keep Safe constrained by strategic useful life.
- [ ] Keep Risky available while prominently reporting overshoot.
- [ ] Re-score leading candidates under weather, pit-loss, and degradation uncertainty.
- [ ] Prefer a robust strategy when its expected cost is close to a fragile nominal winner.

Required output fields:

- `degradation_cost_sec`
- `pit_loss_cost_sec`
- `traffic_cost_sec`
- `weather_mismatch_cost_sec`
- `cliff_risk_cost_sec`
- `expected_total_delta_sec`
- `risk_adjusted_total_delta_sec`
- `max_performance_cliff_overshoot_laps`
- `max_useful_life_overshoot_laps`
- `recommendation_reason`

Exit criteria:

- Cost components sum exactly to their reported totals.
- Mathematical fastest is unchanged when risk scoring is enabled.
- Recommended can differ from mathematical fastest for a numerically explainable reason.
- Runtime remains within the Render request timeout.

### Phase 7: Improve Pit Windows and Explanations

Tasks:

- [ ] Return earliest, target, and latest pit laps instead of only one fixed lap.
- [ ] Return numeric point, lower, and upper fields for tire cliff and useful life.
- [ ] Display both estimates in the UI as `12 ± 1 lap`, with the range limited to one lap below and above the point estimate.
- [ ] Base window width on cliff confidence and degradation acceleration.
- [ ] Ensure future stints remain feasible at each edge of the window.
- [ ] Generate explanations strictly from calculated numeric fields.
- [ ] Report the time tradeoff between extra pit loss and reduced degradation or cliff exposure.

Example:

```text
The recommended two-stop strategy adds 21.5 seconds of pit loss but
saves 25.7 seconds of degradation exposure and avoids 11 laps beyond
the Medium performance cliff.
```

### Phase 8: Expand External Strategy Evaluation

Continue using Pirelli as an expert comparison rather than a target.

Tasks:

- [x] Expand beyond the two-race pilot to representative high-, medium-, and low-degradation races.
- [x] Include street, high-speed, hot, cool, and interrupted race contexts.
- [x] Store event-specific Pirelli C1-C5 nominations.
- [ ] Extract FastF1 actual stint sequences and pit laps.
- [ ] Report stop-count match, sequence match, top-three coverage, pit-window error, and useful-life bias.
- [ ] Create as-of-race model snapshots and contemporaneously available weather inputs for leakage-free strategy evaluation.

### Phase 9: API and Frontend Rollout

Tasks:

- [ ] Add the new recommendation as an additional API field first.
- [ ] Preserve existing response fields during the transition.
- [ ] Present mathematical fastest, recommended, safe, and risky as distinct concepts.
- [ ] Display cliff confidence, overshoot, cost tradeoffs, and pit windows in the existing UI design language.
- [ ] Keep the current recommendation as the default until all evaluation gates pass.
- [ ] Add a rollback flag that restores the unchanged mathematical ranking.

## Test Plan

### Unit and Regression Tests

- [ ] No-cliff curve returns `null`.
- [ ] Known synthetic breakpoint is detected.
- [ ] Confidence rises with stronger persistent acceleration.
- [ ] Cliff-risk penalty is zero before the cliff and increases after it.
- [ ] Safe respects strategic useful life.
- [ ] Risky reports overshoot.
- [ ] Pit windows stay within race and stint constraints.
- [ ] Existing wet-weather and compound rules continue to pass.
- [ ] Historical simulations continue using season-specific engine caching.
- [ ] Existing frontend response fields remain compatible.

### Evaluation Gates

Do not make the risk-adjusted recommendation the UI default until:

- At least five leakage-free holdout races have been evaluated.
- At least three circuit categories are represented.
- High-confidence cliff predictions outperform the current detector.
- False early-cliff behavior is not materially worse.
- Strategy accounting tests pass exactly.
- Clean-race top-three Pirelli strategy coverage does not materially regress.
- No holdout race influenced detector or penalty parameters.
- Simulation runtime remains viable on Render.

## Immediate Next Steps

1. [x] Preserve both raw and monotonic degradation-model curves in
   calibration captures; keep monotonic curves for strategy accounting and UI
   continuity.
2. [x] Diagnose whether raw model outputs separate confirmed-cliff and
   confirmed-no-cliff stints. They currently do not.
3. [x] Screen conditional sustained-regime sample weighting without using
   detector labels as regression targets. Reject it as the default because
   false cliffs and useful-life shortfalls materially worsened.
4. [ ] Test a stint-balanced target or auxiliary slope-change objective that
   can improve cliff/no-cliff separation without indiscriminately increasing
   late-stint degradation.
5. [ ] Re-capture calibration curves and require improved separability before
   expanding another detector grid.
6. [ ] Redesign sustained confirmation around the improved signal and require
   a lasting post-transition regime.
7. [ ] Freeze the model, detector, and thresholds before touching holdout
   data.

## Expected New or Updated Files

```text
implementation_plan.md                 # Canonical implementation plan
benchmarks/                            # Pirelli fixtures and split manifests
cliff_reference.py                     # FastF1 cleaning and reference construction
scripts/prioritize_cliff_reviews.py    # Balanced manual-review queue
scripts/validate_cliff_reviews.py      # Review validation and versioned export
scripts/capture_cliff_predictions.py   # Matching model predictions
scripts/evaluate_tire_cliffs.py        # Calibration and holdout reports
tire_life_analysis.py                  # Detector methods and confidence
sim_engine.py                          # Risk-adjusted strategy ranking
strategy_benchmark.py                  # Pirelli and race-level metrics
tests/                                 # Review, detector, accounting, and regression tests
frontend/src/                          # Validated strategy presentation
```

## Decision Log

- [x] Keep strategic useful life unchanged during the first cliff-calibration cycle.
- [x] Use cleaned FastF1 stints for cliff evidence.
- [x] Use Pirelli for external strategy context, not direct optimization.
- [x] Keep calibration and holdout races strictly separate.
- [x] Preserve mathematical fastest as a separate strategy.
- [x] Add risk-adjusted recommendations behind a feature flag.
- [x] Keep the live UI behavior unchanged until evaluation gates pass.
- [ ] Select and freeze the final cliff detector.
- [ ] Select and freeze cliff-risk penalty parameters.
- [x] Decide whether the underlying degradation model requires retraining.
      The current candidate still attenuates post-cliff shape and does not
      separate cliff existence sufficiently for detector-only tuning.
