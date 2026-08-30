# Tire Cliff and Strategy Plan — SparkNotes

This is the plain-language overview. See [`implementation_plan.md`](implementation_plan.md) for the canonical implementation status, validation gates, and rollout sequence.

## What Is Wrong?

The model predicted short strategic useful lives in Austria, but the strategy engine still recommended much longer one-stop stints.

This does not automatically mean the machine-learning model is wrong. The strategy engine currently:

- Uses the predicted degradation curves to calculate race time.
- Allows the optimal strategy to exceed useful life up to fixed compound limits.
- Uses useful life mainly to constrain the safe strategy.
- Does not use the physical performance cliff to select the optimal strategy.

## What Will Change?

### 1. Explain Every Strategy Score

Break each strategy's score into:

- Tire-degradation cost.
- Pit-stop loss.
- Traffic cost.
- Weather-related cost.
- Laps beyond the performance cliff.
- Laps beyond strategic useful life.

This will show exactly why a one-stop or two-stop strategy wins.

### 2. Improve Cliff Validation

Use cleaned historical FastF1 stints to identify where lap-time degradation actually began accelerating.

The calibration data will exclude:

- Pit-in and pit-out laps.
- Safety-car and red-flag laps.
- Wet laps during initial dry-weather testing.
- Obvious traffic and lap-time outliers.

### 3. Compare Cliff Calculations

Test three methods:

1. The current slope-and-curvature detector.
2. A piecewise-linear breakpoint detector.
3. A hybrid of both methods.

The best method will be chosen using separate calibration and holdout races.

### 4. Add a Risk-Adjusted Recommendation

Keep the pure mathematical fastest strategy, but add a recommended strategy that receives a gradually increasing risk penalty when it runs beyond a reliable performance cliff.

- Mathematical fastest: raw predicted race time.
- Recommended: fastest after accounting for cliff risk.
- Safe: stays within strategic useful life.
- Risky: allows longer stints but clearly reports the exposure.

### 5. Improve Strategy Suggestions

Each recommendation will explain:

- Why it was selected.
- How much time it gains or loses.
- How many laps it spends beyond the cliff.
- Why another pit stop is or is not worthwhile.
- The earliest, target, and latest suggested pit laps.

## What Will Not Change Initially?

- The strategic useful-life formula.
- The underlying machine-learning model.
- The existing live UI recommendation.
- Model training data.

These will only change if broader testing shows a consistent problem.

## Current Progress

Completed:

- Strategy cost and overshoot diagnostics.
- Optional top-five candidate reporting.
- Piecewise-linear and hybrid detector candidates.
- Synthetic cliff and strategy-accounting tests.
- Pirelli benchmark support for the new diagnostics.
- A six-race calibration and six-race holdout FastF1 manifest.
- Conservative historical stint cleaning with rejection reasons.
- Automatic breakpoint candidates and SVG manual-review plots.
- An eight-race Pirelli calibration benchmark covering the six FastF1
  calibration races plus the original two pilot races.

All six calibration races have now been extracted: 275 accepted dry stints, 71
possible cliffs, 204 possible no-cliff stints, and 47 rejected short stints.
A balanced set of 79 stints has been manually reviewed and frozen: 33 confirmed
cliffs, 34 confirmed no-cliffs, and 12 rejected ambiguous traces. The live
strategy ranking has not changed.

The next step is completing and freezing the 2022–2025 Ground Effect baseline
before capturing matching model predictions. Holdout races remain untouched
until calibration is complete.

## Implementation Order

1. Add strategy cost and overshoot diagnostics.
2. Add synthetic cliff-calculation tests.
3. Extract and clean historical FastF1 stints.
4. Split races into calibration and untouched holdout sets.
5. Compare and calibrate cliff detectors.
6. Add the risk-adjusted recommendation behind a disabled feature flag.
7. Evaluate it against FastF1 and Pirelli evidence.
8. Update the UI only after the new method passes validation.

## How Will We Know It Works?

Before changing the live recommendation:

- Evaluate at least five leakage-free holdout races.
- Include at least three circuit types.
- Reduce incorrect early and late cliff detections.
- Keep all strategy cost calculations internally consistent.
- Avoid reducing top-three strategy coverage.
- Keep simulation time within Render's timeout.

## Main Principle

First determine whether the degradation curve or the strategy policy is causing the unrealistic recommendation. Adjust the cliff calculation and strategy policy only with evidence, and retrain the machine-learning model only if its raw degradation curves are consistently wrong.
